"""Single-model planning + OMLX residency sweep for the GUI.

The user routes every stage (planning included) to a model of their choice;
this module produces one validated plan with that model — streaming the
plan's steps into the live tail — and sweeps non-selected resident models
off OMLX backends before a run (one model at a time).
"""

import json
import time
import urllib.error
import urllib.request

BUILTIN = '(built-in structural judge)'
from concurrent.futures import ThreadPoolExecutor, as_completed

from hart import context as ctxmod
from hart import providers
from hart.agents.planner import validate_plan
from hart.config import Model
from hart.protocol import extract_json


class PlanResult:
    def __init__(self, model_name, plan, self_check, calls=1, wall=0.0, error=None):
        self.model_name = model_name
        self.plan = plan or {"steps": []}
        self.self_check = self_check
        self.calls = calls
        self.wall = wall
        self.error = error

    @property
    def n_steps(self):
        return len(self.plan.get("steps", []))

    @property
    def valid(self):
        return bool(self.self_check.get("ok"))

    def to_dict(self):
        return {"model": self.model_name, "plan": self.plan,
                "self_check": self.self_check, "calls": self.calls,
                "wall": self.wall, "error": self.error}


def _omlx_root(model):
    """OMLX serves its admin API at the root, not under /v1."""
    return model.base_url.rsplit("/v1", 1)[0]


def ensure_single_model(models, on_event=None):
    """One-model-at-a-time preflight for OMLX backends (same lesson the
    benchtest harness learned the hard way): OMLX serves whichever weights
    are RESIDENT and answers 507 'projected memory would exceed the ceiling'
    when a request names a model it can't load beside the resident one.
    Before a run: GET /admin/api/stats, POST /unload for every resident that
    is not one of the models this run will use. Non-OMLX backends are
    untouched."""
    import urllib.request
    import urllib.error
    kept_ids = {m.model for m in models if m.provider == "openai"}
    swept = False
    for m in {mm.name: mm for mm in models}.values():
        if m.provider != "openai" or "127.0.0.1" not in m.base_url:
            continue
        root = _omlx_root(m)
        try:
            with urllib.request.urlopen(f"{root}/admin/api/stats",
                                        timeout=6) as r:
                stats = json.loads(r.read() or b"{}")
        except Exception:
            continue          # not an OMLX-style backend — nothing to do
        resident = [x.get("id") for x in
                    ((stats.get("active_models") or {}).get("models") or [])
                    if x.get("id")]
        for mid in resident:
            if mid in kept_ids:
                continue
            try:
                req = urllib.request.Request(
                    f"{root}/v1/models/{mid}/unload", method="POST")
                urllib.request.urlopen(req, timeout=120).close()
                swept = True
                if on_event:
                    on_event("connection",
                             f"unloaded resident model {mid} on {m.name} — "
                             "one model at a time (frees its memory)")
            except Exception as e:
                if on_event:
                    on_event("connection", f"unload {mid}: {e}")
    return swept


def plan_with_model(goal, workdir, model, feedback=None, max_tokens=16384,
                    on_event=None):
    """Run one planning pass on `model`. Returns a PlanResult (no file writes —
    the caller decides which plan to keep)."""
    parts = [f"GOAL:\n{goal}",
             "REPOSITORY INVENTORY:\n" + ctxmod.inventory(workdir)]
    if feedback:
        parts.append("A reviewer asked you to REVISE the plan. Address this "
                     "feedback directly:\n" + json.dumps(feedback, indent=1))
    parts.append("Produce the plan now.")
    msgs = [
        {"role": "system", "content": ctxmod.system_prompt("planning", model.effort)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    t0 = time.time()
    calls = 0
    try:
        text, _rec = providers.chat(model, msgs, max_tokens=max_tokens)
        calls += 1
        plan = extract_json(text)
        ok, problems = validate_plan(plan)
        if not ok:
            msgs = msgs + [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content":
                    "That plan was invalid: " + "; ".join(problems[:6]) +
                    ". Reply with a corrected plan JSON object only."},
            ]
            text, _rec = providers.chat(model, msgs, max_tokens=max_tokens)
            calls += 1
            plan = extract_json(text)
            ok, problems = validate_plan(plan)
        plan = plan or {"steps": []}
        plan["self_check"] = {"ok": ok, "problems": problems[:8]}
        if on_event:
            n = len(plan.get("steps", []))
            on_event(model.name,
                     f"plan ready: {n} step{'s' if n != 1 else ''} "
                     f"(valid={ok}, {calls} call{'s' if calls != 1 else ''}, "
                     f"{round(time.time() - t0, 1)}s)")
            if not ok and problems:
                on_event(model.name, "  plan problems: " + "; ".join(problems[:3]))
            for s in plan.get("steps", []):
                title = s.get("title") or s.get("id", "?")
                acc = s.get("acceptance") or []
                outs = s.get("outputs") or []
                on_event(model.name, f"  step {s.get('id', '·')}: {title}"
                         + (f"  → {', '.join(outs)}" if outs else ""))
                for a in acc[:2]:
                    on_event(model.name, f"     ✓ {a}")
        return PlanResult(model.name, plan, plan["self_check"], calls=calls,
                          wall=round(time.time() - t0, 2))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        msg = f"HTTP {e.code}: {detail or e.reason}"
        if e.code == 507:
            msg += (" — the backend cannot load the requested model beside "
                    "what is already resident. Unload the resident model "
                    "(or pick it for this run) and retry.")
        return PlanResult(model.name, None, {"ok": False, "problems": [msg[:200]]},
                          calls=calls, wall=round(time.time() - t0, 2), error=msg[:300])
    except Exception as e:      # noqa: BLE001 — one model failing shouldn't sink the batch
        return PlanResult(model.name, None, {"ok": False, "problems": [str(e)[:160]]},
                          calls=calls, wall=round(time.time() - t0, 2), error=str(e)[:200])


def make_model(backend, model_id, ctx_tokens=262144):
    """Build a hart Model from a GUI Backend + a discovered model id."""
    return Model(backend.name, backend.provider, backend.base_url,
                 model=model_id, api_key=backend.resolved_api_key,
                 ctx_tokens=ctx_tokens or backend.ctx_tokens)
