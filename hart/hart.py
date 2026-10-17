#!/usr/bin/env python3
"""
hart — a fully agentic, resumable harness for local OpenAI-compatible
backends on Apple Silicon (OMLX / MTPLX / MLX-LM, or any /v1 endpoint).

Design: see DESIGN.md in this folder. Zero dependencies beyond Python 3.9.

Quick start:
  ./hart.py --framework omlx --task "Write hello.html: an HTML page with an <h1>hello</h1>"
  ./hart.py --list-frameworks
  ./hart.py runs
  ./hart.py --workdir hart-run-X --resume        # resume newest interrupted run
"""

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

VERSION = "0.2.0"
CTX_CHARS_PER_TOKEN = 3.6
# Deviation from benchtest QA parity: size threshold 300 (not 500) so small
# but complete pages aren't scored "not usable"; stubs still fail hard.
QA_MIN_BYTES = 300

# ---------------------------------------------------------------------------
# Framework presets — the critical convenience: one flag picks the backend.
# Model ids below are the ids the servers actually serve (verified live).
# ---------------------------------------------------------------------------
FRAMEWORKS = {
    "omlx": {
        "base_url": "http://127.0.0.1:7001/v1",
        "model": "mlx-community--Qwen3.8-27B-8bit", "model_gb": 28,
        "ctx_tokens": 262144,
        "start_hint": "omlx serve --port 7001",
    },
    "mtplx": {
        "base_url": "http://127.0.0.1:7002/v1",
        "model": "mtplx-qwen38-27b-optimized-quality", "model_gb": 28,
        "ctx_tokens": 261000,
        "start_hint": ("mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality "
                       "--context-window 261000 --max-tokens 65536 --reasoning-effort low --port 7002"),
    },
    "mlxserve": {
        "base_url": "http://127.0.0.1:7004/v1",
        "model": "mlx-community/Qwen3.8-27B-8bit", "model_gb": 28,
        "ctx_tokens": 262144,
        "start_hint": "mlx-serve --model mlx-community/Qwen3.8-27B-8bit --serve --port 7004",
    },
    "mlxvlm": {
        "base_url": "http://127.0.0.1:7003/v1",
        "model": "mlx-community/Qwen3.8-27B-8bit", "model_gb": 28,
        "ctx_tokens": 262144,
        "start_hint": ("/opt/homebrew/opt/python@3.14/bin/python3 -m mlx_vlm.server "
                       "--model mlx-community/Qwen3.8-27B-8bit "
                       "--draft-model mlx-community/Qwen3.8-27B-MTP-8bit "
                       "--draft-kind mtp --max-tokens 65536 --port 7003"),
    },
}

PIPELINE = ["connection", "planning", "building", "validating", "summary"]
STAGE_DESC = {
    "connection": "connecting to the backend and probing capabilities",
    "planning": "planning the work with the model",
    "building": "building — writing files, running checks, fixing issues",
    "validating": "validating: deterministic checks + goal-fit review",
    "summary": "writing the final report",
}
LOOP_NUDGE_AT, LOOP_REPLAN_AT, LOOP_FAIL_AT = 3, 4, 5
ACTIONS_PER_STEP = 12  # test-driven steps need room: write → test → read → fix → retest
EFFORT_MAX_TOKENS = {"low": 4096, "medium": 8192, "high": 16384}
# arg type schemas for the JSON-action protocol: (required, optional)
ACTION_SCHEMAS = {
    "write_file": ({"path": str, "content": str}, {}),
    "append_file": ({"path": str, "content": str}, {}),
    "read_file": ({"path": str}, {"start": int, "end": int}),
    "list_files": ({}, {}),
    "run_shell": ({"cmd": str}, {}),
    "done": ({"summary": str}, {}),
}

ABORT = {"flag": False}
CURRENT = {"sock": None}
VERBOSE = False


class RunStopped(Exception):
    pass


class RunFailed(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def now():
    return time.time()


def atomic_write(path, content):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def est_tokens(text, ratio=None):
    return int(len(text) / (ratio or CTX_CHARS_PER_TOKEN))


def file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def workspace_fingerprint(workdir):
    fp = {}
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = [d for d in dirnames if d != ".hart"]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                fp[os.path.relpath(p, workdir)] = (os.path.getsize(p), file_hash(p))
            except OSError:
                pass
    return hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def _loads_lenient(s):
    """json.loads with a strict=False retry — models embed literal newlines
    inside JSON strings (illegal strict JSON, the #1 repair-loop trigger)."""
    for strict in (True, False):
        try:
            return json.loads(s, strict=strict)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def extract_json(text):
    """Graduated parse: whole text → fenced block → balanced span, each with
    the lenient (control-char tolerant) retry."""
    text = (text or "").strip()
    v = _loads_lenient(text)
    if v is not None:
        return v
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        v = _loads_lenient(m.group(1).strip())
        if v is not None:
            return v
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                v = _loads_lenient(text[start:i + 1])
                if v is not None:
                    return v
                start = None
    return None


# ---------------------------------------------------------------------------
# Connection — the only code that talks HTTP (DESIGN.md §3.1)
# ---------------------------------------------------------------------------
def _http(url, payload=None, api_key=None, timeout=1800, method=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers, method=method or ("POST" if payload is not None else "GET"))
    return urllib.request.urlopen(req, timeout=timeout)


def register_sock(resp):
    try:
        CURRENT["sock"] = resp.fp.raw._sock
    except (AttributeError, OSError):
        CURRENT["sock"] = None


def chat_stream(state, messages, max_tokens=None):
    """One streaming completion. Returns (text, call_record). Aborts cleanly
    on SIGTERM (socket shutdown → EOF-with-flag → RunStopped)."""
    cfg = state["backend"]
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    body = {"model": cfg["model"], "messages": messages,
            "temperature": state["budget"].get("temperature", 0.4),
            "max_tokens": max_tokens or state["budget"]["per_call_max_tokens"],
            "stream": True}
    if state["budget"].get("seed") is not None:
        body["seed"] = state["budget"]["seed"]
    ratio = state.get("token_ratio") or CTX_CHARS_PER_TOKEN
    prompt_est = sum(est_tokens(m.get("content", ""), ratio) for m in messages)
    if VERBOSE:
        print(f"[wire] → {len(messages)} msgs, ~{prompt_est} tok est, "
              f"max_tokens {body['max_tokens']}", file=sys.stderr, flush=True)
    try:
        try:
            resp = _http(url, dict(body, stream_options={"include_usage": True}),
                         cfg.get("api_key"))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                resp = _http(url, body, cfg.get("api_key"))
            else:
                raise
        if ABORT["flag"]:
            resp.close()
            raise RunStopped()
        register_sock(resp)

        if VERBOSE:  # pi-style: show the actual request
            for m in messages:
                c = m.get("content", "")
                print(f"[wire→ {m.get('role')}] {c[:2500]}"
                      + (f" …(+{len(c) - 2500} chars)" if len(c) > 2500 else ""),
                      file=sys.stderr, flush=True)

        # in-flight heartbeat: the run is never a black box (always to stderr)
        hb_stop = threading.Event()
        prog = {"tok": 0, "t0": time.time()}
        def _hb():
            hb_stop.wait(45)   # first ping at 45s, then every 60s - liveness
            while not hb_stop.is_set():           # without the 10s flood
                print(f"⏳ model call in flight — {time.time() - prog['t0']:.0f}s, "
                      f"{prog['tok']} tok streamed ({cfg.get('framework') or 'backend'})",
                      file=sys.stderr, flush=True)
                if hb_stop.wait(60):
                    break
        threading.Thread(target=_hb, daemon=True).start()

        t0 = time.time()
        ttft = None
        parts, usage, finish = [], None, None
        err = None
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    if ttft is None:
                        ttft = time.time() - t0
                for ch in chunk.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    tok = delta.get("content") or delta.get("reasoning_content") or ""
                    if tok:
                        prog["tok"] += 1
                        if ttft is None:
                            ttft = time.time() - t0
                    if delta.get("content"):
                        parts.append(delta["content"])
        except (OSError, urllib.error.URLError) as e:
            err = str(e)
        finally:
            hb_stop.set()
            CURRENT["sock"] = None
            try:
                resp.close()
            except OSError:
                pass
        wall = now() - t0
        if ABORT["flag"]:
            raise RunStopped()
        if err and not parts:
            raise ConnectionError(f"stream failed: {err}")

        text = "".join(parts)
        if VERBOSE:  # pi-style: show the actual response
            print(f"[wire←] {text[:4000]}"
                  + (f" …(+{len(text) - 4000} chars)" if len(text) > 4000 else ""),
                  file=sys.stderr, flush=True)
        ptok = (usage or {}).get("prompt_tokens") or prompt_est
        ctok = (usage or {}).get("completion_tokens") or max(len(text.split()), 1)
        # calibrate the chars→token ratio from real usage (recoimprov §3)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        if usage and total_chars > 200:
            state["token_ratio"] = round(
                0.7 * state.get("token_ratio", CTX_CHARS_PER_TOKEN) +
                0.3 * (total_chars / ptok), 2)
        if VERBOSE:
            print(f"[wire] ← {len(text)} chars, finish={finish}, "
                  f"{ctok} tok out", file=sys.stderr, flush=True)
        decode = max(wall - (ttft or 0), 1e-6)
        ctx = state["budget"].get("ctx_tokens") or 262144
        fw_gb = (FRAMEWORKS.get(state["backend"].get("framework") or "")
                 or {}).get("model_gb")
        _tgs = round(ctok / decode, 1) if ctok and decode > 0.05 else None
        rec = {"ts": now(), "ttft": round(ttft, 3) if ttft else None,
               "pp": round(ptok / ttft, 1) if ttft else None,
               "ctx_fill_pct": round(100.0 * ptok / ctx, 1),
               "est_gbps": round(_tgs * fw_gb / 1024, 1)
               if _tgs and fw_gb else None,
               "tgs": round(ctok / decode, 1) if ctok and decode > 0.05 else None,
               "tps": round(ctok / wall, 1) if wall else None,
               "prompt_tokens": ptok, "completion_tokens": ctok,
               "wall": round(wall, 3), "finish": finish, "model": cfg["model"],
               "error": err}
        return text, rec
    except RunStopped:
        raise
    except urllib.error.HTTPError as e:
        raise RunFailed(f"backend HTTP {e.code}: {str(e.read()[:200] if hasattr(e, 'read') else e)}")
    except (ConnectionError, urllib.error.URLError, OSError, socket.timeout) as e:
        rec = {"ts": now(), "ttft": None, "pp": None, "tgs": None, "tps": None,
               "prompt_tokens": None, "completion_tokens": 0, "wall": None,
               "finish": None, "model": cfg["model"], "error": str(e)[:200]}
        if ABORT["flag"]:
            raise RunStopped()
        return "", rec


# ---------------------------------------------------------------------------
# Blackboard (DESIGN.md §2.2 / §4.1)
# ---------------------------------------------------------------------------
class Blackboard:
    def __init__(self, workdir):
        self.workdir = workdir
        self.dir = os.path.join(workdir, ".hart")
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")
        self.events_path = os.path.join(self.dir, "events.jsonl")
        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        self.lock_path = os.path.join(self.dir, "LOCK")

    def acquire_lock(self):
        if os.path.exists(self.lock_path):
            try:
                other = json.load(open(self.lock_path))
                os.kill(other["pid"], 0)
                sys.exit(f"hart: workdir locked by pid {other['pid']} (started "
                         f"{time.strftime('%H:%M', time.localtime(other['ts']))}). "
                         f"Use --workdir <other> or remove {self.lock_path}.")
            except (ProcessLookupError, PermissionError, ValueError, KeyError):
                pass  # stale lock
            except OSError:
                pass
        atomic_write(self.lock_path, json.dumps({"pid": os.getpid(), "ts": now()}))

    def release_lock(self):
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass

    def load(self):
        with open(self.state_path) as f:
            return json.load(f)

    def save(self, state):
        state["saved_at"] = now()
        atomic_write(self.state_path, json.dumps(
            {k: v for k, v in state.items() if not k.startswith("_")}, indent=1))

    def event(self, agent, event, **detail):
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now(), "agent": agent, "event": event,
                                **detail}) + "\n")

    def metric(self, agent, call_index, rec):
        rec.update({"agent": agent, "call": call_index})
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def new_run_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def fresh_state(args, goal, workdir):
    fw = FRAMEWORKS.get(args.framework, {}) if args.framework else {}
    base_url = args.base_url or fw.get("base_url")
    model = args.model or fw.get("model")
    ctx = args.ctx_tokens or fw.get("ctx_tokens") or 262144
    if not base_url or not model:
        sys.exit("hart: --framework NAME or both --base-url and --model are required "
                 "(see --list-frameworks).")
    return {
        "version": 1, "run_id": new_run_id(), "goal": goal, "workdir": workdir,
        "backend": {"base_url": base_url, "model": model,
                    "api_key": args.api_key or "bench", "framework": args.framework},
        "budget": {"max_steps": args.max_steps, "ctx_tokens": ctx,
                   "per_call_max_tokens": (args.per_call_tokens or
                                           EFFORT_MAX_TOKENS.get(args.effort, 8192)),
                   "time_budget": args.time_budget, "temperature": args.temperature,
                   "effort": args.effort, "seed": args.seed,
                   "max_fix_rounds": args.max_fix_rounds,
                   "repair_rounds": args.repair_rounds,
                   "epochs": args.epochs},
        "pipeline": {a: {"status": "pending", "attempts": 0} for a in PIPELINE},
        "plan": None, "compactions": [], "token_ratio": CTX_CHARS_PER_TOKEN,
        "epoch": 1,
        "loop_guard": {"hashes": {}, "last_fp": None, "unchanged": 0, "nudges": 0},
        "observations": [], "rollups": [], "artifacts": [], "fix_mode": None,
        "fix_roundtrips": 0, "steps": 0, "result": None,
    }


# ---------------------------------------------------------------------------
# Protocol (DESIGN.md §2.3)
# ---------------------------------------------------------------------------
def effort_line(effort):
    return {"low": "Reasoning effort: low — think briefly, act promptly.",
            "medium": "Reasoning effort: medium.",
            "high": "Reasoning effort: high — reason carefully before acting."
            }.get(effort, "")


def system_prompt(role, effort):
    base = {
        "planning": (
            "You are the Planning agent of hart, an autonomous build harness. "
            "Decompose the goal into as many steps as the task GENUINELY needs "
            "for a high-quality result — a single focused file may need exactly "
            "1 step; complex goals need more (one coherent, verifiable chunk "
            "per step). Never pad with artificial steps; never collapse "
            "genuinely separate deliverables. Each step MUST define acceptance "
            "criteria — what must be true for the step to count as complete. "
            "No two steps may produce the same output file. Reply with exactly "
            "one JSON object, no prose, no markdown fences:\n"
            '{"steps":[{"id":1,"title":"...","kind":"code|doc|test","detail":"<=500 chars",'
            '"acceptance":["all 7 tetrominoes present","sound plays on line clear"],'
            '"outputs":["tetris.html"],"depends_on":[1]}]}'),
        "building": (
            "You are the Builder agent of hart. Execute the current plan step. "
            "BATCH your work: produce ALL actions the step needs in ONE reply "
            "(writing every file in full). Reply with exactly one JSON object, no "
            "prose, no markdown fences:\n"
            '{"think":"brief private reasoning","step_id":N,"notes":"one line",'
            '"actions":[\n'
            '  {"action":"write_file","args":{"path":"index.html","content":"FULL file"}},\n'
            '  {"action":"run_shell","args":{"cmd":"ls -la"}} ]}\n'
            "A single-action reply {\"action\":\"...\",\"args\":{...}} is also valid. "
            "Actions: write_file {path, content=FULL file, relative path}; "
            "append_file {path, content} — STREAM LARGE FILES TO DISK: for any "
            "file you expect to be substantial (a full HTML app, a spec "
            "suite), write_file the first part, then CONTINUE the same file "
            "with append_file calls of a few thousand characters each (parts "
            "must join seamlessly) instead of ever emitting it whole; "
            "read_file {path, start?, end?}; list_files {}; run_shell {cmd} "
            "(one quick check, 60s max, no installs, no network); "
            "done {summary} — ends the step, include only after the actions. "
            "Never write files outside the workdir."),
        "validating": (
            "You are the Validator agent of hart. Deterministic checks already "
            "ran; review their results plus the artifact heads against the GOAL "
            "and give a verdict. If the goal is not met, provide EXECUTABLE fixes: "
            "write_file actions with the FULL corrected file content, so they can "
            "be applied directly. Reply with exactly one JSON object:\n"
            '{"verdict":"pass|fail","failures":["..."],"fix_actions":[{"action":'
            '"write_file","args":{"path":"...","content":"FULL corrected file"}}]}'),
        "summary": (
            "You are the Summary agent of hart (merged with the Integrator). "
            "Review the whole delivery for coherence and produce: integration "
            "gaps, the project README (markdown), and the final honest report "
            "(markdown: what was built, validation results, how to run it). "
            "Reply with exactly one JSON object:\n"
            '{"gaps":["..."],"readme":"full markdown","report":"full markdown"}'),
    }[role]
    el = effort_line(effort)
    return base + ("\n\n" + el if el else "")


def protocol_repair(messages, err_hint):
    messages = list(messages)
    messages.append({"role": "assistant", "content": "(invalid output)"})
    messages.append({"role": "user",
                     "content": f"PROTOCOL ERROR: {err_hint}. Reply again with exactly "
                                f"one valid JSON object and nothing else."})
    return messages


# ---------------------------------------------------------------------------
# QA gate (ported from benchtest; DESIGN.md §3.4 / §8)
# ---------------------------------------------------------------------------
NODE = shutil.which("node") if (shutil := __import__("shutil")) else None


def js_syntax_ok(js):
    if not NODE or not js.strip():
        return None, None
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js)
        path = f.name
    try:
        p = subprocess.run([NODE, "--check", path], capture_output=True,
                           text=True, timeout=30)
        if p.returncode == 0:
            return True, None
        lines = (p.stderr or "").strip().splitlines()
        return False, (lines[0][:160] if lines else "syntax error")
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    finally:
        os.unlink(path)


def qa_artifact(path):
    """Weighted functionality% + quality% for HTML artifacts.
    Profile-aware: artifacts containing <canvas> are scored as interactive
    (run loop + DOM/canvas checks apply); plain pages are scored without
    those, so a small functional page isn't punished for not being a game."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError:
        return None
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", src, re.S | re.I)
    js = "\n;\n".join(scripts)
    # interactivity may live in <script> blocks OR inline handler attributes
    has_inline_handlers = bool(re.search(r"\son[a-z]+\s*=\s*[\"']", src, re.I))
    interactive = "<canvas" in src.lower()
    checks = [
        ("doctype", bool(re.search(r"<!DOCTYPE html", src, re.I)), 5),
        ("closed document", "</html>" in src.lower(), 10),
        ("has javascript", len(js.strip()) > 50 or has_inline_handlers, 15),
        ("event handlers", bool(re.search(
            r"addEventListener|on(keydown|click|mouse|touch|input)", js, re.I))
         or has_inline_handlers, 15),
        ("run loop", bool(re.search(
            r"requestAnimationFrame|setInterval|setTimeout", js)), 15),
        ("dom/canvas use", "<canvas" in src.lower() or bool(re.search(
            r"getElementById|querySelector", js)), 10),
        ("braces balanced", js.count("{") == js.count("}") and
         js.count("(") == js.count(")"), 10),
        ("substantial", len(src) > QA_MIN_BYTES, 10),
        ("no placeholders", "TODO" not in src and
         "lorem ipsum" not in src.lower(), 5),
    ]
    if not interactive:  # score plain pages on the checks that apply to them
        checks = [c for c in checks
                  if c[0] not in ("run loop", "dom/canvas use", "substantial")]
    notes = [n for n, ok, _ in checks if not ok]
    func = round(100 * sum(w for _, ok, w in checks if ok) / sum(w for _, _, w in checks))
    syn, syn_err = js_syntax_ok(js)
    if syn is False:
        func = min(func, 25)
        notes.append(f"JS syntax error: {syn_err}")
    elif syn:
        notes.append("JS syntax OK")
    funcs = len(re.findall(r"\bfunction\b|=>", js))
    qchecks = [("decomposed", funcs >= 3, 40),
               ("no eval/doc.write", not re.search(r"\beval\s*\(|document\.write", js), 30),
               ("modern decls", bool(re.search(r"\b(const|let)\b", js)), 30)]
    qual = round(100 * sum(w for _, ok, w in qchecks if ok) / sum(w for _, _, w in qchecks))
    return {"qa_func": func, "qa_qual": qual,
            "qa_notes": "; ".join(notes[:6]) or "all checks passed",
            "usable": func >= 90}


def deterministic_checks(artifacts, workdir):
    def check_one(name):
        # anchored outputs (path#section) are legal declarations: the base
        # file must exist and the section must be present in its content -
        # checking the raw name would fail forever on the '#' in the path
        base, _, frag = name.partition("#")
        p = os.path.join(workdir, base)
        if not os.path.isfile(p):
            return {"artifact": name, "check": "exists", "ok": False,
                    "note": "missing"}
        if frag and base.endswith((".html", ".htm")):
            content = ""
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                pass
            if content and not (f'id="{frag}"' in content
                                or f"id='{frag}'" in content
                                or frag in content):
                return {"artifact": name, "check": "section", "ok": False,
                        "note": f"section #{frag} not present in {base}"}
        if base.endswith((".html", ".htm")):
            qa = qa_artifact(p)
            return {"artifact": name, "check": "qa", "ok": qa["usable"], "qa": qa}
        if base.endswith(".py"):
            r = subprocess.run([sys.executable, "-m", "py_compile", p],
                               capture_output=True, text=True, timeout=60)
            return {"artifact": name, "check": "py_compile", "ok": r.returncode == 0,
                    "note": (r.stderr or "ok")[-200:]}
        ok = os.path.getsize(p) > 0
        return {"artifact": name, "check": "nonempty", "ok": ok,
                "note": f"{os.path.getsize(p)} bytes"}

    # independent per-file checks — run concurrently (recoimprov §2)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(check_one, artifacts))


# ---------------------------------------------------------------------------
# Agents (DESIGN.md §3)
# ---------------------------------------------------------------------------
def _say(state, msg):
    """TUI narration (suppressed with --quiet)."""
    if not state.get("_quiet"):
        print(msg, flush=True)


def record(state, agent, rec):
    state["_bb"].metric(agent, state.get("steps", 0), rec)
    state["steps"] = state.get("steps", 0) + 1
    # live per-call log (owner request: see every call + why, like pi's trace)
    if not state.get("_quiet") and not rec.get("error"):
        print(f"[{agent}] call {state['steps']} · "
              f"{rec.get('prompt_tokens')}→{rec.get('completion_tokens')} tok · "
              f"{rec.get('tgs')} tok/s · ttft {rec.get('ttft')}s · "
              f"ctx {rec.get('ctx_fill_pct')}%"
              + (f" · ~{rec.get('est_gbps')} GB/s" if rec.get('est_gbps') else ""),
              flush=True)


def agent_connection(state):
    cfg = state["backend"]
    try:
        with _http(cfg["base_url"].rstrip("/") + "/models",
                   api_key=cfg.get("api_key"), timeout=8) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
    except Exception as e:
        fw = FRAMEWORKS.get(cfg.get("framework") or "", {})
        hint = f"\n  start it: {fw['start_hint']}" if fw else ""
        raise RunFailed(f"backend not reachable at {cfg['base_url']} ({e}).{hint}")
    if cfg["model"] not in ids:
        if len(ids) == 1:
            state["backend"]["model"], adopted = ids[0], True
        else:
            adopted = False
        state["pipeline"]["connection"]["adopted"] = adopted
    state["pipeline"]["connection"].update(status="done", served=ids[:10])
    # Capability probe (recoimprov §9): JSON ability, tokenizer-ratio
    # calibration, and baseline TTFT/TGS — ~2s, prevents 10-min protocol
    # failures discovered at step 3.
    probe_msgs = [
        {"role": "system", "content":
            'Reply with exactly {"ok": true, "count": 7} and nothing else.'},
        {"role": "user", "content": "Go."}]
    ptext, prec = chat_stream(state, probe_msgs, max_tokens=64)
    record(state, "connection", prec)
    pok = extract_json(ptext)
    json_ok = isinstance(pok, dict) and pok.get("ok") is True
    state["probe"] = {"json_ok": json_ok, "ttft": prec.get("ttft"),
                      "tgs": prec.get("tgs")}
    if not json_ok:
        print("hart: ⚠ capability probe failed — the model may not emit valid "
              "JSON reliably; protocol repairs may be frequent", flush=True)
    # Cold-model warm-in: /v1/models answers before the weights are resident.
    # A slow probe TTFT means the 27 GB are still streaming from disk — force
    # full residency now so measured calls don't crawl at 15 tok/s.
    if (state.get("probe", {}).get("ttft") or 0) > 5.0:
        wtext, wrec = chat_stream(state, [{"role": "user", "content": "Warm."}],
                                  max_tokens=96)
        record(state, "connection", wrec)
        print(f"[connection] cold model warmed in "
              f"{wrec.get('wall')}s (ttft {wrec.get('ttft')}s)", flush=True)
    return (f"backend ready — model {cfg['model']} "
            f"(probe: json {'ok' if json_ok else 'UNRELIABLE'}, "
            f"ttft {prec.get('ttft')}s)")


def inventory(workdir, max_chars=2400):
    """Scalable repo inventory for planning: directory-level summary first
    (scales to hundreds of files), then top-level files."""
    SKIP = {".hart", "node_modules", "__pycache__", ".git", ".venv", "venv"}
    dirs = {}
    total = 0
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        rel = os.path.relpath(dirpath, workdir)
        key = "." if rel == "." else rel
        for fn in filenames:
            total += 1
            e = dirs.setdefault(key, [0, 0])
            e[0] += 1
            try:
                e[1] += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    lines = [f"repository: {total} files in {len(dirs)} directories"]
    for d, (n, sz) in sorted(dirs.items()):
        lines.append(f"  {d}/ — {n} files, {sz // 1024} KB")
    top = [f for f in sorted(os.listdir(workdir))
           if os.path.isfile(os.path.join(workdir, f))][:20]
    if top:
        lines.append("top-level files: " + ", ".join(top))
    out = "\n".join(lines)
    return out[:max_chars]



def plan_selfcheck(steps, goal):
    """Deterministic plan validation (recoimprov §5): outputs present, no
    duplicate outputs (steps producing the same file are MERGED — duplicate
    outputs make the Builder rewrite one file repeatedly, the #1 call-churn
    source), count sane. Returns (steps, notes)."""
    notes = []
    for i, s in enumerate(steps, 1):
        s.setdefault("id", i)
        s.setdefault("title", f"step {i}")
        s.setdefault("kind", "code")
        s["detail"] = str(s.get("detail", ""))[:500]
        s.setdefault("acceptance", [])
        if isinstance(s["acceptance"], list):
            s["acceptance"] = [str(a)[:200] for a in s["acceptance"]][:6]
        else:
            s["acceptance"] = [str(s["acceptance"])[:200]]
        s.setdefault("outputs", [])
        s.setdefault("depends_on", [])
        s["done"] = False
        if not s["outputs"]:
            s["outputs"] = [f"step{i}-output.md"]
            notes.append(f"step {i} had no outputs — assigned a default")
    # merge steps sharing a primary output (churn prevention)
    merged, by_out = [], {}
    for s in steps:
        primary = s["outputs"][0]
        if primary in by_out:
            target = by_out[primary]
            target["detail"] = (target["detail"] + " · " + s["detail"])[:500]
            for o in s["outputs"]:
                if o not in target["outputs"]:
                    target["outputs"].append(o)
            target["depends_on"] = sorted(
                set(target.get("depends_on", [])) | set(s.get("depends_on", [])))
            notes.append(f"merged step {s['id']} into step {target['id']} "
                         f"(both produce {primary})")
        else:
            by_out[primary] = s
            merged.append(s)
    steps = merged
    if len(steps) > 7:
        steps = steps[:7]
        notes.append("truncated to 7 steps")
    for i, s in enumerate(steps, 1):
        s["id"] = i
    return steps, notes


def parse_plan_file(path):
    """Load an execution plan from a file: JSON ({steps:[…]}) or a markdown
    checklist (- [ ] item / numbered lines). Enables the two-turn workflow:
    run 1 writes FIXPLAN.md, run 2 executes it. Returns (steps, fmt) or
    (None, None)."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None, None
    try:
        d = json.loads(text)
        if isinstance(d, dict) and isinstance(d.get("steps"), list) and d["steps"]:
            return d["steps"], "json"
    except (json.JSONDecodeError, ValueError):
        pass
    steps = []
    for line in text.splitlines():
        t = line.strip()
        m = re.match(r"^[-*]\s+\[( |x|X)\]\s+(.{4,})$", t)
        if m:
            steps.append({"title": m.group(2).strip()[:200], "kind": "code",
                          "detail": "", "outputs": [],
                          "done": m.group(1).lower() == "x"})
            continue
        m = re.match(r"^(?:\d+)[.)]\s+(.{4,})$", t)
        if m and len(steps) < 12:
            steps.append({"title": m.group(1).strip()[:200], "kind": "code",
                          "detail": "", "outputs": [], "done": False})
    return (steps or None), ("markdown" if steps else None)


def fallback_plan(goal):
    """Template plan when the planner is unusable (recoimprov §5)."""
    g = goal.lower()
    if "html" in g:
        return [
            {"id": 1, "title": "Create the HTML page with full structure and content",
             "kind": "code", "detail": goal[:400], "outputs": ["index.html"],
             "depends_on": [], "done": False},
            {"id": 2, "title": "Verify the page renders and is complete",
             "kind": "test", "detail": "open the file structure, confirm all "
             "required elements exist", "outputs": ["verification.txt"],
             "depends_on": [1], "done": False}]
    if ".py" in g or "python" in g:
        return [
            {"id": 1, "title": "Write the Python module", "kind": "code",
             "detail": goal[:400], "outputs": ["main.py"], "depends_on": [],
             "done": False},
            {"id": 2, "title": "Verify it compiles and runs", "kind": "test",
             "detail": "py_compile + quick smoke", "outputs": ["verification.txt"],
             "depends_on": [1], "done": False}]
    return [{"id": 1, "title": goal[:80], "kind": "code", "detail": goal[:400],
             "outputs": ["result.md"], "depends_on": [], "done": False}]


def agent_planning(state):
    bb = state["_bb"]
    goal = state["goal"]
    listing = inventory(state["workdir"]) or "(empty workdir)"
    messages = [
        {"role": "system", "content": system_prompt("planning",
                                                    state["budget"]["effort"])},
        {"role": "user", "content": f"GOAL:\n{goal}\n\nWORKDIR CONTENTS:\n{listing}\n\n"
                                    f"Produce the plan JSON now."},
    ]
    for attempt in range(3):
        text, rec = chat_stream(state, messages, max_tokens=4096)
        record(state, "planning", rec)
        plan = extract_json(text)
        steps = (plan or {}).get("steps")
        if isinstance(steps, list) and 1 <= len(steps) <= 7:
            steps, notes = plan_selfcheck(steps, goal)
            state["plan"] = {"steps": steps, "created_by": "planning",
                             "revisions": state["plan"]["revisions"] + 1
                             if state.get("plan") else 0}
            for n in notes:
                bb.event("planning", "plan_note", type="plan_note", note=n)
            state["pipeline"]["planning"].update(status="done")
            bb.event("planning", "plan_created", steps=len(steps))
            if not state.get("_quiet"):
                print(f"[planning] plan ({len(steps)} steps):", flush=True)
                for s in steps:
                    print(f"  {s['id']}. {s['title']} → {', '.join(s['outputs'])}",
                          flush=True)
            return f"plan: {len(steps)} steps" + (f" ({len(notes)} notes)"
                                                  if notes else "")
        if attempt < 2:
            messages = protocol_repair(messages, "output was not the required "
                                                 "steps JSON")
    state["plan"] = {"steps": fallback_plan(state["goal"]),
                     "created_by": "fallback", "revisions": 0}
    state["pipeline"]["planning"].update(status="done", degraded=True)
    return "plan degraded to template (planner output unusable)"


def build_context(state, step=None, extra=()):
    obs = state["observations"]
    recent = obs[-5:]
    rolled = state["rollups"][-10:]
    parts = [f"GOAL:\n{state['goal']}",
             "PLAN:\n" + "\n".join(
                 f"  {s['id']}. [{'x' if s['done'] else ' '}] {s['title']} — {s['detail']}"
                 + (f" (outputs: {', '.join(s['outputs'])})" if s["outputs"] else "")
                 for s in state["plan"]["steps"])]
    if state.get("fix_mode"):
        parts.append("FIX MODE — the Validator found problems; address exactly these:\n" +
                     json.dumps(state["fix_mode"]["fix_actions"], indent=1))
    if step:
        parts.append(f"CURRENT STEP {step['id']}: {step['title']} — {step['detail']}")
        if step.get("acceptance"):
            parts.append("STEP ACCEPTANCE CRITERIA (the step is only complete "
                         "when ALL hold):\n" + "\n".join(
                             f"  - {a}" for a in step["acceptance"]))
    # supply current artifacts: full if small, head+tail if large
    supplied = 0
    art_budget = 14000
    art_lines = 200
    if state.get("_slim"):  # 60%-stage slimming: heads only
        art_budget, art_lines = 4000, 40
    for name in state["artifacts"][:6]:
        p = os.path.join(state["workdir"], name)
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue
        if len(lines) <= art_lines and supplied < art_budget:
            parts.append(f"CURRENT {name} ({len(lines)} lines):\n" + "".join(lines))
            supplied += sum(len(l) for l in lines)
        elif supplied < art_budget:
            head = "".join(lines[:40])
            tail = "".join(lines[-20:])
            parts.append(f"CURRENT {name} ({len(lines)} lines — truncated; use "
                         f"read_file for ranges):\n{head}\n…\n{tail}")
            supplied += len(head) + len(tail)
    if rolled:
        parts.append("EARLIER PROGRESS (rolled up):\n" + "\n".join(rolled))
    if recent:
        parts.append("RECENT ACTIONS/OBSERVATIONS:\n" + "\n".join(recent))
    for lr in sorted(state.get("last_results", {}).values(),
                     key=lambda e: e.get("_t", 0)):
        tgt = lr.get("target") or ""
        res = dict(lr["result"])
        content = res.pop("content", None) or res.pop("stdout", None)
        if content:
            parts.append(f"RESULT OF {lr['action']} {tgt}:\n{str(content)[:6000]}")
        else:
            parts.append(f"RESULT OF {lr['action']} {tgt}:\n{json.dumps(res)[:600]}")
    parts.extend(extra)
    parts.append("Reply with the next action JSON now. (Results of reads and "
                 "shell commands are above — do not repeat them.)")
    return [{"role": "system", "content": system_prompt(
        "building", state["budget"]["effort"])},
        {"role": "user", "content": "\n\n".join(parts)}]


def trim_state(state):
    """Bound long-run state growth (days/weeks of work); the full audit
    trail stays in events.jsonl, so nothing is lost — only working memory
    is kept lean."""
    if len(state.get("observations", [])) > 100:
        state["observations"] = state["observations"][-100:]
    if len(state.get("rollups", [])) > 50:
        state["rollups"] = state["rollups"][-50:]
    lg = (state.get("loop_guard") or {}).get("hashes")
    if isinstance(lg, dict):
        for h in [k for k, v in lg.items() if not v]:
            del lg[h]
    rep = (state.get("repair") or {}).get("log")
    if isinstance(rep, list) and len(rep) > 20:
        state["repair"]["log"] = rep[-20:]


def compact_if_needed(state, messages, step=None):
    """Context survival for days/weeks-long runs (owner directive — pi-like
    compaction so the run NEVER exhausts context). Three stages:
      60% — deterministic slimming (drop stale read results, artifact
            supply becomes head-only); free, no model call.
      80% — observation-history rollup via one model call (files are never
            accumulated — always re-read from disk — so history is the
            only context that grows).
      95% — hard guard: fail loudly into a repair cycle rather than send a
            doomed request."""
    ratio = state.get("token_ratio") or CTX_CHARS_PER_TOKEN
    ctx = state["budget"]["ctx_tokens"]

    def _used(msgs):
        return sum(est_tokens(m.get("content", ""), ratio) for m in msgs)

    used = _used(messages)
    if used >= ctx * 0.6:
        lr = state.get("last_results", {})
        if len(lr) > 2:
            for k in sorted(lr, key=lambda k: lr[k].get("_t", 0))[:-2]:
                del lr[k]
        state["_slim"] = True
        messages = build_context(state, step)
        used = _used(messages)
        state["compactions"].append({"at": now(), "stage": "slim",
                                     "used": used, "ctx": ctx})
        _say(state, f"🧹 context slimmed (60% stage): stale results dropped, "
                    f"artifact supply thinned → ~{used} tok")
    if used >= ctx * 0.8 and len(state["observations"]) > 5:
        drop = state["observations"][:-5]
        keep = state["observations"][-5:]
        rollup_call = [
            {"role": "system", "content": "Summarize the following build-progress "
             "observations into at most 5 one-line rollups. Output JSON: "
             '{"rollups":["..."]}.'},
            {"role": "user", "content": "\n".join(drop[-30:])}]
        text, rec = chat_stream(state, rollup_call, max_tokens=600)
        record(state, "building", rec)
        roll = extract_json(text)
        lines = (roll or {}).get("rollups") or [f"(compacted {len(drop)} obs)"]
        state["rollups"].extend(lines if isinstance(lines, list) else [str(lines)])
        state["compactions"].append({"at": now(), "stage": "rollup",
                                     "dropped": len(drop)})
        state["observations"] = keep
        messages = build_context(state, step)
        used = _used(messages)
        _say(state, f"🧹 context rolled up (80% stage): {len(drop)} observations "
                    f"→ {len(state['rollups'])} one-liners → ~{used} tok")
    state.pop("_slim", None)
    if used >= ctx * 0.95:
        raise RunFailed(f"context budget exceeded even after compaction "
                        f"(~{used}/{ctx} tokens) — artifact set too large "
                        f"for the window; reduce file sizes")
    return messages


def validate_action_args(action, args):
    """Type-check args against ACTION_SCHEMAS (required + optional).
    Returns (error_hint_or_None, valid_form_example)."""
    schema = ACTION_SCHEMAS.get(action)
    if schema is None:
        return f"unknown action {action!r}", None
    required, optional = schema
    args = args or {}
    for k in args:
        if k not in required and k not in optional:
            return (f"unexpected arg '{k}' for {action}", None)
    for k, typ in required.items():
        v = args.get(k)
        if not isinstance(v, str) or not v:
            return (f"args.{k} must be a non-empty string", None)
    for k, typ in optional.items():
        if k in args and not isinstance(args[k], typ):
            try:
                args[k] = typ(args[k])  # lenient: "100" → 100
            except (TypeError, ValueError):
                return f"args.{k} must be {typ.__name__}, got {args[k]!r}", None
    example = {a: ("…" if t is str else 1) for a, t in required.items()}
    return None, {"action": action, "args": example}


def execute_action(state, action, args, step_id=None):
    workdir = state["workdir"]
    hint, valid_form = validate_action_args(action, args)
    if hint:
        result = {"error": hint}
        if valid_form:
            result["valid_form"] = valid_form  # let the model self-correct
        return result
    if action == "write_file":
        raw_path = args.get("path", "").split("#", 1)[0].strip()
        if ".." in raw_path or os.path.isabs(raw_path):
            return {"error": f"path must be relative to the workdir (got {raw_path!r})"}
        path = os.path.normpath(os.path.join(workdir, raw_path))
        if not path.startswith(os.path.abspath(workdir) + os.sep):
            return {"error": "path escapes workdir"}
        content = args.get("content")
        if len(content) < 10:
            return {"error": "content is empty or < 10 chars — provide the actual file"}
        if re.match(r"^(this file|the file|below is)\b", content.strip().lower()):
            return {"error": "content looks like a DESCRIPTION of the file — "
                             "provide the actual file content"}
        if len(content) > 200_000:
            return {"error": "content > 200KB — split the file"}
        key = f"{step_id}:{raw_path}"
        writes = state.setdefault("_step_writes", {})
        writes[key] = writes.get(key, 0) + 1
        if writes[key] == 3:
            state["observations"].append(
                f"⚠ REWRITE CHURN: {raw_path} written {writes[key]}× in this "
                f"step — finalize it instead of rewriting again.")
        os.makedirs(os.path.dirname(path) or workdir, exist_ok=True)
        atomic_write(path, content)
        rel = os.path.relpath(path, workdir)
        if rel not in state["artifacts"]:
            state["artifacts"].append(rel)
        return {"ok": f"wrote {rel} ({len(content)} B)"}
    if action == "append_file":
        raw_path = args.get("path", "").split("#", 1)[0].strip()
        if ".." in raw_path or os.path.isabs(raw_path):
            return {"error": "path must be relative to the workdir"}
        path = os.path.normpath(os.path.join(workdir, raw_path))
        if not path.startswith(os.path.abspath(workdir) + os.sep):
            return {"error": "path escapes workdir"}
        content = args.get("content", "")
        if len(content) < 1:
            return {"error": "content empty"}
        existed = os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        rel = os.path.relpath(path, workdir)
        if rel not in state["artifacts"]:
            state["artifacts"].append(rel)
        return {"ok": f"{'appended' if existed else 'created'} {rel} "
                      f"(+{len(content)} B, total {os.path.getsize(path)} B)"}
    if action == "read_file":
        rp = args.get("path", "").split("#", 1)[0].strip()
        path = os.path.normpath(os.path.join(workdir, rp))
        if not path.startswith(os.path.abspath(workdir) + os.sep) or not os.path.isfile(path):
            return {"error": "file not found in workdir"}
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(int(args.get("start") or 1), 1)
        end = min(int(args.get("end") or len(lines)), len(lines))
        sel = "".join(lines[start - 1:end])
        return {"path": args.get("path"), "total_lines": len(lines),
                "lines": f"{start}-{end}", "content": sel[:8000]}
    if action == "list_files":
        sub = os.path.normpath(os.path.join(
            workdir, (args or {}).get("path") or ""))
        if not sub.startswith(os.path.abspath(workdir) + os.sep):
            return {"error": "path escapes workdir"}
        if os.path.isfile(sub):
            return {"file": os.path.relpath(sub, workdir),
                    "lines": sum(1 for _ in open(sub, errors="replace"))}
        entries = []
        for dirpath, dirnames, filenames in os.walk(sub):
            dirnames[:] = [d for d in dirnames
                           if d not in (".hart", "node_modules", "__pycache__",
                                        ".git")]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                entries.append(f"{os.path.relpath(p, workdir)} "
                               f"({os.path.getsize(p)} B)")
                if len(entries) >= 60:
                    entries.append("… (use read_file / run_shell for more)")
                    break
            break
        return {"files": entries or ["(empty)"]}
    if action == "run_shell":
        cmd = args.get("cmd", "")
        # 2000 chars: real verification one-liners (python -c, grep/find
        # chains) run 200-800 chars - the old 200 cap rejected them and
        # forced the model into extra workaround calls. The dangerous-
        # pattern regex still guards.
        if len(cmd) > 2000 or re.search(
                r"rm\s+-rf\s+[~/]|\bsudo\b|\bdd\b|\bmkfifo\b"
                r"|curl[^|]*\|\s*(ba|z)?sh|\|\s*(ba|z)?sh\b", cmd):
            return {"error": "command rejected by safety policy"}
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=60, cwd=workdir)
            return {"rc": p.returncode, "stdout": p.stdout[-1500:],
                    "stderr": p.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"error": "command timed out after 60s"}
    return {"error": f"unknown action {action!r}"}


def run_action_batch(state, bb, step, act):
    """Normalize single/batch form; type-check; execute in order.
    Returns (done_summary_or_None, n_executed, had_write)."""
    if isinstance(act.get("actions"), list):
        items = act["actions"]
    else:
        items = [{k: v for k, v in act.items() if k in ("action", "args")}]
    done_summary = None
    executed = 0
    had_write = False
    for one in items:
        if not isinstance(one, dict) or "action" not in one:
            state["observations"].append("(batch item was not an action — skipped)")
            continue
        action = one.get("action")
        args_ = one.get("args") or {}
        if action == "done":
            done_summary = (args_ or {}).get("summary", "")
            continue
        result = execute_action(state, action, args_, step_id=step["id"])
        executed += 1
        if action == "write_file" and "ok" in result:
            had_write = True
        state["observations"].append(
            f"step {step['id']}: {action} "
            f"{json.dumps(args_.get('path') or args_.get('cmd') or args_.get('start') or '')[:50]} → "
            f"{json.dumps(result)[:150]}"[:240])
        # retain read/shell results per target, refreshed on re-read — the
        # model never loses content it already saw (anti re-read churn)
        if action in ("read_file", "run_shell") and "error" not in result:
            keep = state.setdefault("last_results", {})
            key = f"{action}:{args_.get('path') or args_.get('cmd')}"
            if key in keep:
                state["observations"].append(
                    f"ℹ {key.split(':', 1)[1]} was already read — its content "
                    f"is in the results section above. Do not re-read.")
            keep[key] = {"action": action,
                         "target": args_.get("path") or args_.get("cmd"),
                         "result": result, "_t": now()}
            for stale in sorted(keep, key=lambda k: keep[k]["_t"])[:-4]:
                del keep[stale]
        if not state.get("_quiet"):
            target = args_.get('path') or args_.get('cmd') or ''
            if "error" in result:
                print(f"⚠ {action} {target}: "
                      f"{str(result.get('error'))[:60]}", flush=True)
            else:
                print(f"🔧 {action} {target} ✓ [s{step['id']}]",
                      flush=True)
        bb.event("building", "action_exec", type="action_exec",
                 action=action, result=str(result)[:120])
        # loop ladder (DESIGN.md §4.3), step-scoped (recoimprov §7)
        h = hashlib.sha1(f"{step['id']}|{action}|"
                         f"{json.dumps(args_, sort_keys=True, default=str)}"
                         .encode()).hexdigest()
        counts = state["loop_guard"]["hashes"]
        counts[h] = counts.get(h, 0) + 1
        n = counts[h]
        if n == LOOP_NUDGE_AT:
            state["loop_guard"]["nudges"] += 1
            state["observations"].append(
                "⚠ LOOP: you have repeated this identical action — choose a "
                "different approach.")
            bb.event("building", "loop_nudge", type="loop_nudge", hash=h[:8])
        elif n >= LOOP_FAIL_AT:
            enter_repair(state, bb, f"identical action repeated {n}× — step "
                         f"{step['id']} is stuck in a loop. Diagnose WHY it "
                         f"repeats and take a different approach.")
            return None, executed, had_write
        elif n == LOOP_REPLAN_AT:
            state["pipeline"]["planning"].update(status="running")
            state["pipeline"]["building"].update(status="pending")
            state["_replan_requested"] = True
            bb.event("building", "loop_replan", type="loop_replan")
            return None, executed, had_write
    return done_summary, executed, had_write


def enter_repair(state, bb, problem):
    """The repair cycle (the owner's core requirement): a Builder-level
    failure becomes a diagnosed retry instead of a dead run. Feeds the exact
    error + recent history back to the model with an explicit diagnosis
    directive; resets the stagnation counters that triggered. Bounded by
    budget.repair_rounds — only exhaustion fails the run."""
    budget = state["budget"].get("repair_rounds", 3)
    state.setdefault("repair", {"cycles": 0, "log": []})
    state["repair"]["cycles"] += 1
    cycle = state["repair"]["cycles"]
    state["repair"]["log"].append({"cycle": cycle, "problem": problem[:200],
                                   "at": now()})
    bb.event("building", "repair_cycle", type="repair", cycle=cycle,
             problem=problem[:200])
    if cycle > budget:
        raise RunFailed(f"repair budget exhausted after {budget} cycles; "
                        f"last problem: {problem}")
    diagnosis = [
        f"⚠ REPAIR CYCLE {cycle}/{budget} — the pipeline detected a problem:",
        f"PROBLEM: {problem}",
        "Diagnose the root cause from the observations above, then take a "
        "DIFFERENT approach (different actions, different file structure, or a "
        "smaller increment). Do not repeat what already failed.",
    ]
    # stagnation counters served their purpose; the repair gets a clean slate
    state["loop_guard"]["unchanged"] = 0
    state["loop_guard"]["last_fp"] = None
    for h in list(state["loop_guard"]["hashes"]):
        state["loop_guard"]["hashes"][h] = 0
    state["observations"].extend(diagnosis)
    return f"repair cycle {cycle}: {problem[:120]}"


def agent_oneshot_build(state, bb, step):
    """pi-parity fast path for single-artifact tasks: ONE call that writes the
    complete file, no ceremony. On protocol failure, falls back to the full
    loop (oneshot is never retried)."""
    out = step["outputs"][0]
    messages = [
        {"role": "system", "content":
            "You are an expert developer. Reply with exactly one JSON object, "
            "no prose, no markdown fences:\n"
            '{"action":"write_file","args":{"path":"' + out +
            '","content":"<the COMPLETE, working, polished file>"}}'},
        {"role": "user", "content": f"GOAL: {state['goal']}\n"
         f"Write the complete file as {out}. Quality bar: it must work "
         f"perfectly when opened directly in a browser."}]
    for attempt in range(2):
        text, rec = chat_stream(state, messages,
                                max_tokens=max(
                                    state["budget"]["per_call_max_tokens"], 12288))
        record(state, "building", rec)
        act = extract_json(text)
        if isinstance(act, dict) and act.get("action") == "write_file":
            r = execute_action(state, "write_file", act.get("args") or {},
                               step_id=step["id"])
            if "ok" in r:
                gate = step_gate(state, step)
                if gate:
                    # one-shot didn't meet the quality bar — full loop takes
                    # over with the failure as its starting context
                    state["oneshot"] = False
                    state["observations"].append(
                        f"⚠ oneshot output failed the quality gate: {gate} — "
                        f"continuing with the full build loop.")
                    state["pipeline"]["building"].update(status="running")
                    return None
                step["done"] = True
                state["rollups"].append(f"oneshot ✓ {out}")
                bb.event("building", "oneshot_done", type="oneshot",
                         artifact=out)
                state["pipeline"]["building"].update(status="done")
                return f"oneshot build complete ({out})"
        if attempt == 0:
            messages = protocol_repair(
                messages, "not the required write_file JSON — the content must "
                "be the complete file")
    state["oneshot"] = False  # fall back to the full loop
    state["pipeline"]["building"].update(status="running")
    return None


def step_gate(state, step):
    """Quality gate for step completion (owner directive): a step is only
    accepted when its declared outputs exist and pass the deterministic
    checks. Returns rejection reasons, or None if accepted."""
    if not step.get("outputs"):
        return None  # no outputs declared — nothing to gate
    checks = deterministic_checks(step["outputs"], state["workdir"])
    failed = [c for c in checks if not c["ok"]]
    if failed:
        reasons = []
        for c in failed:
            if c["check"] == "qa":
                reasons.append(f"{c['artifact']}: QA {c['qa']['qa_func']}% below "
                               f"usable ({c['qa']['qa_notes'][:80]})")
            else:
                reasons.append(f"{c['artifact']}: {c['check']} failed "
                               f"({str(c.get('note', ''))[:60]})")
        return "; ".join(reasons)[:300]
    # acceptance criteria are surfaced to the Validator for goal-fit review
    return None


def new_epoch(state, bb, reason):
    """pi-style unattended continuation: when a budget is exhausted, instead
    of dying, checkpoint everything, compact context, reset the per-epoch
    budgets, and keep going in a fresh epoch. Bounded by budget.epochs, with
    a no-progress guard so it can't spin for days producing nothing."""
    epochs = state["budget"].get("epochs", 1)
    nxt = state.get("epoch", 1) + 1
    if nxt > epochs:
        raise RunFailed(f"epoch budget exhausted ({epochs} epochs); "
                        f"last blocker: {reason}")
    # no-progress guard: two consecutive epochs without a workspace change
    fp = workspace_fingerprint(state["workdir"])
    if fp == state.get("_epoch_fp"):
        state["_epoch_noprog"] = state.get("_epoch_noprog", 0) + 1
    else:
        state["_epoch_noprog"] = 0
    state["_epoch_fp"] = fp
    if state["_epoch_noprog"] >= 2:
        raise RunFailed("no progress across consecutive epochs — stopping "
                        "rather than spinning")
    state["epoch"] = nxt
    # fresh per-epoch budget slate
    state["steps"] = 0
    state["started_at"] = now()
    state["_budget_flags"] = set()
    state["_rescue_done"] = False
    state["loop_guard"]["hashes"] = {}
    state["loop_guard"]["unchanged"] = 0
    state["repair"] = {"cycles": 0, "log": []}
    state["observations"] = state["observations"][-5:]  # hard context trim
    trim_state(state)
    state["compactions"].append({"epoch": state["epoch"],
                                 "reason": reason[:120]})
    bb.event("orchestrator", "new_epoch", type="epoch", epoch=state["epoch"],
             reason=reason[:120])
    bb.save(state)
    if not state.get("_quiet"):
        print(f"[epoch {state['epoch']}/{epochs}] continuing — {reason[:90]}",
              flush=True)


def agent_building(state):
    bb = state["_bb"]
    plan = state["plan"]
    step = next((s for s in plan["steps"] if not s["done"]), None)
    if step is None:
        state["pipeline"]["building"].update(status="done")
        return "all plan steps complete"
    # pi-parity fast path: single step, single output file → one-shot it
    if (not state.get("oneshot_tried") and len(plan["steps"]) == 1 and
            len(plan["steps"][0].get("outputs", [])) == 1 and
            not state.get("fix_mode")):
        state["oneshot_tried"] = True
        state["oneshot"] = True
        state["pipeline"]["building"].update(status="running")
        return agent_oneshot_build(state, bb, plan["steps"][0])
    state["pipeline"]["building"].update(status="running")
    state["_step_writes"] = {}
    calls_here = 0
    step_retries = 0
    while True:
        if ABORT["flag"]:
            raise RunStopped()
        if state["steps"] >= state["budget"]["max_steps"]:
            # Intelligent budget break: instead of a dumb wall, spend ONE
            # final call on forced delivery of everything built so far.
            if state.get("_rescue_done"):
                if state["budget"].get("epochs", 1) > 1:
                    new_epoch(state, bb, f"max_steps "
                              f"{state['budget']['max_steps']} reached "
                              f"(rescue call also used)")
                    continue
                raise RunFailed(f"max_steps exceeded ({state['budget']['max_steps']} "
                                f"model calls) — final delivery call did not complete")
            state["_rescue_done"] = True
            state["observations"].append(
                "⚠ FINAL CALL — budget exhausted. Write the COMPLETE final "
                "version of every artifact NOW from what you have built (no "
                "tests, no reads, no more iteration), then done.")
            if not state.get("_quiet"):
                print("[building] ⚠ budget exhausted — forcing final delivery "
                      "call", flush=True)
            bb.event("building", "rescue_call", type="budget")
        used, total = state["steps"], state["budget"]["max_steps"]
        frac = used / max(total, 1)
        flags = state.setdefault("_budget_flags", set())
        if frac >= 0.55 and "w55" not in flags:
            flags.add("w55")
            state["observations"].append(
                f"⚠ BUDGET {used}/{total} calls: consolidate — batch remaining "
                f"work, stop rewriting whole files, prefer done over perfection.")
        elif frac >= 0.75 and "w75" not in flags:
            flags.add("w75")
            state["observations"].append(
                f"⚠ BUDGET {used}/{total}: finish the deliverable NOW — "
                f"complete files, then done. Skip further polish.")
        elif frac >= 0.9 and "w90" not in flags:
            flags.add("w90")
            state["observations"].append(
                f"⚠ FINAL STRETCH {used}/{total}: next calls must produce the "
                f"final artifact and mark done.")
        if calls_here >= ACTIONS_PER_STEP:
            # Step call-budget exhausted without a done — exactly where a
            # test-driven step (write → test → diagnose → fix → retest)
            # legitimately needs more room: repair with a fresh allowance
            # instead of killing the run.
            if step_retries >= 2:
                raise RunFailed(f"step {step['id']} still incomplete after "
                                f"{step_retries} repair cycles × "
                                f"{ACTIONS_PER_STEP} calls")
            step_retries += 1
            enter_repair(state, bb, f"step {step['id']} used "
                         f"{ACTIONS_PER_STEP} calls without completing. If "
                         f"close, BATCH aggressively: write all files at once "
                         f"and run verification in this reply. If stuck, "
                         f"simplify the approach.")
            calls_here = 0
            continue
        messages = compact_if_needed(state, build_context(state, step), step)
        # what is this call ASKING for - the model-facing purpose, one line
        ask = (step.get('title') or
               step.get('id', '') if isinstance(step, dict) else '')
        last_user = next((m for m in reversed(messages)
                          if m.get('role') == 'user'), None)
        if last_user:
            tail = ' '.join(str(last_user.get('content', '')).split())[:110]
            ask = f"{ask} - {tail}" if ask else tail
        _say(state, f"[building] call {state.get('steps', 0) + 1} · "
                    f"asking: {ask}")
        text, rec = chat_stream(state, messages)
        record(state, "building", rec)
        # transport backoff: consecutive connection failures (backend restart
        # mid-run) otherwise escalate through the ladder and kill an
        # unattended run in ~a minute. Wait, then retry the SAME call.
        if rec.get("error"):
            state["_trans_errs"] = state.get("_trans_errs", 0) + 1
            if state["_trans_errs"] >= 2:
                wait = min(30 * state["_trans_errs"], 120)
                _say(state, f"⏳ connection error x{state['_trans_errs']} - "
                            f"backing off {wait}s before retrying this call")
                bb.event("building", "transport_backoff", wait=wait)
                if ABORT["flag"]:
                    raise RunStopped()
                hb_pause = threading.Event()
                threading.Thread(target=hb_pause.wait, args=(wait,),
                                 daemon=True).start()
                time.sleep(wait)
        else:
            state["_trans_errs"] = 0
        calls_here += 1
        bb.save(state)  # fresh checkpoint per call for live monitoring
        _say(state, f"💾 checkpoint · building · step {step['id']} · "
                    f"call {calls_here} · epoch {state.get('epoch', 1)}")
        act = extract_json(text)
        if not isinstance(act, dict) or not ("action" in act or "actions" in act):
            if rec.get("finish") == "length":
                state["_trunc_streak"] = state.get("_trunc_streak", 0) + 1
                state["observations"].append(
                    "⚠ output truncated at the per-call token cap — the file "
                    "is too large for one call. Use write_file for the first "
                    "chunk, then append_file calls to complete it.")
                # circuit breaker: consecutive maximal rambles mean the model
                # is not emitting protocol actions at all — repair, then fail
                # honestly instead of burning the whole epoch (2h observed)
                if state["_trunc_streak"] == 3:
                    enter_repair(state, bb,
                                 "3 consecutive calls hit the token cap without "
                                 "a parseable action — the model is rambling. "
                                 "Emit ONE small valid action per reply.")
                elif state["_trunc_streak"] >= 6:
                    raise RunFailed(
                        "model cannot emit protocol-compliant actions within "
                        f"the per-call cap ({state['_trunc_streak']} consecutive "
                        "truncations) — model/quant limitation, not recoverable "
                        "by retrying")
                continue
            messages = protocol_repair(messages, "not a valid action/batch JSON")
            state["observations"].append("(protocol repair)")
            bb.event("building", "repair", type="repair")
            continue
        state["_trunc_streak"] = 0
        done_summary, executed, had_write = run_action_batch(
            state, bb, step, act, state["steps"])
        if done_summary is None and not executed and \
                state["pipeline"]["planning"]["status"] == "running":
            return None  # loop ladder sent control back to Planning
        if state.pop("_replan_requested", False):
            # the replan rung fired even though actions executed - the
            # builder is repeating itself at scale, hand control to Planning
            return None
        fp = workspace_fingerprint(state["workdir"])
        # no-progress counts only content-changing work: read/diagnose cycles
        # are legitimate agent behavior, not stagnation
        if fp == state["loop_guard"].get("last_fp") and had_write:
            state["loop_guard"]["unchanged"] += 1
            if state["loop_guard"]["unchanged"] >= 3:
                enter_repair(state, bb,
                             "workspace unchanged across 3 write attempts — the "
                             "writes are not landing or are identical. Re-check "
                             "the exact error messages and correct the content.")
                state["loop_guard"]["unchanged"] = 0
                return "repair cycle entered (no-progress)"
        else:
            state["loop_guard"]["unchanged"] = 0
        state["loop_guard"]["last_fp"] = fp
        if done_summary is not None or (act.get("action") == "done"):
            # step-completion gate: done is accepted only when the step's
            # outputs pass deterministic checks (quality-gated transitions)
            gate = step_gate(state, step)
            if gate:
                state["observations"].append(
                    f"⚠ STEP {step['id']} NOT ACCEPTED — outputs failed the "
                    f"quality gate: {gate}. Fix and mark done again.")
                bb.event("building", "step_rejected", type="step_gate",
                         step=step["id"], reasons=gate[:200])
                continue
            step["done"] = True
            summary = done_summary or act.get("args", {}).get("summary", "")
            state["observations"].append(
                f"step {step['id']} done: {summary[:120]}")
            # free step-boundary rollup (recoimprov §1d)
            state["rollups"].append(f"step {step['id']} ✓ {summary[:100]}")
            bb.event("building", "step_done", step=step["id"])
            state["fix_mode"] = None
            state["_step_writes"] = {}
            trim_state(state)
            nxt = next((s for s in plan["steps"] if not s["done"]), None)
            if nxt and not state.get("_quiet"):
                print(f"[next] step {nxt['id']}: {nxt['title']}", flush=True)
            return f"step {step['id']} complete"



def agent_validating(state):
    bb = state["_bb"]
    artifacts = state["artifacts"]
    prev_qa = state.get("_prev_qa")
    checks = deterministic_checks(artifacts, state["workdir"])
    failed = [c for c in checks if not c["ok"]]
    verdict, fix_actions = ("pass", []), []
    if artifacts:
        heads = []
        for name in artifacts[:6]:
            p = os.path.join(state["workdir"], name)
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    heads.append(f"--- {name} ---\n" + "".join(f.readlines()[:120]))
            except OSError:
                pass
        # pre-3.12 pythons reject multi-line expressions inside f-strings —
        # build the JSON blobs first so the prompt stays a plain f-string
        plan_json = json.dumps([
            {'id': s['id'], 'title': s['title'],
             'acceptance': s.get('acceptance', []), 'done': s['done']}
            for s in state['plan']['steps']], indent=1)
        checks_json = json.dumps(checks, indent=1)
        messages = [
            {"role": "system", "content": system_prompt(
                "validating", state["budget"]["effort"])},
            {"role": "user", "content": f"GOAL:\n{state['goal']}\n\nSTEP "
             f"ACCEPTANCE CRITERIA:\n{plan_json}\n\nDETERMINISTIC "
             f"CHECK RESULTS:\n{checks_json}\n\nARTIFACT HEADS:\n"
             + "\n".join(heads)[:12000] + "\n\nJudge whether the delivery meets "
             "the goal AND the acceptance criteria. Give your verdict JSON. If "
             "the goal is not met, your fix_actions must contain write_file "
             "actions with the FULL corrected file content."}]
        text, rec = chat_stream(state, messages, max_tokens=8192)
        record(state, "validating", rec)
        v = extract_json(text)
        if isinstance(v, dict) and v.get("verdict") in ("pass", "fail"):
            verdict = (v["verdict"], v.get("failures") or [])
            fix_actions = v.get("fix_actions") or []

    # Self-heal: apply the Validator's executable write_file patches directly
    # and re-check — a fix round without extra model calls (recoimprov §6).
    self_healed = False
    backup = {}
    qa_now = next((c.get("qa", {}).get("qa_func") for c in checks if c.get("qa")), None)
    need_fix = verdict[0] == "fail" or bool(failed)
    if need_fix and fix_actions:
        for name in artifacts:
            p = os.path.join(state["workdir"], name)
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    backup[name] = f.read()
            except OSError:
                pass
        applied = 0
        for fa in fix_actions:
            if isinstance(fa, dict) and fa.get("action") == "write_file":
                r = execute_action(state, "write_file", fa.get("args") or {})
                applied += 1 if "ok" in r else 0
                bb.event("validating", "self_heal", type="self_heal",
                         result=str(r)[:120])
        if applied:
            checks = deterministic_checks(artifacts, state["workdir"])
            failed = [c for c in checks if not c["ok"]]
            qa_new = next((c.get("qa", {}).get("qa_func") for c in checks
                           if c.get("qa")), None)
            if not failed:
                verdict, self_healed = ("pass", []), True
            # Regression guard (recoimprov §7): a fix that scored worse is undone
            if prev_qa is not None and qa_new is not None and qa_new < prev_qa:
                for name, content in backup.items():
                    atomic_write(os.path.join(state["workdir"], name), content)
                checks = deterministic_checks(artifacts, state["workdir"])
                failed = [c for c in checks if not c["ok"]]
                verdict = ("fail", [f"fix regressed QA {prev_qa}→{qa_new}; reverted"])
                bb.event("validating", "regression_reverted", type="regression",
                         prev=prev_qa, now=qa_new)
            else:
                qa_now = qa_new

    max_rounds = state["budget"].get("max_fix_rounds", 3)
    attempts = state["pipeline"]["validating"]["attempts"] + 1
    state["last_checks"] = checks
    state["_prev_qa"] = qa_now
    # progress-aware fix-loop (recoimprov §7 / owner directive): keep fixing
    # while rounds remain, but a round with an IDENTICAL failure signature to
    # the previous one means the loop is not converging — ship documented.
    sig = json.dumps(sorted(f"{c['artifact']}:{c['check']}" for c in failed)) + \
        f"|{qa_now}"
    stalled = need_fix and sig == state.get("_prev_fail_sig")
    state["_prev_fail_sig"] = sig if need_fix else None
    state["pipeline"]["validating"].update(status="done", verdict=verdict[0],
                                           attempts=attempts)
    bb.event("validating", "verdict", type="verdict", verdict=verdict[0],
             failed=len(failed), self_healed=self_healed, stalled=stalled)
    if (verdict[0] == "fail" or failed) and attempts <= max_rounds and \
            not self_healed and not stalled:
        remaining = [fa for fa in fix_actions
                     if not (isinstance(fa, dict) and fa.get("action") == "write_file"
                             and (fa.get("args") or {}).get("content"))]
        state["fix_mode"] = {"fix_actions": remaining or [
            {"step_hint": "repair failed checks",
             "detail": json.dumps(failed)[:400]}]}
        state["fix_roundtrips"] = attempts
        state["pipeline"]["building"].update(status="running")
        state["pipeline"]["validating"].update(status="pending")
        return None  # bounce to Builder
    state["fix_mode"] = None
    qa = next((c["qa"] for c in checks if "qa" in c), None)
    if qa and not qa["usable"]:
        state["result_flags"] = dict(state.get("result_flags", {}), unusable=True)
    msg = f"validation verdict: {verdict[0]}"
    if self_healed:
        msg += " (self-healed by validator)"
    if failed:
        msg += f" ({len(failed)} failed checks)"
    return msg


def agent_summary(state):
    """Summary agent — merged with the Integrator (recoimprov §3): one call
    produces the coherence review (gaps), the project README, and the final
    report."""
    bb = state["_bb"]
    # fast path: clean oneshot build → deterministic report, no model call
    checks_ok = all(c["ok"] for c in state.get("last_checks", [])) \
        if state.get("last_checks") else False
    if state.get("oneshot") and checks_ok:
        rows = []
        if os.path.exists(bb.metrics_path):
            with open(bb.metrics_path) as f:
                rows = [json.loads(l) for l in f if l.strip()]
        agents = {}
        for r in rows:
            a = agents.setdefault(r.get("agent", "?"),
                                  {"calls": 0, "completion_tokens": 0})
            a["calls"] += 1
            a["completion_tokens"] += r.get("completion_tokens") or 0
        report_doc = (f"# hart run {state['run_id']}\n\n**Goal:** "
                      f"{state['goal']}\n\n**Delivered:** "
                      f"{', '.join(state['artifacts'])}\n\n"
                      f"**Validation:** all deterministic checks passed\n\n"
                      f"## Model metrics\n```json\n{json.dumps(agents, indent=1)}\n```\n")
        atomic_write(os.path.join(state["workdir"], "REPORT.md"), report_doc)
        state["integration_gaps"] = []
        state["pipeline"]["summary"].update(status="done")
        return "summary written (fast path)"
    rows = []
    if os.path.exists(bb.metrics_path):
        with open(bb.metrics_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    agents = {}
    for r in rows:
        a = agents.setdefault(r.get("agent", "?"),
                              {"calls": 0, "completion_tokens": 0, "wall": 0.0})
        a["calls"] += 1
        a["completion_tokens"] += r.get("completion_tokens") or 0
        a["wall"] += r.get("wall") or 0.0
    metrics_block = json.dumps(agents, indent=1)
    checks = json.dumps(state.get("last_checks", []), indent=1)[:2500]
    heads = []
    for name in state["artifacts"][:8]:
        p = os.path.join(state["workdir"], name)
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                heads.append(f"--- {name} ---\n" + "".join(f.readlines()[:60]))
        except OSError:
            pass
    messages = [
        {"role": "system", "content": system_prompt(
            "summary", state["budget"]["effort"])},
        {"role": "user", "content": f"GOAL:\n{state['goal']}\n\nPLAN:\n"
         f"{json.dumps(state['plan']['steps'], indent=1)[:2500]}\n\n"
         f"VALIDATION:\n{checks}\n\nPARTS:\n" + "\n".join(heads)[:12000] +
         f"\n\nMODEL METRICS (per agent):\n{metrics_block}\n\n"
         "Produce the merged integration/summary JSON."}]
    text, rec = chat_stream(state, messages, max_tokens=6144)
    record(state, "summary", rec)
    out = extract_json(text) or {}
    gaps = out.get("gaps") or []
    readme = out.get("readme")
    report = out.get("report")
    if isinstance(readme, str) and len(readme) > 80:
        atomic_write(os.path.join(state["workdir"], "README.md"), readme)
        if "README.md" not in state["artifacts"]:
            state["artifacts"].append("README.md")
    if not isinstance(report, str) or len(report) < 40:
        report = ("(model report unavailable — deterministic results below)\n\n"
                  f"**Validation:** {checks}\n\n**Artifacts:** "
                  f"{', '.join(state['artifacts']) or '(none)'}")
    report_doc = (f"# hart run {state['run_id']}\n\n**Goal:** {state['goal']}\n\n"
                  f"{report}\n\n## Integration gaps\n"
                  f"{json.dumps(gaps, indent=1) if gaps else 'none detected'}\n\n"
                  f"## Model metrics\n```json\n{metrics_block}\n```\n")
    atomic_write(os.path.join(state["workdir"], "REPORT.md"), report_doc)
    state["integration_gaps"] = gaps
    state["pipeline"]["summary"].update(status="done")
    return ("summary written" + (f" ({len(gaps)} integration gaps noted)"
                                 if gaps else ""))


AGENTS = {"connection": agent_connection, "planning": agent_planning,
          "building": agent_building, "validating": agent_validating,
          "summary": agent_summary}


# ---------------------------------------------------------------------------
# Orchestrator (DESIGN.md §2.1)
# ---------------------------------------------------------------------------
def next_stage(state):
    if state["pipeline"]["planning"]["status"] == "running":
        return "planning"                      # loop-ladder re-plan
    if state["pipeline"]["building"]["status"] == "running":
        return "building"                      # fix-loop bounce or resume
    for stage in PIPELINE:
        if state["pipeline"][stage]["status"] != "done":
            return stage
    return None


def run_pipeline(state, bb, quiet):
    start = state.get("started_at") or now()
    state["started_at"] = start
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    while True:
        if ABORT["flag"]:
            bb.save(state)
            bb.event("orchestrator", "aborted")
            raise RunStopped()
        if now() - start > state["budget"]["time_budget"]:
            if state["budget"].get("epochs", 1) > 1:
                new_epoch(state, bb, "epoch time budget reached")
                start = state["started_at"]
                continue
            state["result"] = {"status": "error", "error": "time budget exceeded"}
            bb.save(state)
            raise RunFailed("time budget exceeded")
        stage = next_stage(state)
        if stage is None:
            break
        say(f"[{stage}] → " + STAGE_DESC.get(stage, "working") + " …")
        state["pipeline"][stage]["status"] = "running"
        bb.save(state)
        try:
            msg = AGENTS[stage](state)
        except RunFailed as e:
            state["result"] = {"status": "error", "error": e.reason}
            bb.save(state)
            raise
        except Exception as e:  # noqa: BLE001 — surface, never swallow
            state["result"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            bb.save(state)
            raise
        if msg is None:
            continue  # agent bounced control (fix-loop / re-plan)
        say(f"[{stage}] ✓ {msg}")
        bb.event("orchestrator", "stage_done", stage=stage, message=str(msg))
        bb.save(state)
    state["result"] = {"status": "done"}
    bb.save(state)


# ---------------------------------------------------------------------------
# metrics rollup + machine line
# ---------------------------------------------------------------------------
def finish_metrics(state, bb, status, error=None):
    rows = []
    if os.path.exists(bb.metrics_path):
        with open(bb.metrics_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    agents = {}
    for r in rows:
        a = agents.setdefault(r.get("agent", "?"),
                              {"calls": 0, "prompt_tokens": 0,
                               "completion_tokens": 0, "wall": 0.0})
        a["calls"] += 1
        a["prompt_tokens"] += r.get("prompt_tokens") or 0
        a["completion_tokens"] += r.get("completion_tokens") or 0
        a["wall"] += r.get("wall") or 0.0
    if rows:
        span = max(r.get("ts") or 0 for r in rows) - min(r.get("ts") or 0 for r in rows)
        total = span if span > 0 else sum(r.get("wall") or 0 for r in rows)
    else:
        total = 0
    tokens = sum(r.get("completion_tokens") or 0 for r in rows)
    total_calls = len(rows)
    qa = next((c.get("qa") for c in state.get("last_checks", []) if c.get("qa")), None)
    peak_fill = max((r.get("ctx_fill_pct") or 0 for r in rows), default=0) if rows else 0
    gb = [r.get("est_gbps") for r in rows if r.get("est_gbps")]
    row = {"peak_ctx_fill_pct": peak_fill or None,
           "est_gbps_avg": round(sum(gb) / len(gb), 1) if gb else None,
           "framework": state["backend"].get("framework") or
           re.sub(r"[^a-z0-9]+", "-", state["backend"]["base_url"]),
           "model": state["backend"]["model"], "harness": "hart",
           "task": state["goal"][:80], "status": status,
           "latency": round(now() - state["started_at"], 2),
           "tokens": tokens, "calls": total_calls,
           "tps": round(tokens / total, 1) if total else None,
           "output_url": (state["artifacts"] or [None])[0],
           "qa_func": qa.get("qa_func") if qa else None,
           "qa_qual": qa.get("qa_qual") if qa else None,
           "usable": qa.get("usable") if qa else None,
           "truncated": any(r.get("finish") == "length" for r in rows),
           "ttft": rows[0].get("ttft") if rows else None,
           "tgs": rows[0].get("tgs") if rows else None,
           "pp": rows[0].get("pp") if rows else None,
           "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
           "error": error}
    out = {"run_id": state["run_id"], "result_row": row, "per_agent": agents,
           "artifacts": state["artifacts"],
           "loop_guard": state["loop_guard"]["hashes"],
           "compactions": len(state.get("compactions", []))}
    path = state.get("_metrics_out") or os.path.join(
        os.path.dirname(bb.dir), "hart-metrics.json")
    atomic_write(path, json.dumps(out, indent=1))
    return out


def registry_append(state, status):
    reg = os.path.expanduser("~/.hart/runs-index.jsonl")
    os.makedirs(os.path.dirname(reg), exist_ok=True)
    with open(reg, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": state["run_id"], "ts": now(),
                            "workdir": state["workdir"],
                            "task": state["goal"][:70], "status": status}) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Interactive TUI (pi-style): hart> prompt, quick answers for questions,
# full agentic pipeline (with narration) for tasks. Framework/model config
# lives in ~/.hart/models.json (seeded from the presets on first run).
# ---------------------------------------------------------------------------
HART_CONFIG = os.path.expanduser("~/.hart/models.json")

QUICK_RE = re.compile(
    r"^(hi|hey|hello|yo|sup|thanks|thank you|ok|okay|what|who|how|why|when|"
    r"where|which|can you|could you|do you|are you|tell me|explain|describe)",
    re.I)
TASK_VERBS = re.compile(
    r"\b(write|create|build|make|fix|refactor|generate|implement|develop|"
    r"produce|draft)\b", re.I)


def load_config():
    """pi-style config: ~/.hart/models.json seeded from presets; user edits
    (ports, models, extra frameworks) are merged into FRAMEWORKS."""
    if not os.path.exists(HART_CONFIG):
        os.makedirs(os.path.dirname(HART_CONFIG), exist_ok=True)
        atomic_write(HART_CONFIG, json.dumps(
            {"defaultFramework": "omlx", "frameworks": FRAMEWORKS}, indent=2))
        print(f"hart: wrote default config → {HART_CONFIG}", flush=True)
    try:
        cfg = json.load(open(HART_CONFIG))
        user_fw = cfg.get("frameworks") or {}
        for name, c in user_fw.items():
            if isinstance(c, dict) and c.get("base_url"):
                FRAMEWORKS[name] = c
        return cfg
    except (json.JSONDecodeError, OSError) as e:
        print(f"hart: config unreadable ({e}) — using presets", flush=True)
        return {}


def is_quick_question(text):
    t = text.strip()
    if len(t) > 200 or TASK_VERBS.search(t):
        return False
    return bool(QUICK_RE.match(t)) or t.endswith("?")


def _probe_backend(fw_name):
    """Connection test with verbose output; returns model id or None."""
    cfg = FRAMEWORKS.get(fw_name)
    if not cfg:
        print(f"hart: unknown framework '{fw_name}' (see /frameworks)", flush=True)
        return None
    print(f"[connection] probing {cfg['base_url']} …", flush=True)
    state = {"backend": {"base_url": cfg["base_url"], "model": cfg["model"],
                         "api_key": "bench", "framework": fw_name},
             "budget": {"ctx_tokens": cfg.get("ctx_tokens", 262144),
                        "temperature": 0.4, "per_call_max_tokens": 8192,
                        "seed": None}}
    try:
        import urllib.request as _u
        with _u.urlopen(cfg["base_url"].rstrip("/") + "/models", timeout=8) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        model = cfg["model"] if cfg["model"] in ids else (
            ids[0] if len(ids) == 1 else cfg["model"])
        print(f"[connection] ✓ {fw_name} reachable · {len(ids)} model(s) · "
              f"using {model}", flush=True)
        return model
    except Exception as e:
        hint = cfg.get("start_hint", "")
        print(f"[connection] ✗ {fw_name} not reachable ({e})"
              + (f"\n  start it: {hint}" if hint else ""), flush=True)
        return None


def _last_checkpoint_info():
    """Newest run state across ./hart-run-* — where we last checkpointed."""
    import glob as _g
    cands = []
    for sp in _g.glob(os.path.join(os.getcwd(), "hart-run-*", ".hart",
                                   "state.json")):
        try:
            s = json.load(open(sp))
            cands.append((s.get("saved_at", 0), s))
        except (json.JSONDecodeError, OSError):
            pass
    if not cands:
        return None
    _, s = max(cands)
    stage = next((k for k in PIPELINE if s["pipeline"][k]["status"] != "done"),
                 "complete")
    comp = s.get("compactions") or []
    return {"run": s.get("run_id"), "stage": stage, "epoch": s.get("epoch", 1),
            "steps": s.get("steps"), "compactions": len(comp),
            "workdir": s.get("workdir")}


def execute_task(goal, fw_name, quiet=False):
    """Run the full agentic pipeline on a goal with verbose narration;
    returns the result row dict."""
    cfg = FRAMEWORKS[fw_name]
    workdir = os.path.join(os.getcwd(), "hart-run-" + time.strftime("%Y%m%d-%H%M%S"))
    bb = Blackboard(workdir)
    bb.acquire_lock()
    state = {"version": 1, "run_id": new_run_id(), "goal": goal,
             "workdir": workdir,
             "backend": {"base_url": cfg["base_url"], "model": cfg["model"],
                         "api_key": "bench", "framework": fw_name},
             "budget": {"max_steps": 40, "ctx_tokens": cfg.get("ctx_tokens", 262144),
                        "per_call_max_tokens": 65536, "time_budget": 7200,
                        "temperature": 0.4, "effort": "medium", "seed": None,
                        "max_fix_rounds": 3, "repair_rounds": 3, "epochs": 336},
             "pipeline": {a: {"status": "pending", "attempts": 0}
                          for a in PIPELINE},
             "plan": None, "compactions": [], "token_ratio": CTX_CHARS_PER_TOKEN,
             "epoch": 1,
             "loop_guard": {"hashes": {}, "last_fp": None, "unchanged": 0,
                            "nudges": 0},
             "observations": [], "rollups": [], "artifacts": [], "fix_mode": None,
             "fix_roundtrips": 0, "steps": 0, "result": None,
             "started_at": now(), "_quiet": quiet}
    state["_bb"] = bb
    bb.save(state)
    bb.event("orchestrator", "tui_started", goal=goal[:120])
    print(f"\n[task] {goal[:100]}", flush=True)
    print(f"[task] workdir {workdir}", flush=True)
    status = "error"
    error = None
    try:
        run_pipeline(state, bb, quiet)
        status = "done"
    except RunFailed as e:
        error = e.reason
        print(f"hart: FAILED — {e.reason}", flush=True)
    except RunStopped:
        error = "stopped"
    finally:
        out = finish_metrics(state, bb, status, error)
        registry_append(state, status)
        bb.release_lock()
    row = out.get("result_row", {})
    print(f"\n[done] {status}" + (f" — {error}" if error else ""))
    print(f"  artifacts: {', '.join(state['artifacts']) or '(none)'}")
    print(f"  calls: {row.get('calls')} · tokens: {row.get('tokens')} · "
          f"latency: {row.get('latency')}s"
          + (f" · QA {row.get('qa_func')}%" if row.get("qa_func") else ""))
    print(f"  report: {os.path.join(workdir, 'REPORT.md')}")
    print("HART_RESULT " + json.dumps(row))
    return row


def interactive_mode():
    cfg = load_config()
    fw = cfg.get("defaultFramework") or "omlx"
    if fw not in FRAMEWORKS:
        fw = next(iter(FRAMEWORKS))
    print(f"hart {VERSION} interactive · type /help for commands", flush=True)
    _probe_backend(fw)
    while True:
        try:
            line = input(f"hart({fw})> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nhart: bye 👋", flush=True)
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            print("hart: bye 👋", flush=True)
            return 0
        if line == "/help":
            print("  /frameworks          list configured frameworks\n"
                  "  /framework NAME      switch backend (reconnects)\n"
                  "  /status              backend + last checkpoint info\n"
                  "  /ask TEXT            force a direct quick answer (no pipeline)\n"
                  "  /do TEXT             force the full agentic pipeline\n"
                  "  /exit                leave\n"
                  "  plain text           auto: quick question → direct answer;\n"
                  "                      task → plan/build/validate pipeline\n"
                  "  config: ~/.hart/models.json", flush=True)
            continue
        if line == "/frameworks":
            for name, c in sorted(FRAMEWORKS.items()):
                mark = "← current" if name == fw else ""
                print(f"  {name:10} {c['base_url']:30} {c['model'][:40]} {mark}",
                      flush=True)
            continue
        if line.startswith("/framework "):
            nf = line.split(None, 1)[1].strip()
            if nf in FRAMEWORKS:
                fw = nf
                _probe_backend(fw)
            else:
                print(f"hart: unknown framework '{nf}' (see /frameworks)",
                      flush=True)
            continue
        if line == "/status":
            c = FRAMEWORKS[fw]
            print(f"  backend : {c['base_url']}\n  model   : {c['model']}\n"
                  f"  context : {c.get('ctx_tokens')} tokens", flush=True)
            cp = _last_checkpoint_info()
            if cp:
                print(f"  last ckpt: run {cp['run']} · stage {cp['stage']} · "
                      f"epoch {cp['epoch']} · {cp['steps']} calls · "
                      f"{cp['compactions']} compactions", flush=True)
            continue
        if line.startswith("/ask "):
            quick, text = True, line[5:].strip()
        elif line.startswith("/do "):
            quick, text = False, line[4:].strip()
        else:
            text = line
            quick = is_quick_question(text)
        if quick:
            c = FRAMEWORKS[fw]
            state = {"backend": {"base_url": c["base_url"], "model": c["model"],
                                 "api_key": "bench", "framework": fw},
                     "budget": {"ctx_tokens": c.get("ctx_tokens", 262144),
                                "temperature": 0.4, "per_call_max_tokens": 2048,
                                "seed": None}}
            try:
                text_out, rec = chat_stream(
                    state, [{"role": "user", "content": text}], max_tokens=1024)
                print(f"\n{text_out.strip()}\n", flush=True)
                print(f"  ({rec.get('completion_tokens')} tok · "
                      f"{rec.get('tgs')} tok/s · ttft {rec.get('ttft')}s)",
                      flush=True)
            except RunStopped:
                print("  (interrupted)", flush=True)
            except Exception as e:
                print(f"  hart: backend error — {e}", flush=True)
            continue
        # full agentic pipeline with verbose narration
        try:
            execute_task(text, fw)
        except RunStopped:
            print("  (task interrupted — state checkpointed; resume with "
                  "`hart.py --resume`)", flush=True)


def sig_handler(signum, frame):
    ABORT["flag"] = True
    sock = CURRENT.get("sock")
    if sock:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


signal.signal(signal.SIGTERM, sig_handler)
signal.signal(signal.SIGINT, sig_handler)


def find_resume_root(start):
    """Newest unfinished hart run under `start` (or its hart-run-* children)."""
    if not os.path.isdir(start):
        return None
    cands = []
    for root in ([start] + [os.path.join(start, d) for d in os.listdir(start)
                             if d.startswith("hart-run")]):
        sp = os.path.join(root, ".hart", "state.json")
        if os.path.isfile(sp):
            try:
                st = json.load(open(sp))
                if st.get("result", {}) is None or st.get("result") is None or \
                        st.get("result", {}).get("status") != "done":
                    cands.append((st.get("saved_at", 0), root, st))
            except (json.JSONDecodeError, OSError):
                pass
    return max(cands)[1] if cands else None


def cmd_runs(args):
    reg = os.path.expanduser("~/.hart/runs-index.jsonl")
    if not os.path.exists(reg):
        print("no hart runs recorded yet")
        return 0
    seen = {}
    with open(reg) as f:
        for line in f:
            try:
                r = json.loads(line)
                seen[r["run_id"]] = r  # last status wins
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"{'RUN ID':22} {'STATUS':8} {'TASK':60} WORKDIR")
    for r in sorted(seen.values(), key=lambda x: -x["ts"]):
        print(f"{r['run_id']:22} {r['status']:8} {r['task'][:58]:60} {r['workdir']}")
    return 0


def main(argv):
    global VERBOSE
    if argv and argv[0] == "runs":
        return cmd_runs(argv[1:])
    ap = argparse.ArgumentParser(
        prog="hart", description="Fully agentic harness for local "
                                   "OpenAI-compatible backends (see DESIGN.md)")
    ap.add_argument("task", nargs="?", help="task text (like pi's message arg)")
    ap.add_argument("--task", dest="task_opt", help="task text or file path")
    ap.add_argument("--framework", choices=sorted(FRAMEWORKS),
                    help="use a preset backend (see --list-frameworks)")
    ap.add_argument("--list-frameworks", action="store_true")
    ap.add_argument("--base-url", help="custom OpenAI-compatible base URL (/v1)")
    ap.add_argument("--model", help="model id (overrides preset)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--resume", nargs="?", const="newest", default=None,
                    metavar="RUN_ID")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore/reuse workdir: archive prior state instead of failing")
    ap.add_argument("--max-steps", type=int, default=40,
                    help="SAFETY VALVE — model calls per epoch; the run "
                         "auto-continues in a fresh epoch when reached")
    ap.add_argument("--ctx-tokens", type=int, default=None)
    ap.add_argument("--time-budget", type=int, default=7200,
                    help="SAFETY VALVE — per-epoch wall clock. The harness "
                         "auto-continues across epochs by default; you never "
                         "need this for long tasks")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--per-call-tokens", type=int, default=None,
                    help="default: low=4096, medium=8192, high=16384 by --effort")
    ap.add_argument("--max-fix-rounds", type=int, default=3)
    ap.add_argument("--repair-rounds", type=int, default=3,
                    help="builder-level diagnose-and-retry cycles before a run fails")
    ap.add_argument("--epochs", type=int, default=336,
                    help="SAFETY VALVE — the harness always runs the fastest "
                         "path first and auto-continues across epochs when a "
                         "task needs more time. A "
                         "no-progress guard stops pathological runs; only "
                         "lower this if you WANT a hard wall")
    ap.add_argument("--plan-file", default=None,
                    help="execute a plan from a file (JSON steps schema or a "
                         "markdown checklist, e.g. a FIXPLAN.md written by an "
                         "earlier analysis run)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="dump model wire I/O to stderr")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only — show the plan and exit (no build)")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)

    if args.version:
        print(f"hart {VERSION}")
        return 0
    if args.list_frameworks:
        print(f"{'NAME':8} {'BACKEND':28} MODEL")
        for name, f in sorted(FRAMEWORKS.items()):
            print(f"{name:8} {f['base_url']:28} {f['model']}")
        return 0
    if args.task_opt and os.path.isfile(args.task_opt):
        with open(args.task_opt) as f:
            goal = f.read().strip()
    else:
        goal = (args.task_opt or args.task or "").strip()
    if not goal and not args.resume:
        return interactive_mode()   # pi-style TUI: hart> prompt

    # ---- locate workdir (resume support) ----
    if args.resume:
        base = args.workdir or os.getcwd()
        wd = find_resume_root(base)
        if not wd and args.resume != "newest":
            wd = os.path.join(base, args.resume)
        if not wd or not os.path.isdir(os.path.join(wd, ".hart")):
            sys.exit(f"hart: no resumable run found under {base}")
        workdir = wd
    else:
        workdir = args.workdir or os.path.join(os.getcwd(), "hart-run-" +
                                               time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(workdir, exist_ok=True)
    bb = Blackboard(workdir)
    bb.acquire_lock()

    try:
        if args.resume:
            state = bb.load()
            if isinstance(state.get("result"), dict) and \
                    state["result"].get("status") == "done" and not args.fresh:
                bb.release_lock()
                sys.exit(f"hart: run {state['run_id']} already completed — "
                         f"use --fresh to rerun in this workdir")
            if goal and goal != state["goal"]:
                state["goal"] = goal  # owner revised the request mid-flight
            state["_metrics_out"] = args.metrics_out
            bb.event("orchestrator", "resumed")
            if not args.quiet:
                print(f"resuming run {state['run_id']} in {workdir}", flush=True)
        else:
            if os.path.exists(bb.state_path):
                if args.fresh:
                    bak = bb.state_path + "-" + time.strftime("%H%M%S") + ".bak"
                    os.replace(bb.state_path, bak)
                else:
                    bb.release_lock()
                    sys.exit(f"hart: {workdir} already holds a run — use --fresh "
                             f"to archive it or --resume to continue")
            state = fresh_state(args, goal, workdir)
            state["_metrics_out"] = args.metrics_out
            if args.plan_file:
                steps, fmt = parse_plan_file(args.plan_file)
                if not steps:
                    bb.release_lock()
                    sys.exit(f"hart: no executable plan found in {args.plan_file}")
                steps, notes = plan_selfcheck(steps, goal)
                state["plan"] = {"steps": steps, "created_by": f"plan-file:{fmt}",
                                 "revisions": 0}
                state["pipeline"]["planning"].update(status="done")
                bb.event("orchestrator", "plan_file_loaded", steps=len(steps))
                if not args.quiet:
                    print(f"[planning] loaded {len(steps)} steps from "
                          f"{args.plan_file} ({fmt})", flush=True)
            bb.save(state)
            bb.event("orchestrator", "started", goal=goal[:120])
            registry_append(state, "running")
            if not args.quiet:
                print(f"hart {VERSION} · run {state['run_id']} · "
                      f"{state['backend']['model']} @ {state['backend']['base_url']}",
                      flush=True)
        state["_bb"] = bb
        state["_quiet"] = args.quiet
    except SystemExit:
        raise
    except Exception as e:
        bb.release_lock()
        sys.exit(f"hart: failed to initialize run: {e}")

    VERBOSE = args.verbose

    if args.dry_run:
        # plan-only mode: see the plan before committing GPU time
        try:
            agent_connection(state)
            agent_planning(state)
            bb.save(state)
            bb.release_lock()
            print(json.dumps(state["plan"]["steps"], indent=1))
            print("HART_RESULT " + json.dumps(
                {"run_id": state["run_id"], "status": "dry-run",
                 "steps": len(state["plan"]["steps"]),
                 "workdir": workdir}))
            registry_append(state, "dry-run")
            bb.release_lock()
            return 0
        except Exception as e:
            bb.release_lock()
            sys.exit(f"hart: dry-run failed: {e}")

    status, error = "error", None
    try:
        run_pipeline(state, bb, args.quiet)
        status = "done"
    except RunFailed as e:
        error = e.reason
        if not args.quiet:
            print(f"hart: FAILED — {e.reason}", flush=True)
    except RunStopped:
        error = "stopped"
    except KeyboardInterrupt:
        error = "stopped"
    finally:
        if ABORT["flag"]:
            status = "aborted"
        try:
            out = finish_metrics(state, bb, status, error)
        except Exception as e:  # metrics must never mask the real outcome
            out = {"error": f"metrics failed: {e}"}
        registry_append(state, status)
        bb.release_lock()

    if not args.quiet:
        r = out.get("result_row", {})
        print(f"\nhart {status}" +
              (f" — {error}" if error else "") +
              f"\n  artifacts: {', '.join(state['artifacts']) or '(none)'}"
              f"\n  tokens: {r.get('tokens')} · tps: {r.get('tps')} · "
              f"calls: {r.get('calls')} · latency: {r.get('latency')}s"
              f"\n  report: {os.path.join(workdir, 'REPORT.md')}"
              f"\n  metrics: {state.get('_metrics_out') or os.path.join(workdir, 'hart-metrics.json')}")
    row = out.get("result_row", {})
    row["artifacts"] = state["artifacts"]
    row["report"] = "REPORT.md"
    row["run_id"] = state["run_id"]
    row["workdir"] = workdir
    print("HART_RESULT " + json.dumps(row))
    if ABORT["flag"]:
        return 130
    return 0 if status == "done" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
