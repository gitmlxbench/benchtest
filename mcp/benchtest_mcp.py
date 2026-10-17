#!/usr/bin/env python3
"""benchtest MCP server — exposes the local LLM benchmark to any MCP host
(ZCode, Claude Desktop, ...). Pure HTTP client of the bench backend
(127.0.0.1:7090): no benchtest imports, so it can never interfere with a
running benchmark's config state.

Run:  mcp/venv/bin/python mcp/benchtest_mcp.py   (stdio transport)
"""
import glob
import json
import os
import urllib.error
import urllib.request

from mcp.server.mcpserver import MCPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# bench URL: BENCHTEST_URL env wins; otherwise read the repo's own config
# so the server follows whatever host/port this deployment actually uses
def _bench_url():
    if os.environ.get("BENCHTEST_URL"):
        return os.environ["BENCHTEST_URL"]
    try:
        with open(os.path.join(ROOT, "config.json")) as fh:
            c = json.load(fh)
        host = c.get("host") or "127.0.0.1"
        if host in ("0.0.0.0", "::", ""):   # bind-anywhere → connect locally
            host = "127.0.0.1"
        return f"http://{host}:{c.get('port', 7090)}"
    except (OSError, json.JSONDecodeError):
        return "http://127.0.0.1:7090"

BENCH = _bench_url()

mcp = MCPServer(
    name="benchtest",
    instructions=(
        "Tools for operating the benchtest local-LLM benchmark: live state, "
        "run history, run comparison, log tails and campaign control. "
        "Artifact quality is judged by the human from the dashboard's Open "
        "links — there is no automated scoring. Read tools are always safe; "
        "the control tools (start_campaign, stop_run) mutate the "
        "benchmark and say so in their descriptions."
    ),
)


def _get(path, timeout=10):
    with urllib.request.urlopen(BENCH + path, timeout=timeout) as r:
        return json.loads(r.read())


def _post(path, body, timeout=30):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BENCH + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode()[:200]}"}


@mcp.tool()
def bench_state() -> str:
    """Live benchmark state: is a run or campaign active, which cell is
    executing, per-cell status/run time/TPS of the current run, RAM, framework health."""
    d = _get("/api/state")
    c = d.get("campaign") or {}
    out = {
        "running": d.get("running"),
        "current_step": d.get("current_step"),
        "campaign": {k: c.get(k) for k in
                     ("status", "sets", "index", "plan_total", "current", "skipped")},
        "results": [
            {k: r.get(k) for k in ("framework", "harness", "status",
                                   "qa_score", "tps", "latency", "error",
                                   "output_url")}
            for r in (d.get("results") or [])
        ],
    }
    return json.dumps(out, indent=1)


@mcp.tool()
def list_runs(limit: int = 10) -> str:
    """Recent benchmark runs from history: task, model set, done/error
    counts, fastest cell. Set limit larger to see everything."""
    runs = _get("/api/runs_history")
    out = []
    for r in runs[:limit]:   # runs_history is newest-first
        rows = r.get("rows") or []
        done = [x for x in rows if x.get("status") == "done"]
        failed = [x for x in rows if x.get("status") == "error"]
        best = max(done, key=lambda x: (x.get("tps") or 0), default=None)
        out.append({
            "file": r.get("file"), "task": r.get("task"),
            "model_set": r.get("model_set"),
            "date": r.get("ts"), "cells": len(rows),
            "done": len(done), "failed": len(failed),
            "best": (f"{best.get('framework')}/{best.get('harness')} "
                     f"{best.get('tps')} tok/s") if best else None,
        })
    return json.dumps(out, indent=1)


@mcp.tool()
def get_run(run_file: str) -> str:
    """Full per-cell records of one saved run (pass runs/20260930-140000.json
    as run_file — the file part is enough). Includes run time, tokens, TPS
    and error strings; artifacts are reviewed by humans via output_url."""
    f = os.path.join(ROOT, "runs", os.path.basename(run_file))
    if not f.endswith(".json"):
        f += ".json"
    with open(f) as fh:
        d = json.load(fh)
    rows = [{k: r.get(k) for k in ("framework", "harness", "status", "latency",
                                   "tokens", "tps", "qa_score", "truncated",
                                   "tokens_estimated", "error", "output_url")}
            for r in (d.get("results") or [])]
    return json.dumps({"task": d.get("task_id"), "model_set": d.get("model_set"),
                       "reasoning": d.get("reasoning_level"), "rows": rows}, indent=1)


@mcp.tool()
def compare_runs(run_a: str, run_b: str) -> str:
    """Side-by-side comparison of two saved runs: per framework/harness pair,
    TPS, run time and status from each run. Pass file-name parts of
    two runs that ran the SAME task for a fair model or harness comparison."""
    def load(name):
        f = os.path.join(ROOT, "runs", os.path.basename(name))
        if not f.endswith(".json"):
            f += ".json"
        with open(f) as fh:
            return json.load(fh)

    a, b = load(run_a), load(run_b)
    index = {(r.get("framework"), r.get("harness")): r
             for r in (b.get("results") or [])}
    rows = []
    for r in (a.get("results") or []):
        key = (r.get("framework"), r.get("harness"))
        o = index.get(key, {})
        rows.append({
            "framework": key[0], "harness": key[1],
            "A": {"task": a.get("task_id"), "tps": r.get("tps"),
                  "latency": r.get("latency"), "status": r.get("status")},
            "B": {"task": b.get("task_id"), "tps": o.get("tps"),
                  "latency": o.get("latency"), "status": o.get("status")},
        })
    return json.dumps(rows, indent=1)


@mcp.tool()
def tail_bench_log(lines: int = 60, only_errors: bool = False) -> str:
    """Tail the orchestrator log (logs/bench.log) — the primary forensics
    source for what any harness or framework actually did. Set only_errors
    to filter to failure-class lines."""
    path = os.path.join(ROOT, "logs", "bench.log")
    with open(path, encoding="utf-8", errors="replace") as fh:
        buf = fh.readlines()[-lines:]
    if only_errors:
        buf = [l for l in buf if any(w in l.lower() for w in
                                     ("error", "fail", "timeout", "⚠", "exited", "empty"))]
    return "".join(buf)


@mcp.tool()
def framework_log(framework: str, lines: int = 60, only_errors: bool = True) -> str:
    """Tail a framework server log (omlx, mtplx, mlxvlm, mlxserve) — newest
    file for that framework. Server-side errors (memory guard, stream-stall
    breaks, load failures) show up here, not in bench.log."""
    framework = (framework or "").strip().lower()
    if framework not in ("omlx", "mtplx", "mlxvlm", "mlxserve"):
        return "framework must be one of: omlx, mtplx, mlxvlm, mlxserve"
    hits = sorted(glob.glob(os.path.join(ROOT, "logs", f"{framework}-*.log")),
                  key=os.path.getmtime)
    if not hits:
        return f"no log files for {framework!r}"
    with open(hits[-1], encoding="utf-8", errors="replace") as fh:
        buf = fh.readlines()[-lines:]
    if only_errors:
        buf = [l for l in buf if any(w in l.lower() for w in
                                     ("error", "warning", "fail", "guard", "stall", "reject"))]
    return "".join(buf) or "(no error-class lines in the window)"






@mcp.tool()
def start_campaign(sets: list[str], harnesses: list[str] | None = None) -> str:
    """MUTATES STATE: start a benchmark campaign over whole tasks — sets is a
    list of model-set names ('qwen38-27b', 'flash-next'), harnesses defaults
    to all six (raw, rawplus, pi, opencode, aider, hart). Refuses if a run
    or campaign is already active."""
    return json.dumps(_post("/api/campaign",
                            {"action": "start", "mode": "all", "sets": sets,
                             "harnesses": harnesses or []}))


@mcp.tool()
def stop_run() -> str:
    """MUTATES STATE: request a stop of the current run/campaign. Partial
    results are kept; a paused campaign can be resumed from the dashboard."""
    return json.dumps(_post("/api/stop", {}))


if __name__ == "__main__":
    mcp.run()
