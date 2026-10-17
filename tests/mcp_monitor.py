"""One monitoring pass over the benchtest MCP server: state, error-class
bench-log lines, per-framework server errors. Each MCP call is printed with
its name so the invocation log doubles as the review log."""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, "mcp", "venv", "bin", "python")
SERVER = os.path.join(ROOT, "mcp", "benchtest_mcp.py")

CALLS = []


async def call(session, name, args=None, keep=6):
    r = await session.call_tool(name, args or {})
    payload = r.content[0].text
    CALLS.append({"tool": name, "args": args or {}, "bytes": len(payload)})
    return payload


async def main():
    params = StdioServerParameters(command=VENV_PY, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()

            state = json.loads(await call(s, "bench_state"))
            c = state.get("campaign") or {}
            print("── bench_state ──────────────────────────────")
            print(f"running={state['running']}  step={(state.get('current_step') or '')[:60]}")
            print(f"campaign={c.get('status')}  run {(c.get('index') or 0) + 1}/{c.get('plan_total')}")
            bad = [x for x in state.get("results", [])
                   if x.get("status") not in ("done", None) or x.get("error")]
            for x in state.get("results", []):
                flag = "⚠" if x.get("error") else " "
                print(f"  {flag} {x.get('framework')}/{x.get('harness'):<9} "
                      f"{x.get('status'):<7} "
                      f"{str(x.get('error') or '')[:60]}")

            errs = await call(s, "tail_bench_log", {"lines": 400, "only_errors": True})
            print("── tail_bench_log(400, only_errors) ─────────")
            interesting = [l for l in errs.splitlines()
                           if any(w in l.lower() for w in
                                  ("empty response", "timeout", "no output", "exited",
                                   "stall", "guard", "error"))]
            seen = set()
            for l in interesting[-8:]:
                key = l[-60:]
                if key not in seen:
                    seen.add(key)
                    print("  " + l[:150])

            for fw in ("omlx", "mtplx"):
                fl = await call(s, "framework_log", {"framework": fw, "lines": 200,
                                                     "only_errors": True})
                lines = [l for l in fl.splitlines() if l.strip()]
                print(f"── framework_log({fw}) ── {len(lines)} error-class lines ──")
                for l in lines[-4:]:
                    print("  " + l[:150])

    print("── MCP CALL LOG ─────────────────────────────")
    for c in CALLS:
        print(f"  {c['tool']}({json.dumps(c['args'])[:60]}) -> {c['bytes']} bytes")


asyncio.run(main())
