# benchtest MCP server

An MCP (Model Context Protocol) server that exposes the benchtest benchmark
to any MCP-capable AI host — ZCode, Claude Desktop, Claude Code, Cursor,
Windsurf, Cline, Continue, and anything else that speaks MCP stdio. Once
connected, the host's AI agent can **answer questions about the benchmark,
inspect runs, read logs and start/stop campaigns in natural language**:

> *"What is the benchmark doing right now?"*
> *"Compare the last two tetris runs and tell me which framework decoded faster."*
> *"Tail the bench log for errors from the last hour."*
> *"Start a campaign over the qwen38-27b set with just raw and pi harnesses."*
> *"Stop the run — I need the machine."*

## What it does

The MCP server is a **thin, pure-HTTP client of the benchtest backend**
(`server.py`, default `http://127.0.0.1:7090`). It imports nothing from
benchtest, holds no state of its own, and can never interfere with a running
benchmark's config. Every tool call is translated to one HTTP call against
the backend's `/api/*` endpoints and the JSON response is returned to the
AI host.

```
┌──────────────┐   MCP stdio    ┌───────────────────┐   HTTP /api/*   ┌──────────────┐
│  AI host     │ ◄────────────► │  mcp/             │ ◄─────────────► │  server.py   │
│  (ZCode,     │                │  benchtest_mcp.py │                 │  :7090       │
│  Claude, …)  │                │  (this server)    │                 │  benchmark   │
└──────────────┘                └───────────────────┘                 └──────────────┘
```

- **Transport:** stdio — the AI host spawns this server as a child process
  and talks to it over stdin/stdout. Nothing listens on the network.
- **Backend URL resolution:** the `BENCHTEST_URL` environment variable wins;
  otherwise the server reads `host` + `port` from the repo's `config.json`
  (a `host: "0.0.0.0"` bind is normalized to `127.0.0.1` for connecting);
  fallback default `http://127.0.0.1:7090`.
- **Artifact quality is judged by humans** from the dashboard's Open ▸ links
  — there is no automated scoring, by design.

## Requirements

| Thing | Why |
|---|---|
| The benchtest backend running (`python3 server.py`) | every tool is an HTTP call to it |
| `mcp/venv` with the pinned `mcp` package (`mcp/requirements.txt`) | the only third-party dependency in the repo |
| Python 3.10+ | stdlib + the `mcp` package |

`install.sh` creates the venv automatically (skipping it if present);
`scripts/register_mcp.py` also creates it on demand.

## The tools

**Read-only (always safe):**

| Tool | Parameters | What it does |
|---|---|---|
| `bench_state` | — | Live snapshot: is a run or campaign active, which cell is executing, per-cell status / run time / TPS / error for the current run, framework health. The "what's happening" tool. |
| `list_runs` | `limit: int = 10` | Recent runs from history, newest first: task, model set, date, done/error counts, and the fastest cell of each run. Raise `limit` to see everything. |
| `get_run` | `run_file: str` | Full per-cell records of one saved run — pass the file-name part of a `runs/<ts>.json` entry (e.g. `20261006-032316`). Includes run time, tokens, TPS, truncation flags, error strings and `output_url` per cell. |
| `compare_runs` | `run_a: str`, `run_b: str` | Side-by-side A/B of two runs that ran the SAME task: per framework/harness pair, TPS, run time and status from each. The "which framework/model is faster" tool. |
| `tail_bench_log` | `lines: int = 60`, `only_errors: bool = False` | Tail of `logs/bench.log` — the primary forensics source for what any harness or framework actually did. `only_errors` filters to failure-class lines. |
| `framework_log` | `framework: str`, `lines: int = 60`, `only_errors: bool = True` | Tail of a framework server log (`omlx`, `mtplx`, `mlxvlm`, `mlxserve`), newest file for that framework. Server-side failures (memory guard, stream stalls, load failures) show up here, not in `bench.log`. |

**Mutating (change benchmark state; described as MUTATES in the tool docs):**

| Tool | Parameters | What it does |
|---|---|---|
| `start_campaign` | `sets: list[str]`, `harnesses: list[str] = []` | Starts a campaign over whole tasks: every task × the given model sets, frameworks auto-derived per set. `sets` are model-set names from `config.json` (e.g. `["qwen38-27b"]`); `harnesses` defaults to all six (`raw`, `rawplus`, `pi`, `opencode`, `goose`, `hart`). Refuses if a run or campaign is already active. |
| `stop_run` | — | Requests a stop of the current run or campaign. Partial results are kept; a paused campaign can be resumed from the dashboard. |

## Where it can be deployed

Anywhere an MCP host can run, as long as that machine can reach the
backend over HTTP:

- **Same machine as the benchmark (the normal case)** — the host spawns
  `mcp/benchtest_mcp.py` locally and it connects to `127.0.0.1:7090`.
- **A different machine** — set `BENCHTEST_URL=http://<bench-host>:7090`
  in the MCP server's environment. The backend must be listening on a
  reachable interface (`host: "0.0.0.0"` in `config.json`) — note the
  backend has no authentication, so only do this on a trusted network.

Supported hosts (anything that speaks MCP stdio): **ZCode** (CLI and
desktop), **Claude Desktop**, **Claude Code**, **Cursor**, **Windsurf**,
**Cline**, **Continue**, and any other client that lets you register a
stdio MCP server.

## How to deploy

### ZCode (automatic)

```
python3 scripts/register_mcp.py          # registers; creates mcp/venv if needed
python3 scripts/register_mcp.py --remove # unregister
```

The script computes every path from the repo's own location (no hardcoded
paths), writes the user-scope ZCode config (`~/.zcode/cli/config.json`,
with a backup), and prints what it registered. **Restart ZCode**, then
check Settings → MCP lists `benchtest`.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "benchtest": {
      "command": "/absolute/path/to/benchtest/mcp/venv/bin/python",
      "args": ["/absolute/path/to/benchtest/mcp/benchtest_mcp.py"]
    }
  }
}
```

Restart Claude Desktop; a hammer/tools icon should list the benchtest tools.

### Cursor / Windsurf / Cline / other JSON-configured hosts

The shape is the same everywhere — a stdio server whose command is the
venv python and whose single argument is this script:

```json
{
  "mcp": {
    "servers": {
      "benchtest": {
        "command": "/absolute/path/to/benchtest/mcp/venv/bin/python",
        "args": ["/absolute/path/to/benchtest/mcp/benchtest_mcp.py"],
        "env": { "BENCHTEST_URL": "http://127.0.0.1:7090" }
      }
    }
  }
}
```

(`env` is optional — omit it for a local backend. Some hosts name the key
`mcpServers` instead of `mcp.servers`; follow the host's own docs.)

### First-time venv setup (only if `install.sh` wasn't run)

```
python3 -m venv mcp/venv
mcp/venv/bin/pip install -r mcp/requirements.txt   # pinned: mcp==2.2.0
```

## How to invoke it

You never call the tools by hand — **you ask the host's AI agent in plain
language** and it picks the tool. Examples that map directly:

| You say | The agent calls |
|---|---|
| "Is a benchmark running right now?" | `bench_state` |
| "Which cell is executing and how fast is it going?" | `bench_state` |
| "Show me the last 5 runs and their fastest cells." | `list_runs(limit=5)` |
| "Pull up the full record of run 20261006-032316." | `get_run("20261006-032316")` |
| "Compare yesterday's tetris run with today's." | `compare_runs(a, b)` |
| "Any errors in the bench log recently?" | `tail_bench_log(lines=200, only_errors=true)` |
| "Did the MTPLX server log any stalls?" | `framework_log("mtplx", only_errors=true)` |
| "Run the whole qwen38-27b set over every task, raw and pi only." | `start_campaign(["qwen38-27b"], ["raw", "pi"])` |
| "Stop the benchmark." | `stop_run()` |

In ZCode the tools also appear as direct calls (`mcp__benchtest__bench_state`,
…) once the server is registered — you can invoke them from an agent
session or script them, e.g. a monitoring pass:

```
mcp/venv/bin/python tests/mcp_monitor.py    # from the repo root: live pass
```

`tests/mcp_monitor.py` runs a full read-only pass (state, error-class log
lines, framework logs) and prints each MCP call it made — use it to verify
the deployment before wiring it into a host.

## Security

The server is **local-first and unauthenticated on purpose** (same stance as
the dashboard): it exposes whatever the backend exposes, and the backend
listens without auth so it can be reached from a phone on the same network.
Keep deployments on a trusted network; don't point `BENCHTEST_URL` at a
host you don't control.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Host doesn't list `benchtest` after registration | Restart the host; check the command path is absolute and the venv exists (`mcp/venv/bin/python -V`). |
| Tools error with connection refused | The backend isn't running — `python3 server.py` in the repo root — or `BENCHTEST_URL` points at the wrong host/port. |
| `framework_log` returns "no log files for …" | No server log exists yet for that framework — start that framework once (or it isn't a valid name; allowed: `omlx`, `mtplx`, `mlxvlm`, `mlxserve`). |
| `start_campaign` refuses | A run or campaign is already active — check `bench_state`, `stop_run` first. |
| Claude Desktop shows nothing | Config file location/format is per-OS; use Settings → Developer → Edit Config and restart the app fully. |
