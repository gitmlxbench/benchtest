---
name: benchtest-operator
description: Operate, monitor and debug the benchtest local-LLM benchmark (Apple Silicon; OMLX/MTPLX/MLX-VLM/MLX-Serve/vLLM × raw/raw+/pi/opencode/Aider/hart). Use when the user asks to run, monitor, compare or debug benchmark runs, or when benchmark artifacts or logs need triage.
---

# benchtest operator

## What this is — and what a "skill" is

A **skill** is a folder containing a `SKILL.md`: instructions that an AI
agent loads *on demand*. Unlike a prompt you type every time, a skill is
**persistent, versioned knowledge that travels with the agent**: the host
(ZCode, or any agent runtime that reads `~/.agents/skills/`) matches the
skill's `description` against what you ask, and pulls the whole playbook
into the agent's context at the moment it becomes relevant. Skills carry
*procedure and judgment* — what to check, in what order, and what the
findings mean. They are the complement to **MCP servers**, which carry
*capability* — the actual tools an agent can call. The benchtest MCP server
(`mcp/`) is the hands; this skill is the brain that knows what to do with
them.

## Why this skill exists

Running the benchmark well requires knowledge that took a long time to earn,
most of it learned from failures:

- which of the two logs to read first (the orchestrator's `bench.log` vs the
  framework's own server log) for each class of failure;
- what an `empty response` really means (a thinking model exhausting the
  harness's output cap — a *setting*, not a broken model);
- that `config.json` is rewritten by the backend's in-memory state and
  editing it while the backend runs is silently reverted;
- that MTPLX only reads its own store, so a freshly downloaded HF-cache
  model will never appear in its dropdown.

None of that is in the code, and a new user (or a fresh AI agent session)
would otherwise re-learn it by failing for days. This skill encodes the
operator experience so that **any agent equipped with it behaves like the
person who debugged all of this** — from the very first session.

## What it lets an end user do

With this skill installed (and ideally the benchtest MCP server connected),
you can hand the benchmark to your AI agent in plain language:

- **Operate** — "run the tetris task against OMLX with raw and pi",
  "start a campaign over the qwen38-27b set", "stop the run" — the agent
  knows the endpoints and the order of operations.
- **Monitor** — "keep an eye on the campaign, tell me if anything fails" —
  the agent knows where state lives (`/api/state`) and what a healthy cell
  looks like.
- **Triage** — "the snake run shows `empty response` for every pi cell" —
  the agent follows the runbook below: state → classify → artifact →
  targeted fix, instead of guessing or re-running everything.
- **Avoid the traps** — it will not edit `config.json` under a live backend,
  will not blame a framework for a harness token cap, and knows where each
  framework's models must be placed on disk.

The end result for a community user: **an expert operator on day one**,
without reading the source or repeating the project's debugging history.

## The benchmark in one paragraph

Local LLM benchmark on one Mac. A web backend (port 7090) drives framework
servers (OMLX :7001, MTPLX :7002, MLX-VLM :7003, MLX-Serve :7004), runs agent
harnesses against them, and collects the produced artifacts for review.

### Mental model

- **Model set** = one model per framework (`config.json → model_sets`). Two
  shipped families: `qwen38-27b` (all 4 MLX-side frameworks; vLLM picks its own compatible HF repos) and `flash-next` (MTPLX
  gets the Speed build, OMLX the 4bit-mtp). Day-to-day selection happens in
  the dashboard's Models panel (folder → dropdown → Validate).
- **Run** = one task × the selected frameworks × harnesses. Each cell = one
  framework/harness pair. **Campaign** = sets × all tasks, sequential.
- **Reliability QA** (`qa.py`, deterministic): artifact delivered/substantial/
  complete, self-test N/M result, untruncated → 0-100 with hover breakdown.
  No probes, no heuristics; the human judgment stays with the artifact.
- Tasks (deliberately small): tetris (game logic), codereview (planted
  defects), markdown (compiler + spec tests), agentconsole (agent + skills +
  MCP deploy), plus a custom-prompt option. markdown and codereview are the
  best model/harness discriminators.

### Key files

| Path | What |
|---|---|
| `config.json` | frameworks, model_sets, reasoning, ports |
| `runs/<ts>.json` | one completed run: per-cell status/run time/tokens/tps/error |
| `runs/*-cells.jsonl` | per-cell records appended as each cell FINISHES (crash-safe) |
| `outputs/<oid>.html` | artifacts (referenced by `output_url`) |
| `logs/bench.log` | orchestrator log — the forensics source |
| `logs/<fw>-<ts>.log` | per-framework server logs |
| `campaign.json` | campaign state (resume/complete/paused) |

Backend HTTP API (port 7090): `GET /api/state`, `GET /api/runs_history`,
`GET /api/frameworks`, `POST /api/run`, `POST /api/stop`,
`POST /api/campaign {action: start|resume|cancel, mode, sets, harnesses}`.
If the benchtest **MCP server** is connected, prefer its tools —
`bench_state`, `list_runs`, `get_run`, `compare_runs`, `tail_bench_log`,
`framework_log`, `start_campaign`, `stop_run` — they wrap exactly these
endpoints.

### How to install this skill

Copy (or clone) the folder into the agent runtime's skills directory and it
is discovered automatically — no registration, no restart in most hosts:

```
mkdir -p ~/.agents/skills
cp -r skills/benchtest-operator ~/.agents/skills/
```

The agent triggers it when a request matches the `description` above —
running, monitoring, comparing or debugging benchmark runs, or triaging
benchmark artifacts/logs. Pair it with the benchtest MCP server
(see `mcp/README.md` for per-host deployment) so the agent can *act* and
not just advise.

## Triage runbook (run this top-to-bottom on any failure)

1. **State first**: `GET /api/state` (or the MCP `bench_state` tool) — is a
   run/campaign active, which cell?
2. **Classify the error** from the cell's `error` string:
   - `empty response` → the model spent the harness's output cap on hidden
     thinking before writing content. Check the thinking budget (OMLX
     per-model settings) and the harness cap (pi 32K, opencode 32K).
   - `timeout after 7200s` → the harness ran out of time; scan the bench.log
     window: was it progressing (tool calls streaming) or stalled?
   - MTPLX `mtplx_stream_stall_break` in the server log → MTPLX killed a
     silent thinking phase at its 300s watchdog (`--stream-stall-deadline-s`).
   - `exited 1` / `no output` → the harness CLI died silently; check the
     framework server log (MCP `framework_log`) for whether a request even
     arrived.
   - OMLX `memory-guard` / `forced to SSD` → model does not fit; check
     resident GB vs the wired ceiling (~121.6 GB on 128 GB machines).
3. **Open the artifact** (`output_url` → Open ▸ in the dashboard) and judge
   it yourself: does the game render/animate, does the self-test panel pass,
   does the console connect? The artifact is the ground truth — there is no
   score to over-trust.
4. **Only then change code/settings**, and prefer, in order: harness setting,
   framework flag, model swap. Re-verify by re-running the single cell, not
   the whole run.

## Pitfalls (hard-won rules)

- `config.json` is rewritten by the backend's in-memory state (`save_config`
  on every model-set apply). **Never edit it while a backend lives** — the
  edit will be silently reverted. Sequence: stop backend → edit → start.
  (`~/.omlx/model_settings.json` is safe to edit while OMLX runs; the backend
  restarts OMLX on the file's mtime change.)
- pi and opencode cap per-request output (32K); thinking models can exhaust
  that on hard tasks before emitting content — that reads as `empty response`
  and is a cap setting, not a model bug.
- Cells that hit the harness token cap log `finish=length` truncation — a
  thinking model spent the budget on reasoning; raise the cap or the task's
  reasoning level, don't blame the framework.
- A fresh model download must land where each framework reads it: OMLX and
  MLX-VLM/MLX-Serve read the HF cache; **MTPLX only reads `~/.mtplx/models`**.
- One model is loaded at a time per framework: starting a run sweeps any
  resident model first. If a cell ran against "the wrong model", check
  `bench_state`/the framework card for what was actually served before
  concluding anything about quality.
