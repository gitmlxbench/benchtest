# hart — Harness for AI-Related Tasks

*stylized **hAIrt** · v0.2.0 · single-file, zero-dependency Python 3.9+*

hart is a fully agentic, resumable harness for local LLM backends on Apple
Silicon. You give it a task; it plans, builds, validates, repairs, and proves
the result — taking whatever time quality requires, narrating every step, and
checkpointing so any run can be resumed. Other harnesses answer; hart
**answers, verifies, repairs, and proves**.

```
python3 hart.py --framework omlx "Write tetris.html: a playable Tetris game"
```

---

## 1. Prerequisites

- macOS with Python 3.9+ (`python3 --version`) — nothing else; stdlib only.
- A running OpenAI-compatible backend. Built-in presets:
  `--framework omlx` (:7001, scottlowry--Qwen3.8-27B-oQ6e-mtp), `--framework
  mtplx` (:7002, mtplx-qwen38-27b-optimized-quality), `--framework mlxvlm`
  (:7003, mlx-community/Qwen3.8-27B-8bit), `--framework mlxserve`
  (:7004, mlx-community/Qwen3.8-27B-8bit — repo name or hash id both work).
  Start hints:
  `--list-frameworks`. Any other endpoint: `--base-url` + `--model`.
- `node` on PATH unlocks the QA gate's JS syntax checks (degrades gracefully
  without it).
- For long unattended runs: `caffeinate -dims python3 hart.py …` prevents
  Mac sleep.

## Interactive TUI (pi-style)

Run `hart.py` with no task and you get a `hart(omlx)>` prompt:

- **Quick questions** (hi, "what are your capabilities", "explain X") are
  answered directly inline — one model call, with tok/s + TTFT shown.
- **Tasks** (anything with write/build/fix/review intent) automatically run
  the **full agentic pipeline with verbose narration**: connection probe →
  planning (the decided steps print) → building (every model call, every
  tool action, 💾 checkpoint markers per call) → validation (verdict +
  self-heal) → summary. 🧹 markers show each context compaction (60 % slim /
  80 % rollup) and which epoch you're in.
- Commands: `/frameworks`, `/framework NAME` (reconnect + probe),
  `/status` (backend + last checkpoint: run/stage/epoch/compactions),
  `/ask` (force quick), `/do` (force pipeline), `/exit`.
- Framework/model config lives in **`~/.hart/models.json`** (seeded from the
  presets on first run — edit ports/models/add backends there, pi-style).
- Long-task results land on disk as always (artifacts + REPORT.md in the
  run workdir) with the summary printed in the TUI.

## 2. How to use it

```bash
# simplest — framework preset + task (positional, like pi)
python3 hart.py --framework omlx "Write calc.html: a working calculator"

# workflows
--workdir DIR           # artifacts + .hart state live here (default ./hart-run-<ts>)
--resume [RUN_ID]       # continue an interrupted run (newest by default)
--plan-file FIXPLAN.md  # execute a plan written by an earlier analysis run
--dry-run               # show the plan + acceptance criteria, build nothing

# safety valves — defaults are hands-off (fastest path first, auto-continue):
--epochs 50             # auto-continuation across budget walls (≈ days unattended)
--time-budget 7200      # wall clock per epoch
--max-steps 40          # model calls per epoch
--repair-rounds 3       # diagnose-and-retry cycles before a run fails
--max-fix-rounds 3      # validator fix-loop rounds
--per-call-tokens N     # default scales with --effort (low 4k / med 8k / high 16k)

# observability
--verbose               # full request/response per model call, pi-style
--quiet                 # only the HART_RESULT machine line
runs                    # list all runs (id, status, task, workdir)
```

### The two-turn repo workflow

```bash
# turn 1 — read-only analysis → analysis/*.md + prioritized FIXPLAN.md
python3 hart.py --framework mtplx --workdir /path/to/repo --max-steps 100 \
  "Perform a full codebase analysis… write a prioritized FIXPLAN.md. Do NOT modify sources."

# turn 2 — execute the plan
python3 hart.py --framework mtplx --workdir /path/to/repo \
  --plan-file FIXPLAN.md --task "Execute the P0 items from FIXPLAN.md"
```

Validated live on a seeded-bug repo: 69 prioritized findings (P0–P3), every
planted bug caught (SQL injection, eval RCE, syntax error, KeyError,
session expiry, placeholder tests); P0 fixes verified in code.

## 3. What to expect

- **Live narration, always**: the plan with acceptance criteria, every model
  call (tokens, tok/s, TTFT), every tool action with ok/ERR, `[next] step N`
  previews — and during long generations a 20 s heartbeat
  (`⏳ model call in flight — 62s, 1 840 tok streamed`). Nothing is a black
  box; add `--verbose` for full wire dumps.
- **Per-call metrics everywhere**: every model call records TTFT, PP, TGS,
  TPS, tokens, **context fill %** (prompt ÷ context window) and an
  **effective-bandwidth estimate (~GB/s)** in `.hart/metrics.jsonl`; the
  narration line shows them live, and `HART_RESULT` carries peak context
  fill and average GB/s.
- **A run ends with**: artifacts in the workdir, `REPORT.md`,
  `README.md` (the build's own docs), `hart-metrics.json`, and stdout's last
  line `HART_RESULT {…}` (status, calls, tokens, tps, QA scores, artifacts)
  for scripting and the benchmark.
- **Typical costs**: simple single-file tasks 4–8 calls / 1–5 min; complex
  multi-file or repo tasks 15–60 calls / 10–90 min. hart spends *more* tokens
  than pi on equivalent tasks (plan + validate + report ceremony) — that is
  the price of verification guarantees; on quality-gated single-artifact
  tasks it matches pi's speed (4 calls, 201–288 s measured) while adding a
  QA gate pi doesn't have.
- **Backend reality**: identical configs measured 14–60 tok/s across days
  (thermal/load/cold-cache swings). The heartbeat tells you instantly whether
  "slow" is the model generating or the backend crawling.
- **Failures are diagnosed, not silent**: every repair, rejection, re-plan,
  and epoch is journaled in `.hart/events.jsonl` with structured types.

## 4. High-level design

Sequential state-machine over a shared **blackboard** — one JSON document
(`<workdir>/.hart/state.json`, written atomically after every transition).
The checkpoint *is* the architecture: any stage can be resumed from disk at
any time; there is no hidden in-memory state.

```
connection → planning → building ⇄ validating → summary
                          ↑ repair cycles / re-plans / fix-loop bounces
```

All model interaction uses one prompted **JSON-action protocol** (no native
tool-calling — that's the portability layer across OMLX/MTPLX/MLX-LM, whose
tool support is inconsistent). Parsing is graduated and **lenient**: strict
JSON → `strict=False` (models embed literal newlines inside strings — the
single biggest repair-loop trigger until fixed) → fenced block → balanced
span, with repair retries that feed the exact error back to the model.

## 5. Detailed flow — what each agent does

**1 · connection** — Probes `/v1/models`, adopts the served id when the
server normalizes names (MTPLX does). Runs a **capability probe**: one tiny
strict-JSON call that calibrates the tokenizer ratio, baselines TTFT/TGS,
and warns *before* a long run if the model can't emit JSON reliably. If probe
TTFT > 5 s (27 GB weights still streaming from disk), it **warms the model
in** so measured calls don't crawl.

**2 · planning** — Decomposes the goal into *as many steps as the task
genuinely needs* (never padded, never collapsed — quality-first). Every step
carries **acceptance criteria** ("all 7 tetrominoes present", "sound plays on
line clear"). A deterministic self-check merges steps that produce the same
file (the #1 source of build churn), caps verbosity, and validates outputs;
if the planner is unusable, a task-type template plan takes over.
`--plan-file` bypasses planning entirely with an external plan.

**3 · building** — Two modes. **Oneshot fast path**: single-step,
single-artifact tasks get one call that writes the complete file (pi-parity
speed). **Batched loop**: multi-step work proceeds one step per batch — the
model emits all of a step's actions in one reply (`write_file` with FULL
content, `read_file` with ranges, `list_files`, jailed `run_shell`, `done`).
Around the loop: per-target read-result retention (no re-reading), a
**step-completion gate** (`done` is accepted only when the step's outputs
pass deterministic checks — quality-gated transitions, not model claims),
a loop ladder (3× nudge → re-plan → repair cycle), rewrite-churn warnings,
budget-escalation notices at 55/75/90 %, and a final rescue-delivery call at
exhaustion.

**4 · validating** — Deterministic gate first, model review second. The gate
is **profile-aware**: canvas/game artifacts get the full interactive
checklist (event handlers, run loop, DOM/canvas use, balanced braces,
`node --check` hard-fail); plain pages are scored on applicable checks with
inline-handler credit. Below 90 % = not usable. The model review then judges
**the goal and the acceptance criteria**, and — the differentiator —
produces *executable* fixes (full corrected files) that hart applies and
re-checks with **zero extra model calls** (self-heal), with a regression
guard that reverts any fix that lowers QA. A progress-aware fix-loop stops
when two rounds produce identical failure signatures.

**5 · summary** (Integrator merged in) — One call produces the integration
gaps, the project README, and the final report (a deterministic fast path
handles clean oneshot builds). Always ends with `HART_RESULT {…}`.

### Cross-cutting machinery

| Mechanism | Behavior |
|---|---|
| Checkpoints | Atomic state writes + append-only `events.jsonl` + per-call `metrics.jsonl`; SIGTERM → state saved, exit 130, `--resume` anywhere |
| Repair cycles (3) | Builder failures become diagnosed retries: exact error + "take a different approach", counters reset, bounded |
| Epochs (default 50) | Budget exhaustion → checkpoint, hard-trim context, reset budgets, continue; a **no-progress guard** stops runs where two consecutive epochs change nothing |
| Context survival | pi-like, so days/weeks runs never exhaust the window. hart rebuilds its prompt from bounded components each call (files live on disk, never in history) — the only growth is observations/results, managed in three stages: **60 %** deterministic slimming (stale read results dropped, artifact supply thinned — free, no model call); **80 %** observation rollup via one model call; **95 %** hard guard that fails into repair instead of sending a doomed request. Working state is trimmed (observations 100, rollups 50, repair log 20) with the full audit in `events.jsonl`; standalone default 336 epochs × 2 h ≈ 4 weeks |
| Loop detection | Identical-action hashes scoped per step; workspace no-progress fingerprint; every escalation journaled |
| Safety | write/run jailed to workdir, deny-pattern shell filter, 60 s caps, path-escape checks, workdir LOCK |

## 6. Capabilities vs other harnesses

| Capability | raw | pi | opencode | aider | **hart** |
|---|---|---|---|---|---|
| Agentic file/shell tools | — | ✅ | ✅ | edits-only | ✅ (jailed) |
| Deterministic QA gate (usable threshold) | — | — | — | — | ✅ |
| Plan with acceptance criteria | — | — | — | — | ✅ |
| Step completion gated on output quality | — | — | — | — | ✅ |
| Self-healing validation (0 extra calls) | — | — | — | — | ✅ |
| Repair cycles on failure | — | partial | partial | — | ✅ bounded+journaled |
| Checkpoint/resume any stage | — | sessions | sessions | git | ✅ blackboard |
| Auto-continue past budgets (epochs) | — | ✅ | ✅ | — | ✅ default-on, guarded |
| Per-call SSE metrics (TTFT/PP/TGS/TPS) | caller | — | — | — | ✅ every call |
| Live narration + heartbeat + wire dumps | — | ✅ | ✅ | — | ✅ all three |
| Framework presets + served-id adoption | — | cfg | cfg | — | ✅ one flag |
| Analyze → plan-file → fix workflow | — | manual | manual | manual | ✅ first-class |
| Zero deps, single file | n/a | npm | npm | pip | ✅ stdlib |

## 7. Benchmark integration

hart is the **fifth harness** in the Apple Silicon LLM Benchmark
(`~/Documents/benchtest`): a chip in the web UI, tested against all three
frameworks. Tokens/calls come from `HART_RESULT`; artifacts feed the same
collector as the other agents; QA is scored by the benchmark's own
(profile-aware, identical) gate — so all five harnesses are compared on
equal terms. Cell bounds: standard tasks 2 epochs × 1 500 s; LONG
industry-benchmark tasks 3 × 2 400 s.

## 8. Limitations (honest)

- Sequential pipeline — agents are state-independent, not concurrent.
- No native tool-calling; the JSON protocol is the portability trade-off.
- Shell actions are quick checks only (60 s, no installs/network).
- Semantic review quality is bounded by the backend model; the deterministic
  gate is the floor, not the ceiling.
- Backend throughput swings (14–60 tok/s observed) dominate wall times.
- No GUI/serve mode, no cross-run project memory (deliberate non-goals).
- v0.2 rename note: runs made before the rename (`.agedIn` state) are not
  auto-resumable by `hart`; their artifacts remain readable.

---

**Archive**: the original frozen design, the architecture snapshot it
outgrew, the external-review adoption log, validation evidence, and the
build-task prompt live in [`docs/archive/`](docs/archive/).
