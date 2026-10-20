# hart v2 — a fully agentic, multi-model, parallel build harness

hart takes a goal ("build a Tetris game in one HTML file"), talks to one or
more LLMs — **local Apple-Silicon backends and/or frontier cloud models** — and
runs it to a **validated, tested, documented deliverable** through a pipeline of
**independent agents** that hand their work to each other **through files on
disk**.

v2.0 is a complete rewrite of the v1 single-file sequential harness (preserved
as [`hart_v1.py`](hart_v1.py)). It lifts v1's two non-goals — *no parallel /
multi-agent concurrency* and *no cloud providers* — and makes every stage a
separately-modeled agent. **Read [`DESIGN.md`](DESIGN.md) for the full
architecture.**

Zero third-party dependencies — Python 3.9+ standard library only.

---

## What's new in v2

| | v1 | v2 |
|---|---|---|
| Stages | 5 functions, one file | **7 agent modules** (`hart/agents/*.py`) |
| Models | one backend per run | **per-stage model** — local builds, frontier plans/critiques/evaluates |
| Backends | local OpenAI-compatible | **local + frontier cloud** (OpenAI-compatible *and* Anthropic) |
| Handoff | in-memory state | **file-based**: each stage writes `.hart/stages/<n>-<stage>.json` |
| Concurrency | sequential | **parallel** where inputs are independent |
| Planning | single-shot | **planner ⇄ critic validation loop** |
| Delivery | one validation pass | **final Evaluation agent + fix loop on any test failure** |
| Filesystem | jailed scratch dir | **unrestricted by default** (any folder / git repo); `--sandbox` opt-in |

---

## GUI — the hart console

```bash
pip install PySide6          # the GUI's only dependency
python3 -m hartgui           # from the repo root
```

A dark, pipeline-first console (think opencode), organized as **three
pages in a left nav** so the first look is never overwhelming:

- **⚙ Setup** — everything configurational: backends and per-stage models.
- **▶ Run** — goal/prompt, run controls, the live pipeline bar, and the
  live tail.
- **📂 Results** — what each stage fed the next (the exact JSON), plus
  artifacts on disk with reveal-in-Finder. The app switches here
  automatically when a run finishes.

- **Backends** panel — add any number of local (`http://127.0.0.1:PORT/v1`)
  *and* cloud (OpenAI/Anthropic) backends; Discover probes `/v1/models`.
- **Per-stage models** — use one model everywhere, or a different model
  per stage: e.g. a frontier model for planning and evaluation, a local
  Qwen for building. Planning, building, validation, evaluation, summary —
  each is a dropdown.
- **Planning** — a stage like any other: pick its model on the Setup page
  (e.g. a frontier cloud model to plan, a local model to build). The plan's
  steps stream into the live tail as they're produced.
- **Pipeline bar** — connection → planning → building → validation →
  evaluation → summary, each chip shows live state with a pulsing active
  dot and per-stage elapsed time.
- **Live tail** — timestamped, color-coded narration streaming from every
  agent while it runs, with follow-mode: plan steps as they're produced,
  every file the builder writes, and the final deliverables — **file paths
  are clickable** (directories reveal in Finder, files open in the default
  app). Artifacts land in the run's workdir; stage I/O in `.hart/stages/`.
- **Dark / light theme** — one-click toggle in the header; the whole
  palette (pipeline chips, tail colors, dialogs) switches.
- **⏸ Pause at next stage** — click it while a run is on: the pause fires
  **strictly between stages** (never interrupts the running stage). A
  review dialog opens at the next stage boundary — and every boundary
  after — showing the stage's exact output as editable JSON; an optional
  **prompt note redirects the next stage** (for example: skip the
  scoreboard, focus on game-over logic). Continue, or Abort. The edited
  output is what the next stage actually receives.

Everything the CLI can run, the GUI can — it *is* the harness (the agents
run in-process; the GUI just drives them off the main thread).

## Quick start

```bash
# 1. point at a local backend (or a cloud model) in your config
cp examples/hart.config.json ~/.hart/models.json     # then edit base_url / model / keys

# 2. run a goal
python3 hart.py "build a single-file HTML page with a heading, a working button, and an animated canvas"

# 3. inspect the result
cat REPORT.md
ls .hart/stages/          # the file handoffs
```

`python3 -m hart "…"` is equivalent. On first run, if no config exists, hart
seeds `~/.hart/models.json` from the built-in local presets (omlx / mtplx /
mlxserve / mlxlm).

### Multi-model in one command

```bash
# local model builds, a frontier model plans, critiques, and evaluates
python3 hart.py --framework local \
    --model-for-stage planning=frontier \
    --model-for-stage planning_critic=frontier \
    --model-for-stage evaluation=frontier \
    "build a REST todo API in Python with tests"

# list what's configured
python3 hart.py --list-models
```

---

## The pipeline

```
connection → planning ⇄ critic → building → validation → evaluation ⇄ (fix loop) → summary
```

Each arrow is a **file handoff** in `<workdir>/.hart/stages/`:

| Stage | Agent | Default model | Writes |
|---|---|---|---|
| connection | `ConnectionAgent` | every configured model | `00-connection.json` |
| planning | `PlannerAgent` | `stages.planning` | `01-plan.json` |
| planning ⇄ critic | `CriticAgent` | `stages.planning_critic` | `01-critic.json` |
| building | `BuilderAgent` | `stages.building` | `02-build.json` + artifacts |
| validation | `ValidatorAgent` | `stages.validation` | `03-validation.json` |
| evaluation | `EvaluatorAgent` | `stages.evaluation` | `04-evaluation.json` |
| summary | `SummaryAgent` | `stages.summary` | `05-report.md` + `REPORT.md`/`README.md` |

### The two validation loops

- **Planning loop** — if a `planning_critic` model is configured (and differs
  from the planner), the plan is sent to the critic. On `revise`, the feedback
  goes back to the planner; repeat until `approve` or `budget.plan_critic_rounds`
  is spent. This is "a plan from one model is validated by another."
- **Evaluation fix loop** — the final `EvaluatorAgent` runs the project's real
  test suite (auto-detected, or `--test-command`) plus a model review. **If any
  test fails**, the Builder enters fix-mode with the exact failures, then
  re-validates and re-evaluates, until green or `budget.eval_fix_rounds` is
  spent. This is "if any test from one model or other failed, the loop starts to
  fix the issue."

### Parallelism

Independent work runs concurrently on a bounded thread pool (`--parallel N`,
default 4):

- **connection** probes every configured model at once;
- **building** runs plan steps with no unmet dependencies and no shared outputs
  in parallel (the dependency graph is built from each step's `depends_on` and
  `outputs`);
- **validation / evaluation** run deterministic checks and independent model
  reviews in parallel.

---

## Configuration

Resolved in order: `--config PATH` → `./hart.config.json` → `~/.hart/models.json`.
The v1 shape (`{"frameworks": {…}}`) is read transparently.

```jsonc
{
  "models": {
    "local":    {"provider":"openai","base_url":"http://127.0.0.1:7001/v1",
                 "model":"…","api_key":"bench","ctx_tokens":262144},
    "frontier": {"provider":"openai","base_url":"https://api.openai.com/v1",
                 "model":"gpt-4o","api_key_env":"OPENAI_API_KEY"},
    "claude":   {"provider":"anthropic","base_url":"https://api.anthropic.com",
                 "model":"claude-sonnet-4-20250514","api_key_env":"ANTHROPIC_API_KEY"}
  },
  "default_model": "local",
  "stages": { "planning":"local", "planning_critic":"frontier",
              "building":"local", "evaluation":"frontier" },
  "budget": { "max_steps":40, "per_call_max_tokens":16384,
              "plan_critic_rounds":2, "eval_fix_rounds":3 },
  "parallel": 4,
  "sandbox": false,
  "test_command": null
}
```

- `provider` is `openai` (any `/v1/chat/completions` endpoint) or `anthropic`
  (`/v1/messages`).
- `api_key_env` keeps secrets in the environment instead of the file.
- A stage omitted from `stages` uses `default_model`.

---

## CLI

```
python3 hart.py [task] [options]

  --config PATH            config file
  --framework NAME         use this model for every stage
  --base-url URL           override the default model's endpoint
  --model NAME             override the default model's id
  --model-for-stage S=N    set one stage's model (repeatable)
  --workdir PATH           project folder / git repo to work in
  --resume                 resume an interrupted run in --workdir
  --plan-file PATH         use this plan JSON instead of the Planning agent
  --dry-run                plan only; print the plan and stop
  --parallel N             max parallel model calls / steps
  --sandbox                confine file/shell to the workdir
  --test-command CMD       command the Evaluation agent runs as the test suite
  --no-critic              disable the planning critic loop
  --stage NAME             run ONLY this stage (its input files must exist)
  --list-models            list configured models and exit
  --verbose / --quiet      narration level
  runs                     (positional) list past runs under --workdir's parent
```

---

## The GUI (native Mac client)

A native Mac GUI client (PySide6/Qt) that **is** the harness — it drives the
same agents in-process. It adds what the CLI doesn't:

- **Model discovery** — point at a framework URL, hit *Discover*, and hart lists
  the models that endpoint serves. Add as many backends as you like.
- **Per-stage model selection** — one model for every stage, or a different
  model per stage (planning / building / validation / evaluation / summary):
  e.g. a frontier cloud model plans and evaluates, a local Qwen builds.
- **Live pipeline view** — per-stage status dots + a narration log.

### Run it

```bash
# the GUI needs PySide6 (the harness core stays zero-dependency)
python3 -m venv .venv-gui && source .venv-gui/bin/activate
pip install PySide6
python3 -m hartgui
```

### Use it

1. **Backends** (left): fill in the form and click **Add / Update**:
   - *Name* (e.g. `omlx`, `openai`), *URL* (e.g. `http://127.0.0.1:7001/v1` or
     `https://api.openai.com/v1`), *Provider* (`openai`-compatible or `anthropic`),
     optional *Model* id, and the **API key** — type it directly, or put the
     *env var name* in **Key env** (e.g. `OPENAI_API_KEY`) to keep the secret out
     of the file. Local backends need no key; cloud ones do.
   - Click the backend in the list, then **Discover** to list its models
     (OpenAI-compatible servers expose a model list; for Anthropic, set the
     *Model* id — it has no list). Clicking a backend loads it into the form so
     you can edit it.
2. **Goal** (center): type the prompt.
3. **Stage models**: *Use one model for every stage*, or set each stage's
   model (planning included).
4. **Run** — watch the pipeline bar and live tail, then read the result on
   the Results page. **Stop** aborts in-flight calls; **⏸ Pause at next
   stage** gates the run between stages for review.

### Package as a `.app`

```bash
# Briefcase turns it into a native macOS app bundle + .dmg
pip install briefcase
briefcase create        # first time (generates the app spec)
briefcase build macos
# → dist/hart-gui/macos-arm64/hart-gui-<ver>.dmg
```

(PyInstaller also works: `pyinstaller --windowed --name hart hartgui/main.py`.)

---

## Filesystem policy (no boxing)

By default hart is **unrestricted**: `--workdir` can be any local folder or git
repo, file actions accept relative *or* absolute paths, and `run_shell` has no
command deny-list. This is a local tool on the owner's machine — **an
unrestricted agent can modify anything the OS user can.** For untrusted tasks,
use `--sandbox` (confines paths to the workdir + a conservative shell
deny-list) or run under a throwaway user/container.

---

## Testing

```bash
python3 tests/test_pure.py          # 35 unit tests (protocol, actions, qa, config, plan, dep-graph)
python3 tests/test_orchestrator.py  # 5 end-to-end tests against a mock model server
python3 tests/test_gui_core.py      # GUI-core tests (discovery, planning, pipeline)
# GUI offscreen tests need the GUI venv (PySide6):
source .venv-gui/bin/activate && QT_QPA_PLATFORM=offscreen python3 tests/test_gui_offscreen.py
```

The integration tests run the **full pipeline** against a tiny OpenAI-compatible
mock (`tests/mock_model.py`) and assert: artifacts created, the critic loop ran,
independent steps built in parallel, the evaluation fix loop converged, and
resume skips completed stages. No real model or network is needed.

---

## Layout

```
hart.py                  # thin entry point (keeps the historical path working)
hart_v1.py               # the preserved v1 single-file harness (reference)
hart/
  config.py              # Model, ModelRegistry, config loading (local + cloud, v1 compat)
  providers.py           # OpenAI + Anthropic streaming chat, shared SSE parser
  protocol.py            # lenient JSON-action extraction
  blackboard.py          # control state + file handoffs, atomic write, lock, resume
  actions.py             # file + shell tools (sandbox-aware)
  qa.py                  # deterministic QA gate + test-suite detection
  context.py             # role prompts + repo inventory
  orchestrator.py        # the pipeline, the two loops, parallelism, resume
  cli.py                 # argparse + main()
  agents/                # one module per stage
    base.py  connection.py  planner.py  critic.py
    builder.py  validator.py  evaluator.py  summary.py
hartgui/                 # the native Mac GUI (PySide6; optional layer)
  mainwindow.py          # the main window (backends, stages, plan compare, log)
  workers.py             # QThread workers (pipeline + discovery, off the GUI thread)
  best_of_n.py           # single-model planning + OMLX residency sweep
  pipeline.py            # GUI pipeline (per-stage models, stage gates)
  model_config.py        # backend + model discovery
  main.py                # entry point (python3 -m hartgui)
examples/hart.config.json
tests/
```
