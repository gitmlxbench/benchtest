# benchtest — Configuration reference

All configuration lives in **`config.json`** at the project root. On first run it is
seeded from **`config.example.json`**. The server merges `config.json` over built-in
defaults, so you only need to specify the keys you care about.

> **Live editing:** the dashboard's *Model per Framework* panel writes `model` (and
> `ctx_tokens`) straight back to `config.json` via `POST /api/models/select`. Other
> keys are edited by hand, then restart the server.

## Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Interface the web UI binds to. **Keep `127.0.0.1`** — binding `0.0.0.0` exposes run control and run deletion to your whole network (the server warns loudly if you do). |
| `port` | int | `7090` | Port for the web UI. Override per-invocation with `python3 server.py <port>`. |
| `route_via_proxy` | bool | `false` | Route agent harnesses (pi/opencode/aider/hart) through the local measurement proxy so their rows get per-request PP/TGS/TTFT. `false` = agents connect straight to the framework (byte-identical to an independent harness run; agent rows then report TPS only). |
| `pi_thinking` | string | `""` | pi thinking-level suffix appended to `--model` (e.g. `":high"`). Empty = inherit the server's reasoning setting so all cells share one uniform budget. |
| `reasoning` | object | `{"short": "low", "long": "low"}` | Reasoning level per task type, applied at framework server start. Standard tasks run at the `short` level, LONG tasks (chip8, raytracer, spreadsheet, markdown, conduit) at the `long` level. Each framework maps a level to its own parameter via `reasoning_flags` (CLI flags) or `reasoning_file` (omlx's per-model settings file). See below. |
| `hart_path` | string | `"./hart/hart.py"` (bundled) | Path to the hart harness. Defaults to the bundled copy; override only to use your own. |
| `cell_timeout_s` | int | `7200` | Per-cell wall-clock cap in seconds (clamped 60–21600). A cell that exceeds it is stopped, its partial artifact kept, and the row recorded as a timeout error. Dashboard runs can override per run via *Max cell run time (min)*. |
| `frameworks` | object | (see below) | One entry per framework. Merged per-framework over the defaults. |

## `frameworks.<id>` keys

Each framework has a stable id (e.g. `omlx`, `mtplx`, `mlxvlm`, `mlxserve`). The MLX-VLM framework's id was renamed from `mlxlm` to `mlxvlm` — the wrapped engine is mlx-vlm (MLX-LM is a different project without MTP support); old configs are migrated on load. You can
**add** new frameworks or **remove** ones you don't use — the UI renders whatever is
here.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | string | yes | Display name (e.g. `"OMLX"`). |
| `model_source` | string | no | Where the framework loads models from: `"hf"` (the Hugging Face cache, the default) or `"mtplx"` (`~/.mtplx/models`). This controls which models the *Model per Framework* picker offers for that framework — MTPLX models are only offered to `model_source: "mtplx"` frameworks and vice-versa. |
| `port` | int | yes | Port the framework's inference server listens on. benchtest waits for `/v1/models` here before running, and reuses a healthy server already on this port. |
| `model` | string | yes | The model id the framework serves / should serve. This is the id used in requests. (Some frameworks normalize it — e.g. MTPLX serves `mtplx-qwen38-27b-optimized-quality`.) |
| `repo` | string | no | The Hugging Face repo behind a normalized served id (used for cache mapping in discovery and for the `{repo}` placeholder). Set this when `model` is a normalized id that differs from the repo name. For MTPLX this is also the id used to load the model from `~/.mtplx/models`. |
| `start_cmd` | string[] | yes | The exact command to cold-start the framework's server. May use the placeholders `{model}`, `{model_path}`, and `{reasoning}` (see below). |
| `reasoning_flags` | object | no | Maps a reasoning level (`"low"`, `"medium"`, `"high"` — all three are pre-wired for every framework) to the CLI flag list that implements it (e.g. `"medium": ["--reasoning-effort", "medium"]`). Injected where `{reasoning}` appears in `start_cmd`. If the level for the task type has no entry, the server starts without reasoning flags (and logs a warning). |
| `reasoning_file` | string | no | For frameworks with no CLI reasoning flag (omlx): the per-model settings file the level is written into before the run (`~/.omlx/model_settings.json`). The benchmark rewrites the served model's `chat_template_kwargs[<reasoning_key>]` for the run and restores the original value afterwards. |
| `reasoning_key` | string | `"reasoning_effort"` | The `chat_template_kwargs` key used when `reasoning_file` is set. |
| `reasoning_override` | string | no | Per-framework pin: replaces the task-type reasoning level for that framework only (e.g. MTPLX is pinned `low` — its optimized build degrades at medium+). |
| `ctx_tokens` | int | no | Configured context window. Used for compatibility scoring and the ctx-fill % metric. |
| `model_gb` | number | no | Approximate weight size in GB. Used for RAM-fit scoring and the effective-bandwidth estimate. |
| `notes` | string | no | Free-text shown in the framework info modal — document *when* each parameter applies (per-request vs at-start). |

### Placeholders in `start_cmd`

- `{model}` → the value of `model` (the served/request id).
- `{repo}` → the value of `repo` (or `model` if `repo` is unset). Use this for CLIs whose
  `--model` flag needs the HF repo id rather than the normalized served id — this is how
  MTPLX loads a model from `~/.mtplx/models`.
- `{model_path}` → the resolved local HF-cache snapshot directory (a real path), when
  the model (or its `repo`) is cached; otherwise falls back to `model`. Use this for
  CLIs whose `--model` flag needs a real path rather than a repo id.
- `{reasoning}` → the task-type reasoning flags: `reasoning_flags[<level>]`, where the
  level comes from the top-level `reasoning` map (`short` for standard tasks, `long`
  for LONG tasks). Expands to nothing when the level has no mapping. Because the
  level is baked in at server start, switching task types between runs restarts the
  framework server automatically when needed.
- `{ctx}` → the model set's `ctx_tokens` (context window). Model sets pin this so a
  model with a smaller native window doesn't inherit a larger one the other sets
  run with; switching sets restores the covered frameworks' values.
- `{max_tokens}` → the model set's `max_tokens` (output cap). Must stay below the
  window so harness input budgets (`ctx − max_tokens`) remain positive.

### Example (from `config.example.json`)

```json
"omlx": {
  "name": "OMLX",
  "model_source": "hf",
  "port": 7001,
  "model": "mlx-community--Qwen3.8-27B-8bit",
  "model_gb": 28,
  "ctx_tokens": 131072,
  "start_cmd": ["omlx", "serve", "--port", "7001"]
}
```

MTPLX loads from its own store (`~/.mtplx/models`) by repo id, so it uses
`model_source: "mtplx"` and the `{repo}` placeholder. Reasoning is task-type driven
via `{reasoning}` + `reasoning_flags` (the `short`/`long` levels; both default `low`):

```json
"mtplx": {
  "name": "MTPLX",
  "model_source": "mtplx",
  "port": 7002,
  "model": "mtplx-qwen38-27b-optimized-quality",
  "repo": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
  "model_gb": 28,
  "ctx_tokens": 131072,
  "reasoning_flags": {
    "low": ["--reasoning-effort", "low"],
    "medium": ["--reasoning-effort", "medium"]
  },
  "start_cmd": ["mtplx", "serve", "--model", "{repo}",
                "--context-window", "131072", "--max-tokens", "65536",
                "{reasoning}", "--port", "7002"]
}
```

omlx applies reasoning per model from its settings file at request time (no CLI
flag), so it uses `reasoning_file`/`reasoning_key` instead — the benchmark rewrites
the entry before the run and restores it after:

```json
"omlx": {
  "name": "OMLX",
  "model_source": "hf",
  "port": 7001,
  "model": "mlx-community--Qwen3.8-27B-8bit",
  "model_gb": 28,
  "ctx_tokens": 131072,
  "reasoning_file": "~/.omlx/model_settings.json",
  "reasoning_key": "reasoning_effort",
  "start_cmd": ["omlx", "serve", "--port", "7001"]
}
```

## Adding a new framework

1. Add an entry under `frameworks` with a new id.
2. Set `name`, `port`, `model`, and `start_cmd`.
3. Restart the server. The new framework appears in the top strip, the selection grid,
   and the model picker automatically.

## Removing a framework

Delete its entry under `frameworks` and restart. (Its port is then free for anything
else.)

## What is *not* in config.json

- **Tasks / prompts** — defined in `server.py` (`TASKS`) and editable live in the UI.
- **Harness list** — defined in `server.py` (`HARNESS_LABELS`); the UI renders it from
  `/api/harnesses`.
- **Run history** — written to `runs/*.json`, not config.
