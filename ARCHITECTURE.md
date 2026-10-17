# benchtest — Architecture

A single-process, stdlib-only Python backend serves a single-file HTML dashboard and
orchestrates local LLM inference frameworks and agent harnesses. This document is the
authoritative map of the system.

## 1. High-level flow

```mermaid
flowchart TB
    UI["Browser dashboard<br/>index.html — pick frameworks, harnesses, model, task"]

    API["Python backend<br/>server.py — stdlib only, one process"]

    FW["Local inference server<br/>OMLX · MTPLX · MLX-VLM · MLX-Serve<br/>(one at a time)"]

    H["Harness<br/>raw · raw+ · pi · opencode · aider · hart"]

    M[("Models on disk<br/>HF cache + MTPLX store")]

    R[("Run history<br/>runs/*.json")]

    UI -->|"1 · HTTP / JSON"| API
    API -->|"2 · start / stop"| FW
    API -->|"3 · run task"| H
    H -->|"4 · /chat/completions"| FW
    FW -->|"load weights"| M
    API -->|"5 · save"| R
    API -->|"6 · live results"| UI
```

Read it top to bottom:

1. The **browser** talks to the **backend** over plain HTTP/JSON (no websockets, no
   build step — the UI polls every 1.2s).
2. The **backend** starts (or reuses) a **local inference server** for the current
   framework, and stops it before moving to the next one — so frameworks never compete
   for RAM/GPU.
3. For each selected **harness**, the backend runs the task. The harness (a raw HTTP
   call, or an agent CLI subprocess) sends the prompt to the framework's
   OpenAI-compatible `/chat/completions` endpoint.
4. The framework **loads the model** from disk (HF cache, or the MTPLX store for MTPLX)
   and generates the completion.
5. The backend **saves** the finished run to `runs/*.json`.
6. Results stream back to the **browser** live as each cell completes.

Model discovery (which models exist, and which fit in your RAM) is a read-only side
channel: `discovery.py` scans the same on-disk stores and feeds the model picker — it
never touches the inference path.

## 2. Process & threading model

```mermaid
flowchart LR
    MAIN[main thread<br/>ThreadingHTTPServer.serve_forever]
    subgraph Threads["daemon threads (one per request / run)"]
        T1[HTTP handler thread<br/>per connection]
        T2[run_benchmark thread<br/>the active benchmark]
    end
    LOCK[(LOCK — single RLock<br/>guards STATE / PROCS / results)]
    PLOCK[(PROXY_LOCK<br/>guards proxy counters)]

    MAIN --> T1
    MAIN --> T2
    T1 -.acquire.-> LOCK
    T2 -.acquire.-> LOCK
    T2 -.acquire.-> PLOCK
```

- **One process, many threads.** `ThreadingHTTPServer` gives each HTTP connection its
  own thread. The benchmark itself runs on a single daemon thread spawned by
  `POST /api/run`.
- **One global `LOCK` (RLock)** serializes all reads/writes of shared state
  (`STATE`, `PROCS`, `FW_LOGS`, results). Handlers take it for short critical sections;
  the orchestrator takes it around state transitions. Long work (subprocess waits,
  HTTP calls) is done **outside** the lock.
- **`PROXY_LOCK`** separately guards the measurement proxy's counters/log.
- **`RUN_FLAG` (Event)** is the stop signal: `POST /api/stop` clears it; the
  orchestrator checks it between cells and the harness runners check it while polling
  subprocesses, so Stop works mid-flight.

## 3. A benchmark run, step by step

```mermaid
sequenceDiagram
    participant U as User (browser)
    participant H as Handler
    participant O as run_benchmark
    participant F as Framework mgr
    participant S as Framework server
    participant R as Harness runner
    participant M as Metrics

    U->>H: POST /api/run {frameworks, harnesses, task, prompt, settings}
    H->>H: _validate_run (400 on bad input)
    H->>O: spawn daemon thread
    H-->>U: {ok:true}

    loop for each framework
        O->>F: start_framework(fw)
        F->>S: Popen(start_cmd)  [or reuse if healthy]
        F->>S: poll /v1/models until it answers
        F-->>O: up
        loop for each harness
            O->>R: run_harness(fw, harness, prompt, settings)
            R->>S: OpenAI /chat/completions (raw / raw+)
            R->>R: or spawn pi/opencode/aider/hart (agents)
            R->>M: collect TPS/TGS/PP/latency/tokens
            M->>M: save the artifact to outputs/ (extension preserved)
            R-->>O: result row
            O->>O: append to STATE.results (under LOCK)
        end
        O->>F: stop_framework(fw)  [SIGTERM → SIGKILL]
    end
    O->>O: save runs/*.json (under LOCK)
    Note over U: browser polls /api/state every 1.2s and renders live
```

## 4. The six harnesses

| Harness | What it is | How it's driven | Metrics source |
|---------|-----------|-----------------|----------------|
| **raw** | One-shot OpenAI `/chat/completions` call | `call_chat()` (urllib, streaming) | client-measured TTFT/decode; server counters where available |
| **raw+** | raw with automatic continuation on `finish_reason=length` | `rawplus_generate()` — re-issues with the tail + a "continue at the cut point" instruction, de-dups the seam | client-measured, summed across rounds |
| **pi** | pi coding agent | `pi --print --provider bench --model …` with an **isolated** `PI_CODING_AGENT_DIR` | agent stdout + `HART_RESULT`-style totals |
| **opencode** | opencode agent | `opencode run --dir <workdir> --model bench/<id>` with isolated `XDG_CONFIG_HOME` | agent stdout + artifact |
| **aider** | aider agent | `aider --model openai/<id>` whole-format, per-cell workdir | `XDG_CONFIG_HOME` + `OPENAI_BASE_URL` | agent stdout + artifact |
| **hart** | the `hart` agentic harness | `python3 <hart_path>` (config from `hart_path`) | `HART_RESULT {…}` line on stdout |

The **`hart`** harness is **bundled** in this repo under `hart/` (so a fresh clone
works out of the box); `hart_path` defaults to `./hart/hart.py`. The other agent
harnesses (`pi`, `opencode`, `aider`) are community tools the user installs themselves.

All agent harnesses are pointed at the framework via an **OpenAI-compatible base URL**
(`http://127.0.0.1:<port>/v1`) and a throwaway API key. Each gets an **isolated config
dir** under `harness-configs/` so the user's global agent configs are never touched.

## 5. Metrics pipeline

```mermaid
flowchart LR
    subgraph Sources
        A[Client timing<br/>call_chat: TTFT, decode, wall]
        B[Server counters<br/>/metrics JSON or Prometheus]
        C[Measurement proxy<br/>per-request PP/TGS/TTFT]
    end
    A --> M[per-cell row]
    B -->|server_snapshot diff| M
    C -->|optional| M
    M --> D[derived: ctx_fill_pct, est_gbps]
    D --> T[results table + chart + CSV]
```

- **Client timing** (`call_chat`) measures TTFT (time to first token) and decode time
  directly from the streaming response, and reports `wall` (total) so raw+ can compute
  an honest decode rate.
- **Server counters** (`server_snapshot` / `server_cell_delta`) read the framework's
  metrics endpoint (JSON `/metrics` for MLX-VLM, Prometheus text for MLX-Serve) before
  and after a cell and diff the counters — giving true server-side PP/TGS.
- **Measurement proxy** (optional, `route_via_proxy: true`) sits between the agent
  harnesses and the framework and records per-request PP/TGS/TTFT, so agent rows get
  the same metric depth as raw rows.
- **Derived** values: `ctx_fill_pct` (prompt tokens / configured window) and
  `est_gbps` (TGS × weight size) as an effective-bandwidth estimate.

## 6. Model discovery & compatibility

`discovery.py` answers "which models can this machine actually run right now?":

```mermaid
flowchart TB
    subgraph Sources
        L[Local HF cache<br/>scan ~/.cache/huggingface/hub]
        M[MTPLX store<br/>scan ~/.mtplx/models]
        S[Served /v1/models<br/>live, per framework]
    end
    L --> MG[merge + de-dup<br/>normalize_key: org/name == org--name]
    M --> MG
    S --> MG
    MG --> TAG[tag by model_source<br/>hf → omlx/mlxvlm/mlxserve<br/>mtplx → mtplx]
    TAG --> C[compatibility scoring]
    RAM[free RAM<br/>vm_stat / /proc/meminfo, 3s cache] --> C
    CTX[configured ctx_tokens] --> C
    C --> V[verdict per model]
    V -->|ready| G[green — fits RAM, ctx OK, cached]
    V -->|tight| A[amber — fits, but ctx < configured]
    V -->|too-large| R[red — weights + KV headroom > free RAM]
    V -->|unknown| U[grey — not cached, would download]
```

- **Three sources, de-duplicated.** The live `/v1/models` (authoritative while a server
  runs) is merged with the local HF cache **and** the MTPLX store. Served ids use
  `org--name`; cache ids use `org/name`; `normalize_key()` maps both to the same
  cache-dir key so they collapse.
- **MTPLX has its own model store.** MTPLX models live in `~/.mtplx/models` (a different
  format — MTP sidecar + runtime — not plain HF/MLX repos), so they're discovered
  separately and tagged `model_source: "mtplx"`. The UI greys out MTPLX models for
  non-MTPLX frameworks and vice-versa, based on each framework's `model_source`.
- **RAM fit** uses `size_gb × 1.25` (weights + KV cache + engine overhead) vs current
  free RAM. Free RAM is read from `vm_stat` (macOS) or `/proc/meminfo` (Linux) and
  cached for 3s (the UI polls every 1.2s).
- **Context fit** reads the model's declared context from the cached `config.json`
  (top-level or nested `text_config` for multimodal models) and compares to the
  configured window.
- **MTP draft** detection surfaces a cached speculative-decoding draft matching the base
  model, so the UI can show it.

## 7. Configuration & persistence

```mermaid
flowchart LR
    DEF[Built-in DEFAULT_CONFIG] --> MG[_merge_config]
    EX[config.example.json] -->|first run: seed| CJ[config.json]
    CJ --> MG
    MG --> LIVE[CONFIG (in memory)]
    LIVE -->|POST /api/models/select| CJ
    LIVE --> FW[FRAMEWORKS dict]
```

- **Layered config.** Built-in defaults ← `config.json` (per-machine). Frameworks are
  merged per-framework so a partial `config.json` only overrides what it names.
- **Self-seeding.** On first run, `config.json` is copied from `config.example.json`
  so users can discover and edit it.
- **Live model selection.** `POST /api/models/select` updates the in-memory config and
  persists it back to `config.json` — the UI's model picker writes real config.
- **Run history.** Each completed run is saved to `runs/<timestamp>.json` with the full
  config snapshot (so results are reproducible) and all result rows.

## 8. Security & robustness posture

- **Local-only by default.** `host: 127.0.0.1`; the server logs a loud warning if you
  bind to `0.0.0.0`.
- **Input validation.** `POST /api/run` is validated (`_validate_run`) before a thread
  is spawned; bad input → HTTP 400, not a 500. All JSON bodies are size-capped.
- **No global config mutation.** Agent harnesses use isolated config dirs
  (`PI_CODING_AGENT_DIR`, per-harness `XDG_CONFIG_HOME`) — the user's `~/.pi`,
  `~/.config/opencode`, etc. are never written.
- **Graceful failure.** Every HTTP route is wrapped so errors return JSON (never an HTML
  traceback). Framework start/stop is SIGTERM→SIGKILL with process-group kills. A
  `SIGTERM` to the server shuts down any frameworks it spawned.
- **No secrets.** The throwaway API key is the literal string `bench`; there is no
  network egress except to `127.0.0.1` and (for discovery) the local HF cache.
