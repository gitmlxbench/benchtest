# benchtest — Troubleshooting

Quick fixes for the common failure modes. If something's wrong, check **`logs/bench.log`**
first — every framework start/stop, cell result, and error is timestamped there.

## The server won't start

**"Could not bind 127.0.0.1:7090"**
Another benchtest server (or a framework) is already using the port.
- Stop the other server, or
- Use a different port: `python3 server.py 7100` (or change `port` in `config.json`).

**"config.json unreadable"**
Your `config.json` has a JSON syntax error. The server falls back to built-in defaults
so it still starts — fix the JSON (a trailing comma is the usual culprit) and restart.

## A framework shows red (down) or never comes up

- **CLI not found** — the log says `<name> CLI not found`. Install the CLI or fix
  `start_cmd` in `config.json`. `./install.sh` lists what's missing.
- **Server crashed on start** — the framework's own stdout/stderr is captured to
  **`logs/<fw>-<timestamp>.log`**. Open the newest one for that framework; it has the
  real error (OOM, bad flag, model not found, port in use).
- **Model not in cache** — if the model isn't in `~/.cache/huggingface/hub`, the
  framework may try to download it (slow) or fail. Use the *Model per Framework* panel
  to pick a model that's already cached (verdict **ready**).
- **Port already in use** — if something else is on the framework's port, benchtest
  *reuses* it if it answers `/v1/models` healthily, otherwise the start fails. Free the
  port or change the framework's `port`.

## A cell errors out

- **"empty response"** — the model produced no visible content. On thinking models the
  usual cause: hidden reasoning consumed the harness's per-call output cap before any
  content appeared. Fix: cap the thinking budget in the framework's model settings
  (OMLX: `thinking_budget_tokens`), raise the harness cap, or accept it as a capability
  datum. raw/raw+ are rarely affected (65K caps).
- **"timeout after 7200s"** — the harness ran out of time. Scan `logs/bench.log` over
  the cell's window: steady tool calls = ran out of time honestly; silence = a stall.
- **`mtplx_stream_stall_break` in the MTPLX server log** — MTPLX killed a stream that
  streamed nothing for its deadline (default 300s). Thinking phases longer than that
  get killed; raise `--stream-stall-deadline-s` (the shipped config uses 900).
- **"exited 1" / "no output"** — the harness CLI died without a message. Check the
  framework server log: if no request even arrived, the failure is client-side.
- **OMLX `507 Cannot load … projected memory`** — another model is resident (often a
  GUI-selected one). Unload it (or restart OMLX) and rerun the cell.
- **"stopped by user"** — you hit Stop; expected.
- **Truncated (⚠)** — the response hit `max_tokens`. Raise *Max tokens* and
  rerun, or use **raw+** which auto-continues past the cap automatically.
- **Artifact looks wrong** — open it via the **Artifacts** link and judge visually;
  the tool records throughput, the artifacts are yours to review.

## The model picker shows everything as "too-large"

Your **free RAM** is low (another app or a running framework is using it). The verdict
is `size_gb × 1.25 > free RAM`. Close other apps / stop running frameworks, or pick a
smaller model. Total RAM is shown in the top strip; the verdict uses *free* RAM.

## Agent harness (pi / opencode / aider / hart) rows look empty or fail

- **CLI not installed** — install it (`./install.sh` shows the command).
- **`hart` not found** — set `hart_path` in `config.json` to the real path of `hart.py`.
- **Isolated config** — agents run with an isolated config dir under `harness-configs/`.
  If you changed how an agent authenticates, note that benchtest uses a throwaway key
  (`bench`) and points the agent at `http://127.0.0.1:<port>/v1`.
- **pi thinking** — if pi rows behave differently, check `pi_thinking` in `config.json`.

## Metrics look wrong or missing

- **PP/TGS show "–" for a cell** — that framework doesn't expose server-side counters
  for that harness, and it wasn't routed through the proxy. Set `route_via_proxy: true`
  (and restart) to get per-request PP/TGS/TTFT on agent rows.
- **raw+ TGS looks off** — raw+ sums decode time across continuation rounds; a high
  continuation count (see the `continuations` field) means the task exceeded one
  `max_tokens` budget.

## The UI looks stale / broken

- **Hard-refresh** (Cmd+Shift+R) — the frontend is a single file; a stale cache is the
  usual cause.
- **Restart the backend** if an endpoint 404s (the endpoint is newer than your running
  server).
- The UI polls `/api/state` every 1.2s; if the server is restarting, the poll silently
  retries — give it a moment.

## Reset everything

```bash
make clean        # removes runs/, outputs/, logs/, work/, harness-configs/*
rm config.json    # (optional) force re-seed from config.example.json on next start
```
