# Qwen3.8 Flash-Next (3 frameworks) — frozen publication results

- Tasks (3): agentconsole, codereview, tetris
- Frameworks (3): mlxvlm, mtplx, omlx × 6 harnesses (raw, raw+, pi, opencode, aider, hart)
- Cells: 54 total — 53 done, 1 errors, 0 skipped
- Uniform settings: ctx 131072, max_tokens 65536, one model per framework, one resident at a time

## Models (exactly what each framework served, from the cell records)
- omlx: mlx-community/Qwen3.8-Flash-Next-4bit-mtp
- mtplx: Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed
- mlxvlm: mlx-community/Qwen3.8-Flash-Next-4bit-mtp

MLX-Serve is absent by necessity: its memory preflight requires ~112.5 GB for this build vs ~111 GB available — it does not fit on 128 GB. This is the first complete flash-next pass on MLX-VLM.

## Metric notes for graphing
- Agent rows (pi/opencode/aider/hart): chart server-measured decode (server_tgs); client tps includes tool-execution time.
- iterations = model calls per cell (pi/opencode/hart emit step counts; aider = LLM-history blocks).
- Raw rows (Raw (api), raw+) are the clean apples-to-apples comparison.
- QA score = deterministic reliability checks on the artifact (delivered, substantial, complete, self-test, untruncated); judgment stays human.
- Error inventory: [('agentconsole', 'omlx', 'pi', 'empty response')]
