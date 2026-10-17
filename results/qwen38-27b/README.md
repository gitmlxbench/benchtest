# Qwen 3.8 27B (4 frameworks) — frozen publication results

- Tasks (3): agentconsole, codereview, tetris
- Frameworks (4): mlxserve, mlxvlm, mtplx, omlx × 6 harnesses (raw, raw+, pi, opencode, aider, hart)
- Cells: 72 total — 72 done, 0 errors, 0 skipped
- Uniform settings: ctx 131072, max_tokens 65536, one model per framework, one resident at a time

## Models (exactly what each framework served, from the cell records)
- omlx: Jundot/Qwen3.8-27B-oQ8e-mtp (oQ8e quant)
- mtplx: Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality
- mlxvlm: Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality
- mlxserve: Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality

Note: OMLX runs its oQ8e quant of the same 27B model — OMLX cannot read the MTPLX store format; this is noted per-framework, not a like-for-like weights mismatch beyond quantization.

## Metric notes for graphing
- Agent rows (pi/opencode/aider/hart): chart server-measured decode (server_tgs); client tps includes tool-execution time.
- iterations = model calls per cell (pi/opencode/hart emit step counts; aider = LLM-history blocks).
- Raw rows (Raw (api), raw+) are the clean apples-to-apples comparison.
- QA score = deterministic reliability checks on the artifact (delivered, substantial, complete, self-test, untruncated); judgment stays human.
- Error inventory: none
