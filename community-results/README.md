# Community results

Benchmark results contributed by the community, one folder per machine:
`<github-user>-<machine>` (e.g. `jdoe-m4-16gb/`).

Each folder contains: the run JSONs exported from benchtest (`runs/` →
copy the files for the runs you want to publish), plus a short README
with chip, RAM, OS, the models/frameworks tested, and the settings.
Follow the manifest format in [`../results/`](../results/) so entries
stay comparable. Include the per-cell JSONL files too if you can — they
make your submission crash-auditable.

How to produce a clean set: run your combination, then copy that run's
files from `runs/` (each task writes one `<timestamp>.json` + one
`<timestamp>-cells.jsonl`). The `model` field in every cell records what
each framework actually served — no configuration guesses.

**RAM guidance:** the model picker only offers models that fit your
machine, so a 16 GB Mac simply benchmarks smaller quants — that's a
valid and useful data point, not a limitation.
