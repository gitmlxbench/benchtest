# 🗳️ Vote: shape V2 — what should benchtest benchmark next?

V2's framework list is a starting ballot, not the limit. **How to vote:**
every candidate below is posted as a **separate comment** — add a 👍
reaction to each one you want. Reaction count = votes. Write-ins: add a
new comment in the matching category and it becomes votable too.

Tallies are reviewed for each V2 milestone; top picks move onto the
[roadmap](https://github.com/gitmlxbench/benchtest#roadmap) in order.

---

## Category 1 — Serving frameworks

(Each option below should be posted as its own comment by the maintainer;
voters react 👍.)

**On the ballot to start:**
- llama.cpp (llama-server)
- Ollama
- LM Studio (local server mode)
- vLLM — Qwen MTP builds (blocked on upstream support; vote to show demand)

**Likely write-ins:** MLC-LLM · TensorRT-LLM metal builds · other MLX
runtimes · anything with an OpenAI-compatible /v1 endpoint and a
CLI-launchable server.

## Category 2 — Model families / quants for frozen campaigns

Vote for families worth a full published campaign. Include the RAM class
you'd run them on (16 / 32 / 64 / 128 GB) in a reply if you can —
smaller-RAM configs are first-class.

**On the ballot to start:**
- Qwen3.8 smaller quants (8-16 GB class)
- Llama 4.x local quants
- Gemma family (beyond the vLLM reference build)
- MoE small-active models for 32-64 GB Macs

## Category 3 — Agent harnesses

**On the ballot to start:**
- Claude Code (local-endpoint mode)
- Codex CLI
- Cline / Continue (headless modes)
- Gemini CLI (OpenAI-compat endpoint)

---

**Rules of thumb:** eligible = OpenAI-compatible `/v1` endpoint +
CLI-launchable server, reproducible by a community member on their own
machine. Be civil; one account, one reaction per comment (GitHub enforces
this naturally).
