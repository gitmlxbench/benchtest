#!/usr/bin/env bash
#
# benchtest — one-shot setup check for Apple Silicon.
#
# benchtest itself has no Python dependencies (stdlib only). This script
# verifies the runtime prerequisites, seeds config.json from the example on
# first run, and tells you exactly what (if anything) is missing.
#
# Usage:  ./install.sh
#
set -u   # not -e: this is a check-and-report script

cd "$(dirname "$0")"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
FAILURES=0

have() { command -v "$1" >/dev/null 2>&1; }

echo
echo "  benchtest — prerequisite check"
echo "  ----------------------------------------"

# --- Python (required) ---
PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
  fail "python3 not found on PATH — install Python 3.10+ (brew install python)"
else
  PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  PYOK="$("$PY" -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)' 2>/dev/null)"
  if [ "$PYOK" = "1" ]; then
    ok "python $PYVER ($PY)"
  else
    fail "python $PYVER is too old — benchtest needs 3.10+ (brew install python)"
  fi
fi

# --- Framework CLIs (optional: only needed for the frameworks you run) ---
echo
echo "  Framework CLIs (install only what you benchmark):"
check_cli() { # name  hint
  if have "$1"; then ok "$1 — $(command -v "$1")"; else warn "$1 not found — $2"; fi
}
check_cli omlx      "brew install omlx  (or see the OMLX repo)"
check_cli mtplx     "see the MTPLX repo for install"
check_cli mlx-serve "npm i -g mlx-serve  (or the MLX-Serve repo)"
check_cli vllm      "pip install vllm  (dev 0.31 for Apple Silicon — optional framework)"
check_cli aider     "python3 -m pip install aider-chat  (aider.chat)"

# MLX-VLM needs a python interpreter with the mlx_vlm package.
if "$PY" -c 'import mlx_vlm' >/dev/null 2>&1; then
  ok "mlx_vlm importable by $PY"
else
  warn "mlx_vlm not importable by $PY — MLX-VLM framework will be skipped"
  warn "   (pip install mlx-vlm into the interpreter you point 'mlxvlm' at)"
fi

# --- Agent harnesses (optional: only needed for the agent rows) ---
echo
echo "  Agent harness CLIs (optional):"
check_cli pi        "npm i -g @earendil-works/pi-coding-agent"
check_cli opencode  "curl -fsSL https://opencode.ai/install | bash"
have node && ok "node — $(command -v node)" || warn "node not found (needed by pi/opencode)"

# --- Bundled hart harness (ships with benchtest — no install needed) ---
echo
echo "  Bundled harness:"
if [ -f hart/hart.py ]; then
  ok "hart — bundled at hart/hart.py (no install needed)"
else
  fail "hart/hart.py missing — the bundled harness should ship with this repo"
fi

# --- Seed config.json on first run ---
echo
if [ -f config.json ]; then
  ok "config.json already present (left untouched)"
else
  if [ -f config.example.json ]; then
    cp config.example.json config.json
    ok "seeded config.json from config.example.json — edit it to match your machine"
  else
    warn "config.example.json missing — server will fall back to built-in defaults"
  fi
fi

# --- Runtime dirs ---
mkdir -p runs outputs logs work harness-configs 2>/dev/null
ok "runtime dirs ready (runs/ outputs/ logs/ work/ harness-configs/)"

# --- Optional: MCP server venv (benchtest tools for ZCode / other hosts) ---
if [ -x mcp/venv/bin/python ]; then
  ok "mcp/venv already present (python3 scripts/register_mcp.py to register)"
elif python3 -m venv mcp/venv 2>/dev/null; then
  # pinned in mcp/requirements.txt so an SDK major can't shift under us
  if mcp/venv/bin/pip install --quiet -r mcp/requirements.txt 2>/dev/null; then
    ok "mcp venv ready (python3 scripts/register_mcp.py to register with ZCode)"
  else
    warn "mcp venv created but the 'mcp' package failed to install (optional)"
  fi
else
  warn "could not create mcp/venv (optional component — bench works without it)"
fi

echo
echo "  ----------------------------------------"
echo "  Next:"
echo "    1. Edit config.json  (models, ports, start_cmd per framework)"
echo "    2. python3 server.py            (or: make run)"
echo "    3. Open http://localhost:7090"
echo
# warnings are advisory (optional components); FAILURES are hard misses and
# make the script exit nonzero so CI can catch them
exit $((FAILURES > 0))
