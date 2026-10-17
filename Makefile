# benchtest — convenience targets. Everything is stdlib Python; no build step.
#
#   make setup   check prerequisites + seed config.json
#   make run     start the benchmark server on :7090
#   make run PORT=7100
#   make check   byte-compile the Python (fast syntax check)
#   make clean   remove runtime artifacts (runs, outputs, logs, work, configs)

PORT ?= 7090
PY   ?= python3

.PHONY: setup run check clean

setup:
	@bash install.sh

run:
	@$(PY) server.py $(PORT)

check:
	@$(PY) -m py_compile server.py discovery.py qa.py mcp/benchtest_mcp.py \
		scripts/register_mcp.py tests/mcp_monitor.py tests/test_pure.py \
		&& echo "syntax OK"
	@$(PY) -m unittest tests.test_pure -v 2>&1 | tail -3

clean:
	@rm -rf runs outputs logs work harness-configs/*
	@touch harness-configs/.gitkeep
	@find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
	@echo "cleaned runtime artifacts (config.json kept)"
