#!/usr/bin/env python3
"""Register the benchtest MCP server with ZCode — dynamically.

Computes every path from THIS repository's location (no hardcoded /Users/...),
so it works on any machine and any checkout directory:

    python3 scripts/register_mcp.py           # register / update
    python3 scripts/register_mcp.py --remove  # unregister

Writes the user-scope ZCode config (~/.zcode/cli/config.json → mcp.servers.
benchtest) pointing at mcp/venv/bin/python + mcp/benchtest_mcp.py, creating
the venv first if it is missing.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, "mcp", "venv", "bin", "python")
SERVER = os.path.join(ROOT, "mcp", "benchtest_mcp.py")
ZCODE_CONFIG = os.path.expanduser("~/.zcode/cli/config.json")


def ensure_venv():
    if os.path.isfile(VENV_PY):
        return
    print("creating mcp/venv (one-time)…")
    subprocess.run([sys.executable, "-m", "venv",
                    os.path.join(ROOT, "mcp", "venv")], check=True)
    subprocess.run([VENV_PY, "-m", "pip", "install", "--quiet", "-r",
                   os.path.join(ROOT, "mcp", "requirements.txt")],
                   check=True)


def register(remove=False):
    if not remove:
        ensure_venv()   # unregistering must not create a venv just to delete
    os.makedirs(os.path.dirname(ZCODE_CONFIG), exist_ok=True)
    cfg = {}
    if os.path.isfile(ZCODE_CONFIG):
        with open(ZCODE_CONFIG) as f:
            cfg = json.load(f)
    servers = cfg.setdefault("mcp", {}).setdefault("servers", {})
    if remove:
        servers.pop("benchtest", None)
    else:
        servers["benchtest"] = {
            "type": "stdio",
            "command": VENV_PY,                 # absolute, computed here
            "args": [SERVER],                   # absolute, computed here
        }
    # atomic write with a one-generation backup: this file holds the user's
    # other MCP servers too — a truncated write must not lose them
    backup = ZCODE_CONFIG + ".benchtest-bak"
    if os.path.isfile(ZCODE_CONFIG):
        import shutil
        shutil.copy2(ZCODE_CONFIG, backup)
    tmp = ZCODE_CONFIG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, ZCODE_CONFIG)
    print(("removed" if remove else "registered") + " benchtest in",
          ZCODE_CONFIG)
    if not remove:
        print(json.dumps(servers["benchtest"], indent=1))
    print("restart ZCode, then Settings → MCP should list 'benchtest'.")


if __name__ == "__main__":
    register(remove="--remove" in sys.argv)
