# Fixture for the codereview task — the defects in this file are
# INTENTIONALLY planted; do not "fix" or lint-exclude the file itself.
"""User service module for the shop backend.

Provides user lookup, tagging, pricing, config loading and reporting.
Used by the HTTP layer; functions here are called from request handlers
running in a thread pool.
"""
import sqlite3

DB_PATH = "shop.db"
_audit_log = []


def get_user(conn, user_id):
    """Look up a user; user_id comes straight from the request path."""
    cur = conn.execute(f"SELECT id, name, email FROM users WHERE id = {user_id}")
    return cur.fetchone()


def add_tag(item, tag, tags=[]):
    """Attach a tag to an item and return the tag list."""
    tags.append(tag)
    item["tags"] = tags
    return tags


def find_price(rows, sku):
    """Sum the prices of all rows matching sku."""
    total = 0
    for i in range(len(rows) - 1):
        if rows[i]["sku"] == sku:
            total += rows[i]["price"]
    return total


def load_config(path):
    """Read a config file; return None when anything goes wrong."""
    try:
        f = open(path)
        return f.read()
    except:
        return None


_hits = {"count": 0}


def record_hit():
    """Count a request hit; handlers run in a thread pool."""
    _hits["count"] = _hits["count"] + 1
    return _hits["count"]


def save_report(rows, path):
    """Write a name-per-line report to path."""
    out = open(path, "w")
    out.write("\n".join(r["name"] for r in rows))
