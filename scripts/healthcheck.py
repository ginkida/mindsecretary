#!/usr/bin/env python3
"""Healthcheck: verify DB file exists and is queryable.

In Docker the bot runs as PID 1 — if the process dies, the container dies
and Docker's restart policy handles it. A `pgrep` liveness check is both
redundant (PID 1 == the app) AND broken on `python:3.12-slim` (procps not
installed). DB accessibility is the real signal: the app may be alive but
unable to read/write if the volume is unmounted or the file is corrupted.
"""
import os
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(os.environ.get("MINDSECRETARY_ROOT", "/app"))
DB_PATH = _ROOT / "data" / "mindsecretary.db"


def main() -> int:
    if not DB_PATH.exists():
        print(f"FAIL: database file not found at {DB_PATH}")
        return 1
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=3)
        conn.execute("SELECT 1 FROM memories LIMIT 1")
        conn.close()
    except Exception as e:
        print(f"FAIL: database not accessible: {type(e).__name__}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
