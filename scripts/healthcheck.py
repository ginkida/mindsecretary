#!/usr/bin/env python3
"""Healthcheck: verify process is alive and DB is accessible."""
import sqlite3
import subprocess
import sys
from pathlib import Path

DB_PATH = Path("/app/data/mindsecretary.db")


def main() -> int:
    # Check main process is running
    result = subprocess.run(
        ["pgrep", "-f", "mindsecretary.app"],
        capture_output=True,
    )
    if result.returncode != 0:
        print("FAIL: mindsecretary process not running")
        return 1

    # Check database is accessible
    if not DB_PATH.exists():
        print("FAIL: database file not found")
        return 1
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=3)
        conn.execute("SELECT 1 FROM memories LIMIT 1")
        conn.close()
    except Exception as e:
        print(f"FAIL: database not accessible: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
