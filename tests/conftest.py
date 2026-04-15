"""Shared fixtures for MindSecretary tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mindsecretary.core.database import Database


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    """Create a fresh in-memory-like Database backed by a temp file."""
    db = Database(tmp_path / "test.db")
    # Create the memories table (normally done by Memory class) so
    # Database.get_stats() doesn't fail on missing table.
    db.db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            category TEXT NOT NULL,
            importance INTEGER DEFAULT 5,
            related_person TEXT,
            related_date TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')),
            last_accessed TEXT
        )
    """)
    db.db.commit()
    return db


@pytest.fixture
def raw_conn(tmp_db: Database) -> sqlite3.Connection:
    """Expose the raw sqlite3 connection for assertions."""
    return tmp_db.db
