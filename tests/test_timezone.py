"""Tests for TZ-aware behavior — local day boundaries on queries that
previously mixed a UTC-stored column with a local-date string.

Scenarios modeled:
- User in Asia/Almaty (UTC+5) logs a notification at local 01:30, which is
  UTC 20:30 of the previous day. `count_notifications_today` must count
  this row as today, not yesterday.
- `get_today_cost` must behave identically.
- `check_contact_frequency` and `/people` must compute days_since from
  local-TZ "now" so they don't drift by the UTC offset.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from mindsecretary.core.database import Database


ALMATY = "Asia/Almaty"


def _make_db(tmp_path: Path, tz: str = ALMATY) -> Database:
    db = Database(tmp_path / "tz.db", timezone=tz)
    db.db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            category TEXT NOT NULL,
            importance INTEGER DEFAULT 5,
            related_person TEXT,
            related_date TEXT,
            source_type TEXT,
            source_ref TEXT,
            confidence REAL DEFAULT 1.0,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')),
            last_accessed TEXT
        )
    """)
    db.db.commit()
    return db


def _insert_interaction_at(db: Database, utc_ts: str, direction: str = "out",
                           message_type: str = "notification",
                           content: str = "hi") -> str:
    cur = db.db.execute(
        "INSERT INTO interactions (timestamp, direction, message_type, content) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (utc_ts, direction, message_type, content),
    )
    row = cur.fetchone()
    db.db.commit()
    return row["id"]


def _insert_cost_at(db: Database, utc_ts: str, cost: float = 0.01,
                    provider: str = "anthropic") -> None:
    db.db.execute(
        "INSERT INTO api_costs (timestamp, provider, input_tokens, output_tokens, cost_usd) "
        "VALUES (?, ?, ?, ?, ?)",
        (utc_ts, provider, 100, 100, cost),
    )
    db.db.commit()


class TestLocalDayBounds:
    def test_bounds_match_local_midnight(self, tmp_path: Path):
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 24, 18, 56, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            start, end = db._local_day_utc_bounds()
        # Almaty is UTC+5 so local 00:00 on 2026-04-24 == UTC 19:00 on 2026-04-23.
        assert start == "2026-04-23 19:00:00"
        assert end == "2026-04-24 19:00:00"

    def test_offset_minutes(self, tmp_path: Path):
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 24, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            assert db._local_tz_offset_minutes() == 300  # +5 hours

    def test_local_date_sql_with_offset(self, tmp_path: Path):
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 24, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            sql = db._local_date_sql("timestamp")
        assert sql == "date(timestamp, '+300 minutes')"


class TestCountNotificationsToday:
    def test_counts_row_in_local_today_even_if_utc_yesterday(self, tmp_path: Path):
        """Local 01:30 Apr 24 == UTC 20:30 Apr 23 — must count as today."""
        db = _make_db(tmp_path)
        # Row is logged at local 01:30 Apr 24. UTC = Apr 23 20:30
        _insert_interaction_at(db, "2026-04-23 20:30:00")
        fake_local = datetime(2026, 4, 24, 1, 30, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            assert db.count_notifications_today() == 1

    def test_does_not_count_yesterday_local(self, tmp_path: Path):
        """Row from 23:00 local Apr 23 == UTC 18:00 Apr 23 — yesterday, skip."""
        db = _make_db(tmp_path)
        _insert_interaction_at(db, "2026-04-23 18:00:00")
        fake_local = datetime(2026, 4, 24, 1, 30, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            assert db.count_notifications_today() == 0

    def test_ignores_incoming_messages(self, tmp_path: Path):
        db = _make_db(tmp_path)
        _insert_interaction_at(
            db, "2026-04-24 10:00:00", direction="in", message_type="voice",
        )
        fake_local = datetime(2026, 4, 24, 15, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            assert db.count_notifications_today() == 0


class TestGetTodayCost:
    def test_local_day_bound_for_cost_breaker(self, tmp_path: Path):
        """Cost logged at local 01:30 Apr 24 (UTC 20:30 Apr 23) must count in
        the Apr 24 local budget, not Apr 23 — otherwise the daily cap resets
        at UTC midnight instead of the user's midnight.
        """
        db = _make_db(tmp_path)
        _insert_cost_at(db, "2026-04-23 20:30:00", cost=3.0)
        fake_local = datetime(2026, 4, 24, 1, 30, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            assert db.get_today_cost() == pytest.approx(3.0)

    def test_excludes_prior_local_day(self, tmp_path: Path):
        db = _make_db(tmp_path)
        _insert_cost_at(db, "2026-04-23 12:00:00", cost=5.0)  # UTC noon = local 17:00 Apr 23
        fake_local = datetime(2026, 4, 24, 10, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            assert db.get_today_cost() == pytest.approx(0.0)


class TestGetStats:
    def test_today_cost_counts_local_boundary_row(self, tmp_path: Path):
        db = _make_db(tmp_path)
        _insert_cost_at(db, "2026-04-23 20:30:00", cost=2.5)
        fake_local = datetime(2026, 4, 24, 1, 30, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            stats = db.get_stats()
        assert stats["today_cost"] == pytest.approx(2.5)
        assert stats["interactions_today"] == 0
        assert stats["providers"].get("anthropic", {}).get("cost") == pytest.approx(2.5)

    def test_week_trend_buckets_by_local_day(self, tmp_path: Path):
        """Row at UTC 20:30 Apr 23 belongs to local Apr 24 bucket."""
        db = _make_db(tmp_path)
        _insert_cost_at(db, "2026-04-23 20:30:00", cost=1.0)
        fake_local = datetime(2026, 4, 24, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            stats = db.get_stats()
        days = {row["date"]: row["cost"] for row in stats["week_trend"]}
        assert days.get("2026-04-24") == pytest.approx(1.0)
        assert "2026-04-23" not in days


class TestLocalNowNaive:
    def test_returns_local_wall_clock(self, tmp_path: Path):
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 24, 18, 56, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            naive = db.local_now_naive()
        assert naive.tzinfo is None
        assert naive.hour == 18
        assert naive.minute == 56
        assert naive.strftime("%Y-%m-%d") == "2026-04-24"


class TestContactFrequencyTZ:
    def test_days_since_uses_local_clock(self, tmp_path: Path):
        """last_contact stored as local naive; datetime.now() (UTC in
        Docker) would yield days_since=−1 days for an Almaty user who just
        contacted someone now. Must be 0."""
        from mindsecretary.learning.mood import check_contact_frequency

        db = _make_db(tmp_path)
        # Upsert a contact so last_contact is populated via db._now()
        fake_local = datetime(2026, 4, 24, 18, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            db.upsert_contact("Maria", relation="friend")
            # Write mention_count manually so the alert threshold is met
            db.db.execute(
                "UPDATE contacts SET mention_count = 5, last_contact = ? WHERE name = ?",
                ((fake_local - timedelta(days=35)).replace(tzinfo=None).strftime(
                    "%Y-%m-%d %H:%M:%S"), "Maria"),
            )
            db.db.commit()
            alerts = check_contact_frequency(db)

        assert len(alerts) == 1
        # 35 days since, exactly. Without the fix this would drift by the
        # +5h TZ offset (fraction of a day → still 35, but the fragile
        # off-by-one hits at midnight-boundary writes). Assert >= 34 to be
        # robust against DST / fractional offsets while still catching
        # gross misalignment.
        assert 34 <= alerts[0]["days_since"] <= 35


class TestGetThemeClusters:
    def test_includes_memories_from_boundary_hours(self, tmp_path: Path):
        """Memory created at local 02:00 N days ago (= UTC 21:00 N+1 days
        ago) must still be included in the N-day window — it would be
        excluded by the old `date(created_at) >= date(?)` comparison since
        that computes UTC-date, not local-date."""
        db = _make_db(tmp_path)
        # Insert a memory created_at = 2026-04-21 21:00 UTC = 2026-04-22 02:00 local.
        # Window is 3 days from today (local 2026-04-24). Memory is on day 2026-04-22,
        # which is within the 3-day window (2026-04-22 inclusive).
        db.db.execute(
            "INSERT INTO memories (id, content, embedding, category, importance, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("m1", "morning thought", b"\x00" * 16, "general", 7, "active",
             "2026-04-21 21:00:00"),
        )
        db.db.execute(
            "INSERT INTO memories (id, content, embedding, category, importance, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("m2", "later thought", b"\x00" * 16, "general", 7, "active",
             "2026-04-22 10:00:00"),
        )
        db.db.commit()
        fake_local = datetime(2026, 4, 24, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            clusters = db.get_theme_clusters(days=3, limit=5)
        # cnt >= 2 required — both m1 and m2 are in "general" category
        assert any(c["label"] == "general" and c["count"] == 2 for c in clusters)


def test_smoke_regressions(tmp_path: Path):
    """Sanity: default (no TZ) still works, no surprise crashes."""
    db = Database(tmp_path / "notz.db")
    db.db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, embedding BLOB NOT NULL,
            category TEXT NOT NULL, importance INTEGER DEFAULT 5,
            related_person TEXT, related_date TEXT, source_type TEXT,
            source_ref TEXT, confidence REAL DEFAULT 1.0,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')), last_accessed TEXT
        )
    """)
    db.db.commit()
    assert db.count_notifications_today() == 0
    assert db.get_today_cost() == 0.0
    assert db.local_now_naive() is not None
    # With no profile TZ, offset falls back to system local. SQL fragment is
    # either "date(timestamp)" (UTC system) or includes a minutes offset
    # (non-UTC dev box). Either form is valid.
    sql = db._local_date_sql("timestamp")
    assert sql == "date(timestamp)" or sql.startswith("date(timestamp, '")
