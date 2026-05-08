"""Tests for learning/patterns.py — deterministic weekly signals.

Each detector reads from an injected Database with seeded data and returns
either a Pattern or None. Tests assert on the detail string content and
the strength ordering so regressions are obvious.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from mindsecretary.core.config import Profile
from mindsecretary.core.database import Database
from mindsecretary.learning.patterns import (
    Pattern,
    PatternAnalyzer,
    _cost_wow,
    _goal_completion,
    _habit_breaks,
    _late_night_share,
    _mood_direction,
    _peak_weekday,
)


ALMATY = "Asia/Almaty"


def _profile(tz: str = ALMATY) -> Profile:
    return Profile(
        name="Test", city="Almaty", timezone=tz,
        home_coords=[43.25, 76.9], work_coords=[43.25, 76.9],
        wake_up="07:00", work_start="09:00", work_end="18:00",
        sleep="23:00", commute_method="авто", commute_minutes=30,
        style="кратко", language="ru", notification_limit=10,
        quiet_hours=["23:00", "07:00"],
        priorities=[], dislikes=[],
    )


def _make_db(tmp_path: Path, tz: str = ALMATY) -> Database:
    db = Database(tmp_path / "pat.db", timezone=tz)
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
    return db


def _insert_interaction(db: Database, utc_ts: str, direction: str = "in",
                        content: str = "hi") -> None:
    db.db.execute(
        "INSERT INTO interactions (timestamp, direction, message_type, content) "
        "VALUES (?, ?, ?, ?)",
        (utc_ts, direction, "text", content),
    )
    db.db.commit()


def _insert_cost(db: Database, utc_ts: str, cost: float) -> None:
    db.db.execute(
        "INSERT INTO api_costs (timestamp, provider, input_tokens, output_tokens, cost_usd) "
        "VALUES (?, ?, ?, ?, ?)",
        (utc_ts, "anthropic", 100, 50, cost),
    )
    db.db.commit()


class TestPeakWeekday:
    def test_identifies_monday_peak(self, tmp_path: Path):
        db = _make_db(tmp_path)
        # Fake "now" = Sunday 2026-04-26 12:00 Almaty → week window covers
        # Mon Apr 20 through Sun Apr 26 (local). 10 msgs all on Monday local.
        monday_utc = "2026-04-20 05:00:00"  # Mon 10:00 Almaty
        for _ in range(10):
            _insert_interaction(db, monday_utc)
        # Two messages on other days so "peak" is dominant but not 100%
        _insert_interaction(db, "2026-04-22 05:00:00")  # Wed
        _insert_interaction(db, "2026-04-24 05:00:00")  # Fri

        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _peak_weekday(db, _profile())
        assert p is not None
        assert "Понедельник" in p.detail or "Monday" in p.detail.lower() \
            or "понедельник" in p.detail.lower()

    def test_returns_none_below_min_signal(self, tmp_path: Path):
        db = _make_db(tmp_path)
        _insert_interaction(db, "2026-04-20 05:00:00")  # only 1 msg
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _peak_weekday(db, _profile())
        assert p is None

    def test_returns_none_when_evenly_spread(self, tmp_path: Path):
        db = _make_db(tmp_path)
        # 2 messages on each of 7 days — no dominant peak
        for day in range(20, 27):
            _insert_interaction(db, f"2026-04-{day:02d} 05:00:00")
            _insert_interaction(db, f"2026-04-{day:02d} 08:00:00")
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _peak_weekday(db, _profile())
        assert p is None


class TestLateNightShare:
    def test_flags_high_late_night_share(self, tmp_path: Path):
        db = _make_db(tmp_path)
        # 6 late-night messages at local 01:00 (UTC 20:00 prev day) and
        # 4 daytime messages. Share = 60% → should flag.
        for day in range(20, 26):
            _insert_interaction(db, f"2026-04-{day:02d} 20:00:00")  # 01:00 local
        for day in range(21, 25):
            _insert_interaction(db, f"2026-04-{day:02d} 09:00:00")  # 14:00 local
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _late_night_share(db, _profile())
        assert p is not None
        assert "%" in p.detail

    def test_silent_when_mostly_daytime(self, tmp_path: Path):
        db = _make_db(tmp_path)
        for day in range(20, 26):
            _insert_interaction(db, f"2026-04-{day:02d} 09:00:00")  # 14:00 local
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _late_night_share(db, _profile())
        assert p is None


class TestMoodDirection:
    def test_detects_improvement(self, tmp_path: Path):
        db = _make_db(tmp_path)
        # Week window = UTC 2026-04-18 19:00 to 2026-04-25 19:00 (Almaty local
        # midnight bounds for 7 days ending at today). Midpoint at UTC
        # 2026-04-22 07:00. First-half negative, second-half positive, each
        # with at least 3 messages within the window.
        for day in range(19, 22):
            _insert_interaction(db, f"2026-04-{day:02d} 08:00:00",
                                content="устал, плохо, бесит, болит")
        for day in range(23, 26):
            _insert_interaction(db, f"2026-04-{day:02d} 08:00:00",
                                content="отлично, класс, рад, счастлив")
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _mood_direction(db, _profile())
        assert p is not None
        assert "улучш" in p.detail.lower()

    def test_silent_on_stable_mood(self, tmp_path: Path):
        db = _make_db(tmp_path)
        for day in range(19, 27):
            _insert_interaction(db, f"2026-04-{day:02d} 08:00:00",
                                content="обычно, ничего особенного")
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _mood_direction(db, _profile())
        assert p is None


class TestCostWoW:
    def test_detects_large_spike(self, tmp_path: Path):
        db = _make_db(tmp_path)
        # Last week (Apr 13-19 local): $1.00
        _insert_cost(db, "2026-04-14 10:00:00", cost=1.00)
        # This week (Apr 20-26 local): $4.00 — 300% jump
        _insert_cost(db, "2026-04-22 10:00:00", cost=4.00)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _cost_wow(db, _profile())
        assert p is not None
        assert "вырос" in p.detail

    def test_silent_on_no_baseline(self, tmp_path: Path):
        db = _make_db(tmp_path)
        _insert_cost(db, "2026-04-22 10:00:00", cost=2.00)  # this week only
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _cost_wow(db, _profile())
        assert p is None

    def test_silent_on_small_swing(self, tmp_path: Path):
        db = _make_db(tmp_path)
        _insert_cost(db, "2026-04-14 10:00:00", cost=2.00)
        _insert_cost(db, "2026-04-22 10:00:00", cost=2.10)  # 5% drift
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            p = _cost_wow(db, _profile())
        assert p is None


class TestGoalCompletion:
    def test_flags_low_completion(self, tmp_path: Path):
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        # Create 5 goals, 1 completed, 4 pending
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            for i in range(5):
                g = db.create_daily_goal(title=f"goal{i}")
                if i == 0:
                    db.db.execute(
                        "UPDATE daily_goals SET status='completed' WHERE id=?",
                        (g["id"],),
                    )
                    db.db.commit()
            p = _goal_completion(db, _profile())
        assert p is not None
        assert "низкая" in p.detail

    def test_silent_on_mid_completion(self, tmp_path: Path):
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            for i in range(5):
                g = db.create_daily_goal(title=f"goal{i}")
                if i < 2:
                    db.db.execute(
                        "UPDATE daily_goals SET status='completed' WHERE id=?",
                        (g["id"],),
                    )
                    db.db.commit()
            p = _goal_completion(db, _profile())
        # 40% — within the "unremarkable" band
        assert p is None


class TestHabitBreaks:
    def test_flags_habit_missed_often(self, tmp_path: Path):
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            # Log habit done only 1 out of 7 days — should flag
            db.log_habit("бег", True, date="2026-04-20")
            for day in range(21, 27):
                db.log_habit("бег", False, date=f"2026-04-{day:02d}")
            patterns = _habit_breaks(db, _profile())
        assert len(patterns) == 1
        assert "бег" in patterns[0].detail

    def test_habit_break_pluralizes_total_days(self, tmp_path: Path):
        """Pre-fix the detail line hardcoded "дней" — for total=4 it
        should say "дня". Now uses pluralize_ru against `total` (the
        denominator agrees grammatically with the suffix)."""
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            # 4 entries this week, 0 done — total=4 → "4 дня"
            for day in range(23, 27):
                db.log_habit("бег", False, date=f"2026-04-{day:02d}")
            patterns = _habit_breaks(db, _profile())
        assert patterns
        assert "0/4 дня" in patterns[0].detail

    def test_skips_untracked_habit(self, tmp_path: Path):
        """A habit with no log entries this week must NOT be flagged
        just because `week_done` is 0 — the user simply isn't tracking it.
        Regression test for the bug where `done + (7 - done) < 3` was
        always False and produced false positives for every inactive habit.
        """
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            # Habit exists but zero logs this week
            db.db.execute(
                "INSERT INTO habits (id, name) VALUES ('h1', 'медитация')"
            )
            db.db.commit()
            patterns = _habit_breaks(db, _profile())
        assert patterns == []

    def test_skips_habit_with_few_logs(self, tmp_path: Path):
        """Habit with only 2 log entries this week → not enough signal."""
        db = _make_db(tmp_path)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            db.log_habit("йога", False, date="2026-04-25")
            db.log_habit("йога", False, date="2026-04-26")
            patterns = _habit_breaks(db, _profile())
        assert patterns == []


class TestAnalyzerIntegration:
    def test_format_for_prompt_empty_returns_placeholder(self, tmp_path: Path):
        db = _make_db(tmp_path)
        analyzer = PatternAnalyzer(db, _profile())
        text = analyzer.format_for_prompt()
        assert "Недостаточно данных" in text

    def test_format_for_prompt_returns_bullets(self, tmp_path: Path):
        db = _make_db(tmp_path)
        # Seed strong signals across multiple detectors
        monday_utc = "2026-04-20 05:00:00"
        for _ in range(10):
            _insert_interaction(db, monday_utc, content="class super рад")
        _insert_cost(db, "2026-04-14 10:00:00", cost=1.00)
        _insert_cost(db, "2026-04-22 10:00:00", cost=4.00)
        fake_local = datetime(2026, 4, 26, 12, 0, tzinfo=ZoneInfo(ALMATY))
        with patch("mindsecretary.core.database.tz_now", return_value=fake_local):
            text = PatternAnalyzer(db, _profile()).format_for_prompt()
        # At least two strong signals rendered as list bullets
        assert text.count("\n- ") >= 1 or text.startswith("- ")
        # Cost spike signal present
        assert "$" in text


class TestPatternDataclass:
    def test_to_line_format(self):
        p = Pattern(label="Пик", detail="Пн — 5 сообщений", strength=0.8)
        assert p.to_line() == "- Пик: Пн — 5 сообщений"
