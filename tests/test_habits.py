"""Tests for habit streak tracking and recurring reminders."""
from __future__ import annotations

from datetime import timedelta

import pytest

from mindsecretary.core.database import Database


class TestHabitStats:
    def test_no_habits_returns_empty(self, tmp_db: Database):
        assert tmp_db.get_habit_stats() == []

    def test_streak_counting(self, tmp_db: Database):
        # Anchor on the DB's clock so the "today" used for the log dates
        # matches the one `get_habit_stats` compares against (breaks at
        # UTC vs local wall-clock boundaries otherwise).
        today = tmp_db._now()
        # Log 3 consecutive days
        for i in range(3):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            tmp_db.log_habit("exercise", done=True, date=d)
        stats = tmp_db.get_habit_stats()
        assert len(stats) == 1
        assert stats[0]["name"] == "exercise"
        assert stats[0]["streak"] == 3

    def test_streak_breaks_on_miss(self, tmp_db: Database):
        today = tmp_db._now()
        tmp_db.log_habit("reading", done=True, date=today.strftime("%Y-%m-%d"))
        # Skip yesterday
        tmp_db.log_habit("reading", done=False,
                         date=(today - timedelta(days=1)).strftime("%Y-%m-%d"))
        tmp_db.log_habit("reading", done=True,
                         date=(today - timedelta(days=2)).strftime("%Y-%m-%d"))
        stats = tmp_db.get_habit_stats()
        assert stats[0]["streak"] == 1  # Only today counts

    def test_week_rate(self, tmp_db: Database):
        today = tmp_db._now()
        for i in range(4):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            tmp_db.log_habit("meditation", done=True, date=d)
        stats = tmp_db.get_habit_stats()
        assert stats[0]["week_done"] == 4
        assert stats[0]["week_rate"] == 57  # 4/7 * 100 rounded


class TestRecurringReminders:
    def test_non_recurring_stays_sent(self, tmp_db: Database):
        r = tmp_db.create_reminder("Once", "2026-04-15 10:00:00")
        tmp_db.mark_reminder_sent(r["id"])
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 0

    def test_daily_recurrence_creates_next(self, tmp_db: Database):
        r = tmp_db.create_reminder("Daily task", "2026-04-15 09:00:00",
                                   recurrence="daily")
        tmp_db.mark_reminder_sent(r["id"])
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        assert pending[0]["text"] == "Daily task"
        assert pending[0]["recurrence"] == "daily"
        assert "2026-04-16" in pending[0]["trigger_at"]

    def test_weekly_recurrence_creates_next(self, tmp_db: Database):
        r = tmp_db.create_reminder("Weekly review", "2026-04-15 18:00:00",
                                   recurrence="weekly")
        tmp_db.mark_reminder_sent(r["id"])
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        assert "2026-04-22" in pending[0]["trigger_at"]

    def test_monthly_recurrence_creates_next(self, tmp_db: Database):
        r = tmp_db.create_reminder("Monthly report", "2026-04-15 10:00:00",
                                   recurrence="monthly")
        tmp_db.mark_reminder_sent(r["id"])
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        assert "2026-05-15" in pending[0]["trigger_at"]
