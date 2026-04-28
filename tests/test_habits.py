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

    def test_streak_breaks_on_unlogged_gap(self, tmp_db: Database):
        """A missing day (no row at all) must break the streak — earlier
        the loop just iterated rows, so logs on day 0 / -2 / -3 (gap on
        -1) showed streak=3 instead of 1. Conservative model: missing
        day = unknown = break."""
        today = tmp_db._now()
        for offset in (0, 2, 3):  # log day 0, skip day 1, log days 2 + 3
            d = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            tmp_db.log_habit("yoga", done=True, date=d)
        stats = tmp_db.get_habit_stats()
        assert stats[0]["streak"] == 1  # only today counts; gap on day -1

    def test_streak_zero_when_today_not_logged(self, tmp_db: Database):
        """Streak is zero if today wasn't logged, even with prior done logs.
        Catches the off-by-one shape where 'most recent done log' != 'today'."""
        today = tmp_db._now()
        # Log yesterday and 2-days-ago, but NOT today
        for offset in (1, 2):
            d = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            tmp_db.log_habit("walk", done=True, date=d)
        stats = tmp_db.get_habit_stats()
        assert stats[0]["streak"] == 0
        assert stats[0]["logged_today"] is False

    def test_logged_today_flag(self, tmp_db: Database):
        today = tmp_db._now().strftime("%Y-%m-%d")
        tmp_db.log_habit("water", done=True, date=today)
        stats = tmp_db.get_habit_stats()
        assert stats[0]["logged_today"] is True

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
