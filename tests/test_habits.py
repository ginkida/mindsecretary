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
        # Future trigger so the catch-up loop (added v0.13.15) doesn't roll
        # past the +1d boundary — this test covers the basic on-time path.
        r = tmp_db.create_reminder("Daily task", "2099-04-15 09:00:00",
                                   recurrence="daily")
        tmp_db.mark_reminder_sent(r["id"])
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        assert pending[0]["text"] == "Daily task"
        assert pending[0]["recurrence"] == "daily"
        assert "2099-04-16" in pending[0]["trigger_at"]

    def test_weekly_recurrence_creates_next(self, tmp_db: Database):
        r = tmp_db.create_reminder("Weekly review", "2099-04-15 18:00:00",
                                   recurrence="weekly")
        tmp_db.mark_reminder_sent(r["id"])
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        assert "2099-04-22" in pending[0]["trigger_at"]

    def test_monthly_recurrence_creates_next(self, tmp_db: Database):
        r = tmp_db.create_reminder("Monthly report", "2099-04-15 10:00:00",
                                   recurrence="monthly")
        tmp_db.mark_reminder_sent(r["id"])
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        assert "2099-05-15" in pending[0]["trigger_at"]

    def test_recurring_catchup_rolls_past_now(self, tmp_db: Database):
        """If bot was offline for multiple periods, a daily reminder set 5
        days ago should NOT fire 5 times back-to-back when bot returns.
        mark_reminder_sent rolls next_trigger forward past now in one
        UPDATE — pre-fix it advanced by exactly +1 day, leaving next still
        in the past, retriggering on the next 5-min check, repeat. Spam."""
        from datetime import timedelta as _td
        five_days_ago = (tmp_db.local_now_naive() - _td(days=5)).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        r = tmp_db.create_reminder("daily water", five_days_ago,
                                   recurrence="daily")

        tmp_db.mark_reminder_sent(r["id"])

        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        # Next must be in the FUTURE — that's the point of the catch-up
        from datetime import datetime as _dt
        next_trigger = _dt.fromisoformat(
            pending[0]["trigger_at"].replace(" ", "T"),
        )
        assert next_trigger > tmp_db.local_now_naive(), (
            "next recurring trigger landed in the past — bot would fire it "
            "again on the next 5-min check, leading to spam after downtime"
        )

    def test_recurring_catchup_picks_first_future_occurrence(self, tmp_db: Database):
        """Roll forward should land on the FIRST future occurrence, not jump
        ahead arbitrarily. Daily reminder from 5 days ago caught up at
        12:00 today should land tomorrow at the same hour, not later."""
        from datetime import timedelta as _td
        now_local = tmp_db.local_now_naive()
        # Anchor the test trigger at noon-of-(now-5days) to make the
        # expected next-occurrence math deterministic.
        anchor = now_local.replace(hour=12, minute=0, second=0, microsecond=0)
        five_days_ago_at_noon = anchor - _td(days=5)

        r = tmp_db.create_reminder(
            "daily noon thing",
            five_days_ago_at_noon.strftime("%Y-%m-%d %H:%M:%S"),
            recurrence="daily",
        )
        tmp_db.mark_reminder_sent(r["id"])

        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        from datetime import datetime as _dt
        next_trigger = _dt.fromisoformat(
            pending[0]["trigger_at"].replace(" ", "T"),
        )
        # Next must be in the future, AND within the next 24 hours (the
        # daily delta) — anything farther means we over-rolled.
        assert next_trigger > now_local
        assert next_trigger - now_local <= _td(days=1)
