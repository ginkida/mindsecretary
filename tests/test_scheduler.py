"""Tests for proactive/scheduler.py — quiet hours and notification logic."""
from __future__ import annotations

from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_scheduler(quiet_hours: list[str], notification_limit: int = 10):
    """Build a ProactiveScheduler with minimal mocks."""
    from mindsecretary.proactive.scheduler import ProactiveScheduler

    profile = MagicMock()
    profile.quiet_hours = quiet_hours
    profile.notification_limit = notification_limit
    profile.wake_up = "07:00"
    profile.timezone = "Europe/Moscow"

    settings = MagicMock()
    settings.morning_briefing = False
    settings.evening_summary = False
    settings.smart_questions = False
    settings.decision_followups = False
    settings.weekly_review = False
    settings.weather_monitor = False
    settings.birthday_alerts = False
    settings.reminder_check_minutes = 5
    settings.weather_check_minutes = 60
    settings.quiet_contact_days = 30
    settings.quiet_contact_min_mentions = 3

    db = MagicMock()
    send_fn = MagicMock()

    return ProactiveScheduler(
        db=db, profile=profile, settings=settings,
        send_fn=send_fn, weather=None,
    )


class TestParseQuietHours:
    def test_valid_same_day(self):
        s = _make_scheduler(["12:00", "14:00"])
        start, end = s._parse_quiet_hours()
        assert start == time(12, 0)
        assert end == time(14, 0)

    def test_valid_wraps_midnight(self):
        s = _make_scheduler(["23:00", "07:00"])
        start, end = s._parse_quiet_hours()
        assert start == time(23, 0)
        assert end == time(7, 0)

    def test_empty_list(self):
        s = _make_scheduler([])
        assert s._parse_quiet_hours() == (None, None)

    def test_single_value(self):
        s = _make_scheduler(["12:00"])
        assert s._parse_quiet_hours() == (None, None)

    def test_invalid_format(self):
        s = _make_scheduler(["noon", "midnight"])
        assert s._parse_quiet_hours() == (None, None)


class TestInQuietHours:
    def _patch_now(self, hour, minute=0):
        """Patch tz_now in the scheduler module to return a fixed time."""
        from datetime import datetime
        fake_dt = datetime(2026, 4, 15, hour, minute, 0)
        return patch("mindsecretary.proactive.scheduler.tz_now", return_value=fake_dt)

    def test_same_day_inside(self):
        s = _make_scheduler(["12:00", "14:00"])
        with self._patch_now(13):
            assert s._in_quiet_hours() is True

    def test_same_day_outside(self):
        s = _make_scheduler(["12:00", "14:00"])
        with self._patch_now(11):
            assert s._in_quiet_hours() is False

    def test_midnight_wrap_late_night(self):
        s = _make_scheduler(["23:00", "07:00"])
        with self._patch_now(23, 30):
            assert s._in_quiet_hours() is True

    def test_midnight_wrap_early_morning(self):
        s = _make_scheduler(["23:00", "07:00"])
        with self._patch_now(5):
            assert s._in_quiet_hours() is True

    def test_midnight_wrap_daytime(self):
        s = _make_scheduler(["23:00", "07:00"])
        with self._patch_now(12):
            assert s._in_quiet_hours() is False

    def test_equal_start_end_always_off(self):
        s = _make_scheduler(["08:00", "08:00"])
        assert s._in_quiet_hours() is False

    def test_no_quiet_hours_always_off(self):
        s = _make_scheduler([])
        assert s._in_quiet_hours() is False


class TestActionNudge:
    def test_builds_nudge_from_open_loops(self):
        s = _make_scheduler(["23:00", "07:00"])
        s.db.get_open_loops.return_value = {
            "counts": {
                "overdue_reminders": 1,
                "due_today_reminders": 1,
                "upcoming_events": 1,
                "pending_goals": 2,
                "due_decisions": 1,
            },
            "overdue_reminders": [{"text": "Call Mom", "trigger_at": "2026-04-15 09:00:00"}],
            "due_today_reminders": [{"text": "Pay bill", "trigger_at": "2026-04-15 13:30:00"}],
            "upcoming_events": [{"title": "Standup", "start_at": "2026-04-15 14:00:00"}],
            "pending_goals": [{"title": "Write report", "priority": "high"}],
            "due_decisions": [{"description": "Choose hosting provider"}],
        }
        with patch("mindsecretary.proactive.scheduler.check_contact_frequency", return_value=[]), \
             patch("mindsecretary.proactive.scheduler.tz_now", return_value=datetime(2026, 4, 15, 12, 30, 0)):
            text = s._build_action_nudge()
        assert text is not None
        assert "На контроле" in text
        assert "Call Mom" in text

    def test_ignores_nonurgent_open_items(self):
        s = _make_scheduler(["23:00", "07:00"])
        s.db.get_open_loops.return_value = {
            "counts": {
                "overdue_reminders": 0,
                "due_today_reminders": 0,
                "upcoming_events": 1,
                "pending_goals": 1,
                "due_decisions": 0,
            },
            "overdue_reminders": [],
            "due_today_reminders": [],
            "upcoming_events": [{"title": "Tomorrow event", "start_at": "2026-04-16 18:00:00"}],
            "pending_goals": [{"title": "Normal goal", "priority": "medium"}],
            "due_decisions": [],
        }
        with patch("mindsecretary.proactive.scheduler.check_contact_frequency", return_value=[]), \
             patch("mindsecretary.proactive.scheduler.tz_now", return_value=datetime(2026, 4, 15, 12, 30, 0)):
            text = s._build_action_nudge()
        assert text is None

    @pytest.mark.asyncio
    async def test_smart_question_used_when_nudge_on_cooldown(self):
        s = _make_scheduler(["23:00", "07:00"])
        s.smart_questions = MagicMock()
        s.smart_questions.generate_question = AsyncMock(return_value="🤔 Как дела с проектом?")
        s._send_proactive = AsyncMock(return_value=True)

        with patch.object(s, "_get_last_nudge", return_value=datetime.now()), \
             patch.object(s, "_build_action_nudge", return_value="⚠️ На контроле"):
            await s._smart_question()

        s.smart_questions.generate_question.assert_awaited_once()
        s._send_proactive.assert_awaited_once_with(
            "🤔 Как дела с проектом?", kind="smart_question",
        )
