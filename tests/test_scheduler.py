"""Tests for proactive/scheduler.py — quiet hours and notification logic."""
from __future__ import annotations

from datetime import time
from unittest.mock import MagicMock, patch

import pytest


def _make_scheduler(quiet_hours: list[str], notification_limit: int = 10):
    """Build a ProactiveScheduler with minimal mocks."""
    from mindsecretary.proactive.scheduler import ProactiveScheduler

    profile = MagicMock()
    profile.quiet_hours = quiet_hours
    profile.notification_limit = notification_limit
    profile.wake_up = "07:00"

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
    def test_same_day_inside(self):
        s = _make_scheduler(["12:00", "14:00"])
        with patch("mindsecretary.proactive.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = time(13, 0)
            assert s._in_quiet_hours() is True

    def test_same_day_outside(self):
        s = _make_scheduler(["12:00", "14:00"])
        with patch("mindsecretary.proactive.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = time(11, 0)
            assert s._in_quiet_hours() is False

    def test_midnight_wrap_late_night(self):
        s = _make_scheduler(["23:00", "07:00"])
        with patch("mindsecretary.proactive.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = time(23, 30)
            assert s._in_quiet_hours() is True

    def test_midnight_wrap_early_morning(self):
        s = _make_scheduler(["23:00", "07:00"])
        with patch("mindsecretary.proactive.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = time(5, 0)
            assert s._in_quiet_hours() is True

    def test_midnight_wrap_daytime(self):
        s = _make_scheduler(["23:00", "07:00"])
        with patch("mindsecretary.proactive.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = time(12, 0)
            assert s._in_quiet_hours() is False

    def test_equal_start_end_always_off(self):
        s = _make_scheduler(["08:00", "08:00"])
        assert s._in_quiet_hours() is False

    def test_no_quiet_hours_always_off(self):
        s = _make_scheduler([])
        assert s._in_quiet_hours() is False
