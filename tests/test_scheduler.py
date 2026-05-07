"""Tests for proactive/scheduler.py — quiet hours and notification logic."""
from __future__ import annotations

from datetime import datetime, time, timedelta
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


class TestConversationAwareDefer:
    """`_send_proactive` skips firing when the user has sent a message
    in the last N minutes — prevents scheduled jobs from interrupting
    an active conversation. Reminders (separate code path via
    monitor.check_reminders → send_fn) intentionally bypass."""

    @pytest.mark.asyncio
    async def test_skips_when_user_recently_active(self):
        s = _make_scheduler(["23:00", "07:00"])
        # Pretend the user just typed something
        s.db.has_recent_user_messages = MagicMock(return_value=True)
        s.db.is_snoozed_now = MagicMock(return_value=False)
        s.send_fn = AsyncMock()

        result = await s._send_proactive("☀️ Доброе утро", kind="morning_briefing")

        assert result is False
        # Critical: send_fn must NOT have been called — message dropped
        s.send_fn.assert_not_called()
        # And no notification logged (would inflate the daily count)
        s.db.log_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_proceeds_when_user_quiet(self):
        s = _make_scheduler(["23:00", "07:00"])
        s.db.has_recent_user_messages = MagicMock(return_value=False)
        s.db.is_snoozed_now = MagicMock(return_value=False)
        s.db.count_notifications_today = MagicMock(return_value=0)
        s.profile.notification_limit = 10
        s.send_fn = AsyncMock()

        # Pin the scheduler clock to 12:00 — well outside the
        # "23:00-07:00" quiet hours window. Without this the test was
        # flaky between 23:00-07:00 wall-clock since _in_quiet_hours()
        # uses live tz_now.
        fake_dt = datetime(2026, 4, 15, 12, 0, 0)
        with patch("mindsecretary.proactive.scheduler.tz_now",
                   return_value=fake_dt):
            result = await s._send_proactive(
                "☀️ Доброе утро", kind="morning_briefing",
            )

        assert result is True
        s.send_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recency_check_db_error_does_not_block_send(self):
        """If has_recent_user_messages raises (DB transient issue), we
        log and proceed with the send — missing a briefing over a
        broken query is worse than risking an interrupt."""
        s = _make_scheduler(["23:00", "07:00"])
        s.db.has_recent_user_messages = MagicMock(
            side_effect=RuntimeError("db locked"),
        )
        s.db.is_snoozed_now = MagicMock(return_value=False)
        s.db.count_notifications_today = MagicMock(return_value=0)
        s.profile.notification_limit = 10
        s.send_fn = AsyncMock()

        fake_dt = datetime(2026, 4, 15, 12, 0, 0)
        with patch("mindsecretary.proactive.scheduler.tz_now",
                   return_value=fake_dt):
            result = await s._send_proactive("text", kind="evening_summary")

        # Send still happened despite the recency check failing
        assert result is True
        s.send_fn.assert_awaited_once()


class TestSnoozeGate:
    """`/snooze 2h` writes to preferences; scheduler reads via
    is_snoozed_now and gates _send_proactive. Reminders bypass through
    a separate code path (monitor.check_reminders → send_fn)."""

    @pytest.mark.asyncio
    async def test_send_proactive_blocked_when_snoozed(self):
        s = _make_scheduler(["23:00", "07:00"])
        s.db.is_snoozed_now = MagicMock(return_value=True)
        s.send_fn = AsyncMock()

        result = await s._send_proactive("☀️ Доброе утро", kind="morning_briefing")

        assert result is False
        s.send_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_snooze_check_failure_doesnt_block_send(self):
        """A DB error on is_snoozed_now must NOT silently kill briefings —
        log and proceed. Same defensive policy as the recency check."""
        s = _make_scheduler(["23:00", "07:00"])
        s.db.is_snoozed_now = MagicMock(side_effect=RuntimeError("db locked"))
        s.db.has_recent_user_messages = MagicMock(return_value=False)
        s.db.count_notifications_today = MagicMock(return_value=0)
        s.profile.notification_limit = 10
        s.send_fn = AsyncMock()

        fake_dt = datetime(2026, 4, 15, 12, 0, 0)
        with patch("mindsecretary.proactive.scheduler.tz_now",
                   return_value=fake_dt):
            result = await s._send_proactive("text", kind="evening_summary")

        assert result is True
        s.send_fn.assert_awaited_once()


class TestSendProactiveLogFailure:
    """Pre-fix log_interaction raised inside the same try as send_fn, so
    a transient DB error AFTER successful Telegram send returned False —
    callers (birthday alerts, event reflections) treated it as "not sent"
    and re-fired the same notification on the next tick. Worse than a
    missing log line: actual user-visible duplicate alerts."""

    @pytest.mark.asyncio
    async def test_returns_true_when_send_succeeds_but_log_fails(self):
        s = _make_scheduler([])
        s.db.is_snoozed_now = MagicMock(return_value=False)
        s.db.has_recent_user_messages = MagicMock(return_value=False)
        s.db.count_notifications_today = MagicMock(return_value=0)
        s.db.log_interaction = MagicMock(side_effect=RuntimeError("disk full"))
        s.send_fn = AsyncMock()

        result = await s._send_proactive("text", kind="birthday_alert")

        assert result is True  # message went out, log loss is acceptable
        s.send_fn.assert_awaited_once()
        s.db.log_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_send_itself_fails(self):
        """Telegram failure → False → caller retries next tick. Log
        attempt must NOT happen because there's nothing to log."""
        s = _make_scheduler([])
        s.db.is_snoozed_now = MagicMock(return_value=False)
        s.db.has_recent_user_messages = MagicMock(return_value=False)
        s.db.count_notifications_today = MagicMock(return_value=0)
        s.db.log_interaction = MagicMock()
        s.send_fn = AsyncMock(side_effect=RuntimeError("telegram down"))

        result = await s._send_proactive("text", kind="event_reflection")

        assert result is False
        s.send_fn.assert_awaited_once()
        s.db.log_interaction.assert_not_called()


class TestActionNudgeReminderPlural:
    """The 'Просрочено N напоминаний' line is the most-prominent count
    in the action nudge — it reaches the user directly. Pre-fix the
    hardcoded plural form read wrong for any count != 5+ ("Просрочено 1
    напоминаний", "Просрочено 2 напоминаний"). Pluralize so the form
    matches the count."""

    def test_singular_form_for_one(self):
        s = _make_scheduler([])
        s.db.get_open_loops = MagicMock(return_value={
            "counts": {"overdue_reminders": 1},
            "overdue_reminders": [{"text": "позвонить", "trigger_at": "2099-01-01 10:00:00"}],
        })
        # Stub other contributors so the test focuses on the target line
        s.db.get_pending_decision_followups = MagicMock(return_value=[])
        with patch("mindsecretary.proactive.scheduler.check_contact_frequency",
                   return_value=[]):
            nudge = s._build_action_nudge()
        assert "Просрочено 1 напоминание" in nudge
        assert "Просрочено 1 напоминаний" not in nudge

    def test_few_form_for_two(self):
        s = _make_scheduler([])
        s.db.get_open_loops = MagicMock(return_value={
            "counts": {"overdue_reminders": 2},
            "overdue_reminders": [{"text": "позвонить", "trigger_at": "2099-01-01 10:00:00"}],
        })
        s.db.get_pending_decision_followups = MagicMock(return_value=[])
        with patch("mindsecretary.proactive.scheduler.check_contact_frequency",
                   return_value=[]):
            nudge = s._build_action_nudge()
        assert "Просрочено 2 напоминания" in nudge

    def test_many_form_for_five(self):
        s = _make_scheduler([])
        s.db.get_open_loops = MagicMock(return_value={
            "counts": {"overdue_reminders": 5},
            "overdue_reminders": [{"text": "позвонить", "trigger_at": "2099-01-01 10:00:00"}],
        })
        s.db.get_pending_decision_followups = MagicMock(return_value=[])
        with patch("mindsecretary.proactive.scheduler.check_contact_frequency",
                   return_value=[]):
            nudge = s._build_action_nudge()
        assert "Просрочено 5 напоминаний" in nudge


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

    def test_quiet_contact_failure_logs_and_continues(self, caplog):
        """Pre-fix: quiet-contact check inside _build_action_nudge swallowed
        all exceptions silently. Now ops sees a warning and the rest of
        the nudge still renders."""
        import logging as _log
        s = _make_scheduler(["23:00", "07:00"])
        s.db.get_open_loops.return_value = {
            "counts": {
                "overdue_reminders": 1, "due_today_reminders": 0,
                "upcoming_events": 0, "pending_goals": 0, "due_decisions": 0,
            },
            "overdue_reminders": [
                {"text": "Call Mom", "trigger_at": "2026-04-15 09:00:00"},
            ],
            "due_today_reminders": [], "upcoming_events": [],
            "pending_goals": [], "due_decisions": [],
        }
        with patch(
            "mindsecretary.proactive.scheduler.check_contact_frequency",
            side_effect=RuntimeError("DB schema drift"),
        ), patch(
            "mindsecretary.proactive.scheduler.tz_now",
            return_value=datetime(2026, 4, 15, 12, 30, 0),
        ), caplog.at_level(_log.WARNING):
            text = s._build_action_nudge()

        # Nudge content still rendered — overdue reminder isn't lost
        assert text is not None
        assert "Call Mom" in text
        # Failure surfaced in logs (no longer silent)
        assert any(
            "Quiet-contact check" in record.message
            for record in caplog.records
        )

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


class TestSchedulerTimezone:
    """Scheduler must use profile.timezone for cron jobs, not system TZ.

    Old behavior: AsyncIOScheduler() defaults to system TZ (UTC in slim
    containers), so cron hour=7 fired at 07:00 UTC = 12:00 Asia/Almaty.
    Fix: pass profile.timezone to AsyncIOScheduler constructor.
    """

    def test_scheduler_uses_profile_timezone(self):
        s = _make_scheduler(["23:00", "07:00"])
        # Scheduler's timezone should match profile
        tz = s.scheduler.timezone
        # tz could be ZoneInfo or pytz — accept any string-matchable form
        assert "Moscow" in str(tz) or "Europe" in str(tz)

    def test_invalid_timezone_falls_back_without_crash(self):
        from unittest.mock import MagicMock
        from mindsecretary.proactive.scheduler import ProactiveScheduler

        profile = MagicMock()
        profile.quiet_hours = []
        profile.notification_limit = 10
        profile.wake_up = "07:00"
        profile.timezone = "Nonsense/Invalid"

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

        # Should NOT raise — invalid TZ falls back to system
        s = ProactiveScheduler(
            db=MagicMock(), profile=profile, settings=settings,
            send_fn=MagicMock(), weather=None,
        )
        assert s.scheduler is not None


class TestCronJobMisfireGrace:
    """All cron-scheduled jobs must have misfire_grace_time + coalesce
    set so a delayed startup (bot restart at 09:01 for a 09:00 cron)
    still fires the missed daily job. APScheduler's default 1-second
    grace silently dropped briefings / birthday checks / cleanups
    after even a brief startup delay."""

    def _start_with_all_cron(self):
        from unittest.mock import patch
        from mindsecretary.proactive.scheduler import ProactiveScheduler

        profile = MagicMock()
        profile.quiet_hours = []
        profile.notification_limit = 10
        profile.wake_up = "07:00"
        profile.timezone = "UTC"

        settings = MagicMock()
        # Enable every cron job we care about.
        settings.morning_briefing = True
        settings.evening_summary = True
        settings.smart_questions = True
        settings.decision_followups = True
        settings.weekly_review = True
        settings.weather_monitor = False
        settings.birthday_alerts = True
        settings.event_alerts = False
        settings.event_reflections = False
        settings.reminder_check_minutes = 5
        settings.weather_check_minutes = 60
        settings.quiet_contact_days = 30
        settings.quiet_contact_min_mentions = 3
        settings.event_alert_lead_minutes = 0
        settings.event_reflection_lag_minutes = 0

        s = ProactiveScheduler(
            db=MagicMock(), profile=profile, settings=settings,
            send_fn=MagicMock(), weather=None,
        )
        # Don't actually start the loop — just register jobs, assert,
        # then shutdown.
        with patch.object(s.scheduler, "start"):
            s.start()
        return s

    def test_all_cron_jobs_have_misfire_grace(self):
        s = self._start_with_all_cron()
        try:
            cron_ids = {
                "birthday_check", "morning_prompt", "smart_question",
                "decision_followup", "evening_prompt", "weekly_review",
                "daily_backup", "cleanup_old_data",
            }
            seen = set()
            for job in s.scheduler.get_jobs():
                if job.id in cron_ids:
                    seen.add(job.id)
                    assert job.misfire_grace_time == 3600, (
                        f"{job.id} missing misfire_grace_time"
                    )
                    assert job.coalesce is True, (
                        f"{job.id} should have coalesce=True"
                    )
            missing = cron_ids - seen
            assert not missing, f"cron jobs not registered: {missing}"
        finally:
            if s.scheduler.running:
                s.scheduler.shutdown(wait=False)

    def test_interval_jobs_keep_default_no_grace(self):
        """Interval jobs (reminder_check) fire often enough that a
        misfire is not catastrophic — keep the default behavior so a
        weekend offline doesn't unleash a stampede on restart.

        APScheduler doesn't set the `misfire_grace_time` attribute on
        a Job when none is passed, so confirm the attribute is absent
        (or, if APScheduler internals change to always set it, that
        the value differs from our 3600s cron policy)."""
        s = self._start_with_all_cron()
        try:
            interval_ids = {"reminder_check"}
            for job in s.scheduler.get_jobs():
                if job.id in interval_ids:
                    grace = getattr(job, "misfire_grace_time", None)
                    assert grace != 3600
        finally:
            if s.scheduler.running:
                s.scheduler.shutdown(wait=False)


class TestFormatRainAlert:
    """Rendering of the rain alert message — smart grouping + lead time."""

    def test_single_hour_imminent(self):
        from mindsecretary.proactive.scheduler import _format_rain_alert
        text = _format_rain_alert([(14, 80, 63)], now_hour=13)
        assert "через час" in text
        assert "в 14:00" in text
        assert "до 80%" in text
        assert "🌧" in text

    def test_range_merged(self):
        from mindsecretary.proactive.scheduler import _format_rain_alert
        fresh = [(13, 60, 63), (14, 80, 63), (15, 70, 63)]
        text = _format_rain_alert(fresh, now_hour=12)
        assert "с 13:00 до 16:00" in text
        assert "до 80%" in text

    def test_thunderstorm_uses_storm_emoji(self):
        from mindsecretary.proactive.scheduler import _format_rain_alert
        text = _format_rain_alert([(15, 90, 95)], now_hour=13)
        assert "⛈" in text
        assert "Гроза" in text

    def test_multiple_non_consecutive_windows(self):
        from mindsecretary.proactive.scheduler import _format_rain_alert
        fresh = [(14, 60, 63), (18, 80, 63)]
        text = _format_rain_alert(fresh, now_hour=12)
        assert "в 14:00" in text
        assert "в 18:00" in text

    def test_in_progress_lead(self):
        from mindsecretary.proactive.scheduler import _format_rain_alert
        text = _format_rain_alert([(14, 70, 63)], now_hour=14)
        assert "начинается" in text

    def test_far_future_no_lead(self):
        from mindsecretary.proactive.scheduler import _format_rain_alert
        text = _format_rain_alert([(20, 70, 63)], now_hour=10)
        # lead only added up to +6h; 10h out is too far for a countdown
        assert "через" not in text


class TestWeatherCheck:
    """End-to-end behaviour of _check_weather against mocked fixture."""

    def _make_with_weather(self, tz: str = "Asia/Almaty"):
        from mindsecretary.proactive.scheduler import ProactiveScheduler

        profile = MagicMock()
        profile.quiet_hours = []
        profile.notification_limit = 10
        profile.wake_up = "07:00"
        profile.timezone = tz

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
        db.get_preference.return_value = None  # no prior alert state

        weather = MagicMock()
        weather.get_forecast = AsyncMock()

        send_fn = AsyncMock()

        s = ProactiveScheduler(
            db=db, profile=profile, settings=settings,
            send_fn=send_fn, weather=weather,
        )
        s._send_proactive = AsyncMock(return_value=True)
        return s

    @pytest.mark.asyncio
    async def test_skips_past_hours_defensively(self):
        """Even if weather.py returned a past hour, scheduler must filter it."""
        from datetime import datetime as real_dt

        s = self._make_with_weather()
        s.weather.get_forecast.return_value = {
            "rain_today": [(13, 80, 63), (14, 70, 63)],  # all past at 18:56
        }
        fake_now = real_dt(2026, 4, 24, 18, 56)
        with patch("mindsecretary.proactive.scheduler.tz_now", return_value=fake_now):
            await s._check_weather()
        s._send_proactive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_alerts_on_future_rain(self):
        from datetime import datetime as real_dt

        s = self._make_with_weather()
        s.weather.get_forecast.return_value = {
            "rain_today": [(20, 80, 63), (21, 90, 63)],
        }
        fake_now = real_dt(2026, 4, 24, 18, 56)
        with patch("mindsecretary.proactive.scheduler.tz_now", return_value=fake_now):
            await s._check_weather()
        s._send_proactive.assert_awaited_once()
        text = s._send_proactive.await_args.args[0]
        assert "с 20:00 до 22:00" in text
        assert "до 90%" in text

    @pytest.mark.asyncio
    async def test_dedup_via_preference(self):
        """Hours already alerted today must not be re-sent after 'restart'."""
        import json
        from datetime import datetime as real_dt

        s = self._make_with_weather()
        s.db.get_preference.return_value = {
            "value": json.dumps({"date": "2026-04-24", "hours": [20, 21]}),
        }
        s.weather.get_forecast.return_value = {
            "rain_today": [(20, 80, 63), (21, 90, 63)],
        }
        fake_now = real_dt(2026, 4, 24, 18, 56)
        with patch("mindsecretary.proactive.scheduler.tz_now", return_value=fake_now):
            await s._check_weather()
        s._send_proactive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_alerts_only_new_hours(self):
        """Only hours beyond the stored set trigger a new message."""
        import json
        from datetime import datetime as real_dt

        s = self._make_with_weather()
        s.db.get_preference.return_value = {
            "value": json.dumps({"date": "2026-04-24", "hours": [20]}),
        }
        s.weather.get_forecast.return_value = {
            "rain_today": [(20, 80, 63), (21, 90, 63), (22, 70, 63)],
        }
        fake_now = real_dt(2026, 4, 24, 18, 56)
        with patch("mindsecretary.proactive.scheduler.tz_now", return_value=fake_now):
            await s._check_weather()
        s._send_proactive.assert_awaited_once()
        text = s._send_proactive.await_args.args[0]
        assert "с 21:00 до 23:00" in text  # 21-22 merged, 20 skipped
        # Merged set of alerted hours is persisted
        call = s.db.set_preference.call_args
        stored = json.loads(call.args[1])
        assert sorted(stored["hours"]) == [20, 21, 22]
        assert stored["date"] == "2026-04-24"

    @pytest.mark.asyncio
    async def test_new_day_resets_dedup(self):
        """Yesterday's alerted hours do not suppress today's alerts."""
        import json
        from datetime import datetime as real_dt

        s = self._make_with_weather()
        s.db.get_preference.return_value = {
            "value": json.dumps({"date": "2026-04-23", "hours": [20, 21]}),
        }
        s.weather.get_forecast.return_value = {
            "rain_today": [(20, 80, 63)],
        }
        fake_now = real_dt(2026, 4, 24, 18, 56)
        with patch("mindsecretary.proactive.scheduler.tz_now", return_value=fake_now):
            await s._check_weather()
        s._send_proactive.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_forecast_error_silent(self):
        from datetime import datetime as real_dt

        s = self._make_with_weather()
        s.weather.get_forecast.return_value = {"error": "timeout"}
        fake_now = real_dt(2026, 4, 24, 18, 0)
        with patch("mindsecretary.proactive.scheduler.tz_now", return_value=fake_now):
            await s._check_weather()
        s._send_proactive.assert_not_awaited()
        s.db.set_preference.assert_not_called()


class TestNudgeCooldown:
    """Cooldown comparison must handle both legacy (naive UTC) and new
    (TZ-aware profile-local) preferences without drifting by the offset."""

    def _make(self):
        return _make_scheduler(["23:00", "07:00"])

    @pytest.mark.asyncio
    async def test_legacy_naive_pref_honored(self):
        """Pref written pre-v0.12.2 is TZ-naive system UTC. Cooldown math
        must treat it as UTC, not as local wall-clock, otherwise an upgrade
        would re-fire the nudge prematurely on users with offset > 0."""
        from datetime import datetime as real_dt
        from datetime import timezone as real_tz

        s = self._make()
        s.smart_questions = MagicMock()
        s._send_proactive = AsyncMock(return_value=True)
        s._build_action_nudge = MagicMock(return_value="⚠️ На контроле")
        # Legacy pref: naive UTC, 2 hours ago (< 44h cooldown — should suppress)
        utc_two_hours_ago = real_dt.now(real_tz.utc).replace(tzinfo=None) - timedelta(hours=2)
        with patch.object(s, "_get_last_nudge", return_value=utc_two_hours_ago):
            # smart_question should be reached (nudge on cooldown)
            s.smart_questions.generate_question = AsyncMock(return_value="q")
            await s._smart_question()
        s._build_action_nudge.assert_not_called()

    @pytest.mark.asyncio
    async def test_aware_profile_pref_honored(self):
        """New aware pref in profile TZ must convert to UTC for the elapsed-
        time calculation — otherwise comparison against a naive `now` either
        drifts or TypeErrors out."""
        from datetime import datetime as real_dt
        from zoneinfo import ZoneInfo

        s = self._make()
        s.smart_questions = MagicMock()
        s._send_proactive = AsyncMock(return_value=True)
        s._build_action_nudge = MagicMock(return_value="⚠️ На контроле")
        # New-format pref: aware, profile TZ, 2h ago (should suppress)
        tz = ZoneInfo("Europe/Moscow")
        aware_two_hours_ago = real_dt.now(tz) - timedelta(hours=2)
        with patch.object(s, "_get_last_nudge", return_value=aware_two_hours_ago):
            s.smart_questions.generate_question = AsyncMock(return_value="q")
            await s._smart_question()
        s._build_action_nudge.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_cooldown_allows_nudge(self):
        """Once 44h have passed the nudge fires, regardless of pref format."""
        from datetime import datetime as real_dt
        from datetime import timezone as real_tz

        s = self._make()
        s.smart_questions = MagicMock()
        s._send_proactive = AsyncMock(return_value=True)
        s._build_action_nudge = MagicMock(return_value="⚠️ На контроле")
        stale = real_dt.now(real_tz.utc).replace(tzinfo=None) - timedelta(hours=48)
        with patch.object(s, "_get_last_nudge", return_value=stale), \
             patch.object(s, "_set_last_nudge"):
            await s._smart_question()
        s._build_action_nudge.assert_called_once()
        s._send_proactive.assert_awaited_once_with(
            "⚠️ На контроле", kind="open_loops_nudge",
        )


class TestNudgeCooldownGatedOnSend:
    """Pre-fix _set_last_nudge() ran unconditionally after _send_proactive,
    locking the next 44h even when the message was suppressed by quiet
    hours / snooze / recent-activity defer. User mid-chat at 13:00 lost
    nudges for ~two days. Now cooldown is set ONLY when the send actually
    went out (return True from _send_proactive)."""

    def _make(self):
        return _make_scheduler(["23:00", "07:00"])

    @pytest.mark.asyncio
    async def test_cooldown_not_set_when_send_suppressed(self):
        from datetime import datetime as real_dt
        from datetime import timezone as real_tz

        s = self._make()
        s.smart_questions = MagicMock()
        # Suppression path: send returns False (quiet hours / snooze /
        # recent-activity defer all funnel through this).
        s._send_proactive = AsyncMock(return_value=False)
        s._build_action_nudge = MagicMock(return_value="⚠️ На контроле")
        stale = real_dt.now(real_tz.utc).replace(tzinfo=None) - timedelta(hours=48)
        set_last = MagicMock()
        with patch.object(s, "_get_last_nudge", return_value=stale), \
             patch.object(s, "_set_last_nudge", set_last):
            await s._smart_question()
        s._send_proactive.assert_awaited_once()
        set_last.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_set_when_send_succeeded(self):
        """Mirror of the suppression test — confirm the happy path still
        stamps cooldown so the nudge doesn't re-fire every 13:00 forever."""
        from datetime import datetime as real_dt
        from datetime import timezone as real_tz

        s = self._make()
        s.smart_questions = MagicMock()
        s._send_proactive = AsyncMock(return_value=True)
        s._build_action_nudge = MagicMock(return_value="⚠️ На контроле")
        stale = real_dt.now(real_tz.utc).replace(tzinfo=None) - timedelta(hours=48)
        set_last = MagicMock()
        with patch.object(s, "_get_last_nudge", return_value=stale), \
             patch.object(s, "_set_last_nudge", set_last):
            await s._smart_question()
        s._send_proactive.assert_awaited_once()
        set_last.assert_called_once()


class TestBirthdayAlertFormat:
    """Birthday alert text used to dump the raw 'YYYY-MM-DD' birthday into
    the message — uninformative to the user and missing the age calc the
    DB already had data for. Reformatter computes days-until + age (when
    year is known) + the right Russian plural for the day count."""

    @staticmethod
    def _format(contact, year=2026, month=4, day=28):
        from mindsecretary.proactive.scheduler import ProactiveScheduler
        now = datetime(year, month, day, 9, 0, 0)
        today_md = now.strftime("%m-%d")
        return ProactiveScheduler._format_birthday_alert(contact, now, today_md)

    def test_today_with_year_shows_age(self):
        result = self._format({
            "name": "Маша", "relation": "друг", "birthday": "1990-04-28",
        })
        # Turning 36 in 2026 — age computed from next-occurrence year
        assert "🎂 Сегодня ДР: Маша (36) (друг)" == result

    def test_today_without_year_omits_age(self):
        """Year-less birthdays must NOT render parens — guessing an age
        is worse than skipping it."""
        result = self._format({
            "name": "Аня", "relation": "коллега", "birthday": "04-28",
        })
        assert "(0)" not in result
        assert "🎂 Сегодня ДР: Аня (коллега)" == result

    def test_upcoming_with_year_renders_days_and_age(self):
        result = self._format({
            "name": "Иван", "relation": "брат", "birthday": "1985-04-30",
        })
        # 2 days from 04-28 to 04-30 → "2 дня" (genitive plural)
        assert "📅 ДР через 2 дня:" in result
        assert "Иван (41) (брат)" in result
        assert "04-30" in result

    def test_upcoming_without_year_omits_age(self):
        result = self._format({"name": "Olga", "birthday": "04-29"})
        assert "(0)" not in result
        # 1 day = nominative singular "день"
        assert "📅 ДР через 1 день:" in result
        assert "Olga" in result
        assert "04-29" in result

    def test_eleven_days_uses_many_form_via_teens(self):
        """11-14 are the teens special case in Russian — they end in
        1/2/3/4 but use 'дней', not 'день/дня'. Pre-fix the inline rule
        emitted '11 день' which read very wrong."""
        result = self._format(
            {"name": "X", "birthday": "1990-05-09"},
            year=2026, month=4, day=28,
        )
        assert "📅 ДР через 11 дней:" in result

    def test_year_wrap_picks_next_year(self):
        """Birthday in January, today in December → days-until counts
        forward into next year, age uses next year's value."""
        result = self._format(
            {"name": "Боб", "birthday": "1980-01-05"},
            year=2026, month=12, day=30,
        )
        # 6 days from Dec 30 to Jan 5 next year
        assert "📅 ДР через 6 дней:" in result
        assert "Боб (47)" in result  # 47 = 2027 - 1980

    def test_empty_birthday_returns_empty_string(self):
        """Defensive: caller skips empty results — avoids '🎂 Сегодня ДР: !'"""
        assert self._format({"name": "X", "birthday": ""}) == ""
        assert self._format({"name": "X", "birthday": None}) == ""
        assert self._format({"name": "X"}) == ""

    def test_feb29_renders_today_in_non_leap_year(self):
        """Iter 27 made Feb 29 contacts surface in get_upcoming_birthdays
        on Feb 28 in non-leap years. The formatter must complete the
        carve-out by rendering "🎂 Сегодня ДР" — pre-fix it fell into
        the days_until path, ValueError'd on now.replace(day=29) for a
        non-leap year, and ended at the "Скоро ДР" fallback."""
        # Feb 28, 2026 — non-leap. Contact born Feb 29, 1996.
        result = self._format(
            {"name": "Leap", "birthday": "1996-02-29"},
            year=2026, month=2, day=28,
        )
        assert result.startswith("🎂 Сегодня ДР:")
        assert "Leap" in result
        # Age computed from next-occurrence year (2026 since Feb 29 is
        # treated as today via the carve-out)
        assert "(30)" in result
        # Must NOT fall through to the upcoming/skoro renderer
        assert "Скоро ДР" not in result

    def test_feb29_renders_today_normally_in_leap_year(self):
        """In leap years Feb 29 is a real day; the carve-out branch
        must not double-fire."""
        result = self._format(
            {"name": "Leap", "birthday": "1996-02-29"},
            year=2028, month=2, day=29,
        )
        assert result.startswith("🎂 Сегодня ДР:")
        assert "Leap" in result
        assert "(32)" in result

    def test_implausible_age_omitted(self):
        """A birth year of 1700 (typo, junk data) gives age > 150 — the
        guard drops the parens so we don't render '(326)' which would
        make the bot look broken."""
        result = self._format({
            "name": "X", "birthday": "1700-04-28",
        })
        assert "(326)" not in result
        assert "🎂 Сегодня ДР: X" == result

    def test_invalid_birthday_returns_empty(self):
        """Garbage in the birthday column should be SKIPPED, not rendered.
        Earlier the function fell back to '📅 Скоро ДР: X — rbage' (slice
        of the garbage), which leaks DB corruption into Telegram. The
        early MM-DD validation now drops such rows; /people still shows
        the contact, the bot just doesn't fire a birthday alert for them."""
        # Pure non-numeric garbage — 7 chars passes the length gate but
        # fails MM-DD parse.
        assert self._format({"name": "X", "birthday": "garbage"}) == ""
        # Out-of-range month — 13 doesn't exist
        assert self._format({"name": "X", "birthday": "2026-13-15"}) == ""
        # Out-of-range day — Feb has no 32
        assert self._format({"name": "X", "birthday": "2026-02-32"}) == ""
        # Missing dash separator
        assert self._format({"name": "X", "birthday": "20260415"}) == ""


class TestDecisionFollowupWording:
    """Followup proactive used 'Ты решил' for pending decisions — misleading,
    since by definition these decisions haven't been made yet (status=pending,
    follow_up_at past). 'Обдумывал' frames the reflection accurately."""

    @pytest.mark.asyncio
    async def test_followup_text_uses_pending_framing(self):
        from unittest.mock import AsyncMock, MagicMock
        s = _make_scheduler([])
        s.db.get_pending_decision_followups = MagicMock(return_value=[
            {"id": "abc", "description": "купить велосипед",
             "context": "бюджет 50к"},
        ])
        # Capture the proactive text that would be sent
        sent: list[str] = []
        async def fake_send(text, kind=None):
            sent.append(text)
            return True
        s._send_proactive = fake_send
        s.db.push_decision_followup = MagicMock()

        await s._check_decision_followups()

        assert len(sent) == 1
        body = sent[0]
        # New framing
        assert "Обдумывал" in body
        # Old misleading framing must NOT appear
        assert "Ты решил" not in body
        # Decision content + context surface
        assert "купить велосипед" in body
        assert "бюджет 50к" in body
