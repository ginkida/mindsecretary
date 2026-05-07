from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..core import pluralize_ru, tz_now
from ..core.config import Profile, Settings
from ..core.database import Database
from ..integrations.weather import WMO_CODES, WeatherClient, _merge_rain_hours
from ..learning.mood import check_contact_frequency
from .monitor import check_event_alerts, check_event_reflections, check_reminders

logger = logging.getLogger(__name__)

ACTION_NUDGE_COOLDOWN = timedelta(hours=44)
ACTION_NUDGE_EVENT_HORIZON = timedelta(hours=3)
ACTION_NUDGE_REMINDER_HORIZON = timedelta(hours=2)

WEATHER_ALERT_PREF_KEY = "weather_alerted_rain"
THUNDERSTORM_CODES = {95, 96, 99}
# WMO codes that represent actual rain / storm conditions. If the heaviest
# hour reports something else (e.g. clear + precipitation probability >= 50
# is an Open-Meteo data glitch), fall back to a generic "дождь" label
# rather than announcing "🌧 Ясно" at the user.
RAIN_WMO_CODES = {
    51, 53, 55, 56, 57,
    61, 63, 65, 66, 67,
    80, 81, 82,
    95, 96, 99,
}


class ProactiveScheduler:
    """Manages all proactive scheduled tasks."""

    def __init__(self, db: Database, weather: WeatherClient | None,
                 profile: Profile, settings: Settings, send_fn):
        self.db = db
        self.weather = weather
        self.profile = profile
        self.settings = settings
        self.send_fn = send_fn
        # Pass profile.timezone explicitly — APScheduler's default uses
        # system TZ (UTC in slim containers), which made cron hour=7 fire
        # at 07:00 UTC = 12:00 Asia/Almaty. Profile.timezone is the single
        # source of truth for all time rendering in the app.
        try:
            self.scheduler = AsyncIOScheduler(timezone=profile.timezone)
        except Exception as e:
            logger.error(
                "Invalid PROFILE_TIMEZONE '%s' (%s), falling back to system TZ",
                profile.timezone, type(e).__name__,
            )
            self.scheduler = AsyncIOScheduler()
        logger.info(
            "Scheduler TZ set to %s — cron jobs use this clock", profile.timezone,
        )
        # Set externally after creation
        self.briefing_generator = None
        self.weekly_reflection = None
        self.smart_questions = None

    # --- Quiet hours and notification limit ---

    def _parse_quiet_hours(self) -> tuple[time, time] | tuple[None, None]:
        try:
            parts = self.profile.quiet_hours
            if not parts or len(parts) != 2:
                return None, None
            start_h, start_m = map(int, parts[0].split(":"))
            end_h, end_m = map(int, parts[1].split(":"))
            return time(start_h, start_m), time(end_h, end_m)
        except (ValueError, TypeError, AttributeError):
            return None, None

    def _in_quiet_hours(self) -> bool:
        start, end = self._parse_quiet_hours()
        if start is None or end is None or start == end:
            return False
        now = tz_now(self.profile.timezone).time()
        if start < end:
            # Same-day window (e.g. 12:00-14:00)
            return start <= now < end
        # Wraps midnight (e.g. 23:00-07:00)
        return now >= start or now < end

    # Defer scheduled jobs when the user has been active in the last N
    # minutes — nothing kills flow like getting a morning briefing while
    # you're mid-typing. Reminders take a separate path
    # (monitor.check_reminders → send_fn) and intentionally bypass this.
    _CONVERSATION_DEFER_MINUTES = 5

    async def _send_proactive(self, text: str, kind: str = "notification") -> bool:
        """Send a proactive message respecting quiet hours, notification
        limit, and active-conversation deferral.

        Returns True if actually sent, False if dropped (quiet hours /
        limit / active conversation / error). Logs every sent message
        as message_type='notification' for counting.
        """
        if not text:
            return False
        if self._in_quiet_hours():
            logger.info("Skipping proactive (%s): quiet hours", kind)
            return False
        # User-initiated snooze (/snooze 2h) overrides everything except
        # reminders, which take a separate code path. Stored in
        # preferences so it survives restarts.
        try:
            if self.db.is_snoozed_now():
                logger.info("Skipping proactive (%s): user snooze active", kind)
                return False
        except Exception as e:
            logger.warning(
                "Snooze check failed (%s); proceeding",
                type(e).__name__,
            )
        # Best-effort recency check — DB error logs but doesn't block the
        # send, since the alternative is missing the briefing entirely
        # over a transient query failure.
        try:
            if self.db.has_recent_user_messages(self._CONVERSATION_DEFER_MINUTES):
                logger.info(
                    "Skipping proactive (%s): user active in last %dm",
                    kind, self._CONVERSATION_DEFER_MINUTES,
                )
                return False
        except Exception as e:
            logger.warning(
                "Conversation-recency check failed (%s); proceeding",
                type(e).__name__,
            )
        try:
            count = self.db.count_notifications_today()
            limit = self.profile.notification_limit
            if count >= limit:
                logger.info(
                    "Skipping proactive (%s): notification limit %d/%d",
                    kind, count, limit,
                )
                return False
            # Warn when approaching limit (80%+)
            if count >= limit * 0.8:
                text += f"\n\n⚠️ Уведомлений сегодня: {count + 1}/{limit}"
        except Exception as e:
            # Don't block sends if counting fails — but DO log so ops can
            # see DB-related failures instead of silently shipping past
            # the rate limit.
            logger.warning(
                "Notification-count check failed (%s); proceeding without limit gate",
                type(e).__name__,
            )
        try:
            await self.send_fn(text)
        except Exception as e:
            logger.error("Proactive send failed: %s", type(e).__name__)
            return False
        # Log failure must NOT downgrade the return to False — the message
        # already went out. If we returned False here, callers like
        # _check_birthdays would skip mark_birthday_alerted, the next day's
        # job re-sends the same alert (for events: reflected_at would stay
        # NULL, retriggering on every 15-min check). Worse than missing a
        # log line.
        try:
            self.db.log_interaction(
                direction="out",
                message_type="notification",
                content=text[:500],
                metadata={"kind": kind},
            )
        except Exception as e:
            logger.warning(
                "Proactive send succeeded but log_interaction failed (%s) — "
                "history may miss this %s",
                type(e).__name__, kind,
            )
        return True

    def _build_action_nudge(self) -> str | None:
        """Build a concrete midday nudge from open loops and stale follow-ups."""
        loops = self.db.get_open_loops(days_ahead=2, limit_per_section=3)
        counts = loops.get("counts", {})
        lines: list[str] = []
        quiet_days = self.settings.quiet_contact_days
        if not isinstance(quiet_days, int):
            quiet_days = 30
        quiet_mentions = self.settings.quiet_contact_min_mentions
        if not isinstance(quiet_mentions, int):
            quiet_mentions = 3
        now = tz_now(self.profile.timezone).replace(tzinfo=None)

        if counts.get("overdue_reminders"):
            first = loops["overdue_reminders"][0]
            n = counts["overdue_reminders"]
            word = pluralize_ru(n, ("напоминание", "напоминания", "напоминаний"))
            lines.append(
                f"Просрочено {n} {word}. "
                f"Ближайшее: {first['text'][:80]}"
            )

        for reminder in loops.get("due_today_reminders", []):
            try:
                trigger_at = datetime.fromisoformat(reminder["trigger_at"].replace(" ", "T"))
            except (ValueError, TypeError, AttributeError):
                continue
            delta = trigger_at - now
            if timedelta(0) <= delta <= ACTION_NUDGE_REMINDER_HORIZON:
                lines.append(
                    f"Скоро напоминание: {trigger_at.strftime('%H:%M')} "
                    f"{reminder['text'][:80]}"
                )
                break

        for event in loops.get("upcoming_events", []):
            try:
                start_at = datetime.fromisoformat(event["start_at"].replace(" ", "T"))
            except (ValueError, TypeError, AttributeError):
                continue
            delta = start_at - now
            if timedelta(0) <= delta <= ACTION_NUDGE_EVENT_HORIZON:
                lines.append(
                    f"Скоро событие: {start_at.strftime('%H:%M')} "
                    f"{event['title'][:80]}"
                )
                break

        urgent_goals = [
            g for g in loops.get("pending_goals", [])
            if g.get("priority") == "high"
        ]
        if urgent_goals:
            lines.append(f"Незакрытая приоритетная цель: {urgent_goals[0]['title'][:90]}")

        if counts.get("due_decisions"):
            first_decision = loops["due_decisions"][0]
            lines.append(
                f"Нужно закрыть follow-up по решению: "
                f"{first_decision['description'][:90]}"
            )

        try:
            alerts = check_contact_frequency(self.db)
            filtered = [
                a for a in alerts
                if a.get("days_since", 0) > quiet_days
                and a.get("mention_count", 0) >= quiet_mentions
            ]
            if filtered:
                first = filtered[0]
                days = int(first.get("days_since") or 0)
                plural = pluralize_ru(days, ("день", "дня", "дней"))
                lines.append(
                    f"Тихий контакт: {first['name']} — не общались {days} {plural}"
                )
        except Exception as e:
            # Quiet-contact check is best-effort — failure shouldn't block
            # the rest of the action nudge, but ops needs to see broken
            # contact-frequency queries (typically a TZ or column-rename
            # regression) instead of silently dropping the section.
            logger.warning(
                "Quiet-contact check in nudge failed: %s", type(e).__name__,
            )

        if not lines:
            return None
        return "⚠️ На контроле:\n" + "\n".join(f"- {line}" for line in lines[:4])

    # --- Scheduler lifecycle ---

    # Cron jobs miss their fire window if the bot is offline at the exact
    # scheduled minute. APScheduler's default misfire_grace_time=1s would
    # silently drop a 09:00 briefing if the bot started at 09:01 — next
    # fire 24h later. Grant 1h grace so a delayed startup still catches
    # daily jobs, with coalesce=True collapsing multiple missed fires
    # (multi-day downtime) into a single run instead of stampeding.
    # Interval jobs (reminder_check, event_alert_check, etc.) keep
    # the default — missing one is fine, next interval fires soon
    # anyway, and stale catch-up could spam after a weekend offline.
    _CRON_GRACE_SECONDS = 3600

    def _add_cron(self, fn, **trigger_kwargs):
        """Wrapper for cron jobs that uniformly applies the grace and
        coalesce policy. id and replace_existing must be passed by
        caller; trigger fields (hour/minute/day_of_week) go through
        as kwargs."""
        job_id = trigger_kwargs.pop("id")
        self.scheduler.add_job(
            fn, "cron",
            id=job_id, replace_existing=True,
            misfire_grace_time=self._CRON_GRACE_SECONDS,
            coalesce=True,
            **trigger_kwargs,
        )

    def start(self):
        # Reminders — always on, bypass quiet hours (user intent)
        self.scheduler.add_job(
            self._check_reminders, "interval", minutes=self.settings.reminder_check_minutes,
            id="reminder_check", replace_existing=True,
        )

        if self.settings.event_alerts and self.settings.event_alert_lead_minutes > 0:
            self.scheduler.add_job(
                self._check_event_alerts, "interval",
                minutes=self.settings.event_alert_check_minutes,
                id="event_alert_check", replace_existing=True,
            )

        if (self.settings.event_reflections
                and self.settings.event_reflection_lag_minutes > 0):
            self.scheduler.add_job(
                self._check_event_reflections, "interval",
                minutes=self.settings.event_reflection_check_minutes,
                id="event_reflection_check", replace_existing=True,
            )

        if self.settings.birthday_alerts:
            self._add_cron(
                self._check_birthdays, hour=9, minute=0,
                id="birthday_check",
            )

        if self.weather and self.settings.weather_monitor:
            self.scheduler.add_job(
                self._check_weather, "interval", minutes=self.settings.weather_check_minutes,
                id="weather_monitor", replace_existing=True,
            )

        if self.settings.morning_briefing:
            wake_h, wake_m = map(int, self.profile.wake_up.split(":"))
            self._add_cron(
                self._morning_prompt, hour=wake_h, minute=wake_m,
                id="morning_prompt",
            )

        if self.settings.smart_questions:
            self._add_cron(
                self._smart_question, hour=13, minute=0,
                id="smart_question",
            )

        if self.settings.decision_followups:
            self._add_cron(
                self._check_decision_followups, hour=10, minute=0,
                id="decision_followup",
            )

        if self.settings.evening_summary:
            self._add_cron(
                self._evening_prompt, hour=21, minute=0,
                id="evening_prompt",
            )

        if self.settings.weekly_review:
            self._add_cron(
                self._weekly_review, day_of_week="sun", hour=20, minute=0,
                id="weekly_review",
            )

        # Daily DB backup at 03:30 — mirrors scripts/backup.sh defaults
        # (data/backups/mindsecretary_*.db, keep last 30) but runs
        # automatically without a user-set crontab. 03:30 lands after
        # the Sunday 03:00 cleanup so weekly snapshots reflect the
        # post-cleanup state; on other days it's just a quiet pre-dawn
        # slot far from morning_prompt and reminder_check noise.
        self._add_cron(
            self._daily_backup, hour=3, minute=30,
            id="daily_backup",
        )

        # Weekly data cleanup — deletes interactions / api_costs / soft-deleted
        # memories older than settings.data_retention_days. Sunday 03:00 to
        # avoid colliding with weekly_review at 20:00 and morning briefing.
        self._add_cron(
            self._cleanup_old_data, day_of_week="sun", hour=3, minute=0,
            id="cleanup_old_data",
        )

        self.scheduler.start()
        logger.info("Proactive scheduler started with %d jobs",
                     len(self.scheduler.get_jobs()))

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    # --- Jobs ---

    async def _check_reminders(self):
        """User-defined reminders — always fire regardless of quiet hours."""
        try:
            sent = await check_reminders(self.db, self.send_fn)
            if sent:
                logger.info("Sent %d reminders", sent)
        except Exception as e:
            logger.error("Reminder check failed: %s", type(e).__name__)

    async def _check_event_alerts(self):
        """Pre-event alerts — fire `event_alert_lead_minutes` before each
        calendar event. Bypasses quiet hours: imminent calendar events
        are user-scheduled commitments, not discretionary pings, so
        treating them like reminders is the right semantics."""
        try:
            sent = await check_event_alerts(
                self.db, self.send_fn,
                lead_minutes=self.settings.event_alert_lead_minutes,
            )
            if sent:
                logger.info("Sent %d event alerts", sent)
        except Exception as e:
            logger.error("Event alert check failed: %s", type(e).__name__)

    async def _check_event_reflections(self):
        """Post-event reflection — fire `event_reflection_lag_minutes`
        after each event's end_at. Routes through _send_proactive so
        quiet hours / snooze / notification limits apply. Reflections
        are discretionary (unlike pre-event alerts which match user-
        scheduled commitments), so respecting the gates is the right
        semantics — late-night events shouldn't trigger 02:00 pings.

        Window cap (event_reflection_window_minutes, default 3h) drops
        events that ended too long ago. A reflection suppressed by
        quiet hours retries each tick until either the gate clears or
        the window closes — better than at-most-once losing every
        late-evening event.
        """
        try:
            sent = await check_event_reflections(
                self.db,
                lambda text: self._send_proactive(text, kind="event_reflection"),
                lag_minutes=self.settings.event_reflection_lag_minutes,
                window_minutes=self.settings.event_reflection_window_minutes,
            )
            if sent:
                logger.info("Sent %d event reflections", sent)
        except Exception as e:
            logger.error("Event reflection check failed: %s", type(e).__name__)

    async def _check_birthdays(self):
        """Daily birthday alert with 7-day dedup per contact."""
        try:
            now = tz_now(self.profile.timezone)
            today_md = now.strftime("%m-%d")
            upcoming = self.db.get_upcoming_birthdays(days=3, skip_recent_alerts=True)
            for c in upcoming:
                text = self._format_birthday_alert(c, now, today_md)
                if not text:
                    continue
                if await self._send_proactive(text, kind="birthday_alert"):
                    self.db.mark_birthday_alerted(c["id"])
        except Exception as e:
            logger.error("Birthday check failed: %s", type(e).__name__)

    @staticmethod
    def _format_birthday_alert(contact: dict, now: datetime, today_md: str) -> str:
        """Render a birthday alert line for one contact.

        The DB stores birthdays as either 'YYYY-MM-DD' (full) or 'MM-DD'
        (year unknown). When the year is present we compute age this
        year, which makes "Маше исполняется 35" type framing possible.
        For year-less rows we omit the parens entirely — better than
        rendering a wrong age guess.

        Days-until is computed against the next occurrence of the
        birthday (this year if upcoming, next year if MM-DD already
        passed). 0 days = today, 1 day = tomorrow, etc.
        """
        bday = (contact.get("birthday") or "").strip()
        if not bday or len(bday) < 5:
            return ""
        bday_md = bday[-5:]
        # Validate MM-DD shape early — birthday column is mostly clean
        # (LLM tool sets it via update_contact) but a corrupted row like
        # "garbage" would otherwise reach the fallback render path and
        # produce "📅 Скоро ДР: X — rbage", which looks broken to the user.
        # Better to skip silently — the contact still appears in /people.
        try:
            mm, dd = bday_md.split("-")
            month_check = int(mm)
            day_check = int(dd)
            if not (1 <= month_check <= 12 and 1 <= day_check <= 31):
                return ""
        except (ValueError, AttributeError):
            return ""
        name = contact.get("name") or "?"
        relation = f" ({contact['relation']})" if contact.get("relation") else ""

        # Pull year if we have one; YYYY-MM-DD has length >= 10.
        birth_year: int | None = None
        if len(bday) >= 10:
            try:
                birth_year = int(bday[:4])
            except (ValueError, TypeError):
                birth_year = None

        # Compute age if we know the year. Use the next occurrence year
        # so an alert 3 days before the birthday says "isthis year's age"
        # rather than "last year's age".
        age_str = ""
        if birth_year is not None:
            try:
                month, day = (int(x) for x in bday_md.split("-"))
                next_year = now.year if (month, day) >= (now.month, now.day) else now.year + 1
                age = next_year - birth_year
                if 0 < age < 150:
                    age_str = f" ({age})"
            except (ValueError, TypeError):
                age_str = ""

        if bday_md == today_md:
            return f"🎂 Сегодня ДР: {name}{age_str}{relation}"

        # Days-until: count calendar days from `now` to the next bday_md.
        try:
            month, day = (int(x) for x in bday_md.split("-"))
            target_year = now.year if (month, day) >= (now.month, now.day) else now.year + 1
            target = now.replace(
                year=target_year, month=month, day=day,
                hour=0, minute=0, second=0, microsecond=0,
            )
            today_midnight = now.replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            days_until = (target - today_midnight).days
        except (ValueError, TypeError):
            days_until = None

        if days_until is None or days_until <= 0:
            # Defensive: can't compute or somehow today — fall back to
            # the today-format so we don't render "через 0 дней" or
            # negative numbers.
            return f"📅 Скоро ДР: {name}{age_str}{relation} — {bday_md}"

        plural = pluralize_ru(days_until, ("день", "дня", "дней"))
        return (
            f"📅 ДР через {days_until} {plural}: "
            f"{name}{age_str}{relation} — {bday_md}"
        )

    def _load_alerted_rain(self, today: str) -> set[int]:
        """Return hours already alerted for today, empty set on new day/missing.

        Defensive: a corrupted pref (non-dict JSON, missing keys, wrong
        types) returns an empty set rather than blowing up the scheduler.
        Worst case: same hour gets re-alerted once. Cheap downside.
        """
        pref = self.db.get_preference(WEATHER_ALERT_PREF_KEY)
        if not pref:
            return set()
        try:
            stored = json.loads(pref["value"])
        except (ValueError, TypeError):
            return set()
        if not isinstance(stored, dict) or stored.get("date") != today:
            return set()
        hours = stored.get("hours") or []
        if not isinstance(hours, list):
            return set()
        return {int(h) for h in hours if isinstance(h, (int, float))}

    def _save_alerted_rain(self, today: str, hours: set[int]) -> None:
        self.db.set_preference(
            WEATHER_ALERT_PREF_KEY,
            json.dumps({"date": today, "hours": sorted(hours)}),
            confidence=1.0, source="system",
        )

    async def _check_weather(self):
        """Alert on new rain hours — TZ-aware, future-only, dedup per-day.

        Guards against three past failure modes:
        1. weather.py computed "now" in system TZ (UTC in Docker), so rain
           at 13:00 local got reported at 18:56 local.
        2. _last_forecast lived only in memory, so process restarts re-
           alerted on the same rain that was already sent.
        3. The bare hour list read poorly when rain spanned 4+ hours.
        """
        try:
            if not self.weather:
                return
            forecast = await self.weather.get_forecast(days=1)
            if "error" in forecast:
                return

            now = tz_now(self.profile.timezone)
            today = now.strftime("%Y-%m-%d")

            # Defensive re-filter: accept only hours strictly in the future
            # relative to scheduler's own clock, even if weather.py slips.
            future_rain = [
                (h, p, c) for (h, p, c) in (forecast.get("rain_today") or [])
                if h >= now.hour
            ]
            if not future_rain:
                return

            already_alerted = self._load_alerted_rain(today)
            fresh = [(h, p, c) for (h, p, c) in future_rain if h not in already_alerted]
            if not fresh:
                return

            text = _format_rain_alert(fresh, now.hour)
            if await self._send_proactive(text, kind="weather_alert"):
                self._save_alerted_rain(
                    today, already_alerted | {h for h, _, _ in fresh}
                )
        except Exception as e:
            logger.error("Weather check failed: %s", type(e).__name__)

    async def _morning_prompt(self):
        try:
            if self.briefing_generator:
                text = await self.briefing_generator.generate_morning()
                if text:
                    await self._send_proactive(text, kind="morning_briefing")
                    return
                logger.warning("Morning briefing returned None, sending fallback")
            await self._send_proactive(
                "☀️ Доброе утро! Брифинг пока недоступен — расскажи, что сегодня в планах?",
                kind="morning_briefing",
            )
        except Exception as e:
            logger.error("Morning prompt failed: %s", type(e).__name__)
            await self._send_proactive(
                "☀️ Доброе утро! (брифинг не удался — расскажи сам, что в планах)",
                kind="morning_briefing",
            )

    def _get_last_nudge(self) -> datetime | None:
        pref = self.db.get_preference("action_nudge_last_sent")
        if pref:
            try:
                return datetime.fromisoformat(pref["value"])
            except (ValueError, TypeError):
                pass
        return None

    def _set_last_nudge(self):
        # Store in profile TZ so the preference is readable in the user's
        # clock (old rows were naive system time = UTC in Docker).
        self.db.set_preference(
            "action_nudge_last_sent",
            tz_now(self.profile.timezone).isoformat(),
            confidence=1.0, source="system",
        )

    async def _smart_question(self):
        """Midday: urgent action nudge or fallback smart question.

        A 44h cooldown intentionally leaves room for smart questions on the
        intervening days when urgent loops stay open for a while.
        """
        try:
            last_nudge = self._get_last_nudge()
            nudge_on_cooldown = False
            if last_nudge is not None:
                # Normalize both sides to UTC-naive before subtracting.
                # Legacy rows (pre-v0.12.2) were saved via `datetime.now()`
                # which is system-UTC in Docker — already UTC-naive.
                # New rows are TZ-aware in the profile clock; convert to UTC
                # so the elapsed-time math is correct regardless of offset.
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                if last_nudge.tzinfo is not None:
                    last_ref = last_nudge.astimezone(timezone.utc).replace(tzinfo=None)
                else:
                    last_ref = last_nudge
                nudge_on_cooldown = now_utc - last_ref < ACTION_NUDGE_COOLDOWN
            if not nudge_on_cooldown:
                nudge = self._build_action_nudge()
                if nudge:
                    # Only stamp cooldown when the nudge actually went out.
                    # Pre-fix _set_last_nudge ran unconditionally, so a
                    # nudge suppressed by recent-activity defer or snooze
                    # still locked the next 44h — user mid-chat at 13:00
                    # lost their nudge for two days. Now suppression keeps
                    # the cooldown clear so tomorrow's tick can retry.
                    if await self._send_proactive(nudge, kind="open_loops_nudge"):
                        self._set_last_nudge()
                    return
            if not self.smart_questions:
                return
            text = await self.smart_questions.generate_question()
            if text:
                await self._send_proactive(text, kind="smart_question")
        except Exception as e:
            logger.error("Smart question failed: %s", type(e).__name__)

    async def _check_decision_followups(self):
        """Check decisions due for follow-up. Push follow_up_at +14 days after sending
        so the same decision doesn't repeat every day until the LLM resolves it."""
        try:
            due = self.db.get_pending_decision_followups()
            for d in due:
                # "Ты решил" misled — these rows are status='pending', the
                # user explicitly hasn't decided yet. Use "Обдумывал" so
                # the reflective framing matches the actual state.
                text = (
                    f"📋 Follow-up по решению:\n"
                    f"Обдумывал: {d['description']}\n"
                )
                if d.get("context"):
                    text += f"Контекст: {d['context'][:200]}\n"
                text += "\nК чему пришёл?"

                if await self._send_proactive(text, kind="decision_followup"):
                    self.db.push_decision_followup(d["id"], days=14)
                    logger.info("Decision follow-up sent: %s", d["id"])
        except Exception as e:
            logger.error("Decision follow-up failed: %s", type(e).__name__)

    async def _evening_prompt(self):
        """Evening: summary + diary entry."""
        try:
            if self.briefing_generator:
                text = await self.briefing_generator.generate_evening()
                if text:
                    await self._send_proactive(text, kind="evening_summary")

                diary = await self.briefing_generator.generate_diary()
                if diary:
                    await self._send_proactive(
                        f"📖 Запись в дневнике:\n\n{diary}",
                        kind="diary",
                    )
                    logger.info("Diary entry saved.")
                return

            logger.warning("Evening briefing returned None, sending fallback")
            await self._send_proactive(
                "🌙 Вечерний обзор пока недоступен. Что важного было сегодня?",
                kind="evening_summary",
            )
        except Exception as e:
            logger.error("Evening prompt failed: %s", type(e).__name__)
            await self._send_proactive(
                "🌙 Вечерний обзор не удался. Расскажи сам — что было важного?",
                kind="evening_summary",
            )

    async def _weekly_review(self):
        try:
            if not self.weekly_reflection:
                return
            text = await self.weekly_reflection.generate_weekly_review()
            if text:
                # TelegramBot.send_message already splits long text, so pass whole thing
                await self._send_proactive(text, kind="weekly_review")
                logger.info("Weekly review sent.")
        except Exception as e:
            logger.error("Weekly review failed: %s", type(e).__name__)

    async def _daily_backup(self):
        """Online DB backup via Database.create_backup. Best-effort —
        failures already log inside create_backup; this wrapper just
        catches the worst-case (Database method itself raising) so a
        backup glitch never disturbs the rest of the scheduler."""
        try:
            self.db.create_backup(keep=30)
        except Exception as e:
            logger.error("Daily backup wrapper failed: %s", type(e).__name__)

    async def _cleanup_old_data(self):
        """Weekly hard-delete of old interactions, api_costs, soft-deleted memories."""
        retention = self.settings.data_retention_days
        if retention <= 0:
            logger.info("Cleanup skipped: data_retention_days=%d (disabled)", retention)
            return
        try:
            counts = self.db.cleanup_old_data(days=retention)
            logger.info(
                "Cleanup: removed %d interactions, %d api_costs, %d memories",
                counts.get("interactions", 0),
                counts.get("api_costs", 0),
                counts.get("memories", 0),
            )
        except Exception as e:
            logger.error("Cleanup failed: %s", type(e).__name__)


def _rain_window_text(start: int, end: int) -> str:
    """Render an inclusive (start..end) rain window in Russian."""
    if start == end:
        return f"в {start:02d}:00"
    end_display = (end + 1) % 24
    return f"с {start:02d}:00 до {end_display:02d}:00"


def _format_rain_alert(fresh: list[tuple[int, int, int]], now_hour: int) -> str:
    """Build a Russian rain alert from (hour, prob, code) triples.

    Groups consecutive hours into windows, picks the heaviest WMO code for
    the label, adds a lead-time hint ("через час", "через ~2ч") so the
    user can tell whether to grab an umbrella right now or later.
    """
    ranges = _merge_rain_hours(fresh)
    window_parts = [_rain_window_text(s, e) for s, e, _ in ranges]
    max_prob = max((p for _, _, p in ranges), default=0)

    # Pick heaviest hour (highest probability) to label the alert and
    # choose between ⛈ (thunderstorm) and 🌧 (plain rain) emojis. If the
    # code is clear/cloudy (data inconsistency), fall back to "дождь" so
    # we don't render "🌧 Ясно".
    heaviest = max(fresh, key=lambda x: (x[1], x[2]))
    heaviest_code = heaviest[2]
    if heaviest_code in RAIN_WMO_CODES:
        label = WMO_CODES.get(heaviest_code, "дождь")
    else:
        label = "дождь"
    emoji = "⛈" if heaviest_code in THUNDERSTORM_CODES else "🌧"

    first_hour = ranges[0][0]
    delta = first_hour - now_hour
    if delta <= 0:
        lead = "начинается"
    elif delta == 1:
        lead = "через час"
    elif delta <= 6:
        lead = f"через ~{delta}ч"
    else:
        lead = None

    windows_text = ", ".join(window_parts)
    headline = f"{emoji} {label.capitalize()}"
    if lead:
        headline = f"{headline} {lead}"
    return f"{headline}: {windows_text} (до {max_prob}%)"
