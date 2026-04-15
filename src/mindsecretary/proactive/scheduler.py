from __future__ import annotations

import logging
from datetime import datetime, time

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..core import tz_now
from ..core.config import Profile, Settings
from ..core.database import Database
from ..integrations.weather import WeatherClient
from .monitor import check_reminders

logger = logging.getLogger(__name__)


class ProactiveScheduler:
    """Manages all proactive scheduled tasks."""

    def __init__(self, db: Database, weather: WeatherClient | None,
                 profile: Profile, settings: Settings, send_fn):
        self.db = db
        self.weather = weather
        self.profile = profile
        self.settings = settings
        self.send_fn = send_fn
        self.scheduler = AsyncIOScheduler()
        self._last_forecast: dict | None = None
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

    async def _send_proactive(self, text: str, kind: str = "notification") -> bool:
        """Send a proactive message respecting quiet hours and notification limit.

        Returns True if actually sent, False if dropped (quiet hours / limit / error).
        Logs every sent message as message_type='notification' for counting.
        """
        if not text:
            return False
        if self._in_quiet_hours():
            logger.info("Skipping proactive (%s): quiet hours", kind)
            return False
        try:
            count = self.db.count_notifications_today()
            if count >= self.profile.notification_limit:
                logger.info(
                    "Skipping proactive (%s): notification limit %d/%d",
                    kind, count, self.profile.notification_limit,
                )
                return False
        except Exception:
            pass  # Don't block sends if counting fails
        try:
            await self.send_fn(text)
            self.db.log_interaction(
                direction="out",
                message_type="notification",
                content=text[:500],
                metadata={"kind": kind},
            )
            return True
        except Exception as e:
            logger.error("Proactive send failed: %s", type(e).__name__)
            return False

    # --- Scheduler lifecycle ---

    def start(self):
        # Reminders — always on, bypass quiet hours (user intent)
        self.scheduler.add_job(
            self._check_reminders, "interval", minutes=self.settings.reminder_check_minutes,
            id="reminder_check", replace_existing=True,
        )

        if self.settings.birthday_alerts:
            self.scheduler.add_job(
                self._check_birthdays, "cron", hour=9, minute=0,
                id="birthday_check", replace_existing=True,
            )

        if self.weather and self.settings.weather_monitor:
            self.scheduler.add_job(
                self._check_weather, "interval", minutes=self.settings.weather_check_minutes,
                id="weather_monitor", replace_existing=True,
            )

        if self.settings.morning_briefing:
            wake_h, wake_m = map(int, self.profile.wake_up.split(":"))
            self.scheduler.add_job(
                self._morning_prompt, "cron", hour=wake_h, minute=wake_m,
                id="morning_prompt", replace_existing=True,
            )

        if self.settings.smart_questions:
            self.scheduler.add_job(
                self._smart_question, "cron", hour=13, minute=0,
                id="smart_question", replace_existing=True,
            )

        if self.settings.decision_followups:
            self.scheduler.add_job(
                self._check_decision_followups, "cron", hour=10, minute=0,
                id="decision_followup", replace_existing=True,
            )

        if self.settings.evening_summary:
            self.scheduler.add_job(
                self._evening_prompt, "cron", hour=21, minute=0,
                id="evening_prompt", replace_existing=True,
            )

        if self.settings.weekly_review:
            self.scheduler.add_job(
                self._weekly_review, "cron", day_of_week="sun", hour=20, minute=0,
                id="weekly_review", replace_existing=True,
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
            logger.error("Reminder check failed: %s", e)

    async def _check_birthdays(self):
        """Daily birthday alert with 7-day dedup per contact."""
        try:
            today_md = tz_now(self.profile.timezone).strftime("%m-%d")
            upcoming = self.db.get_upcoming_birthdays(days=3, skip_recent_alerts=True)
            for c in upcoming:
                bday = c.get("birthday", "") or ""
                bday_md = bday[-5:] if len(bday) >= 5 else bday
                name = c["name"]
                relation = f" ({c['relation']})" if c.get("relation") else ""

                if bday_md == today_md:
                    text = f"🎂 Сегодня день рождения: {name}{relation}!"
                else:
                    text = f"📅 Скоро день рождения: {name}{relation} — {bday}"

                if await self._send_proactive(text, kind="birthday_alert"):
                    self.db.mark_birthday_alerted(c["id"])
        except Exception as e:
            logger.error("Birthday check failed: %s", e)

    async def _check_weather(self):
        """Alert on new rain hours appearing in the forecast."""
        try:
            if not self.weather:
                return
            forecast = await self.weather.get_forecast(days=1)
            if "error" in forecast:
                return

            if self._last_forecast is None:
                self._last_forecast = forecast
                return

            old_rain = set(h for h, _ in (self._last_forecast.get("rain_today") or []))
            new_rain = set(h for h, _ in (forecast.get("rain_today") or []))
            new_rain_hours = new_rain - old_rain
            self._last_forecast = forecast

            if new_rain_hours:
                hours_str = ", ".join(f"{h}:00" for h in sorted(new_rain_hours))
                await self._send_proactive(
                    f"🌧 Появился дождь в прогнозе: {hours_str}",
                    kind="weather_alert",
                )
        except Exception as e:
            logger.error("Weather check failed: %s", e)

    async def _morning_prompt(self):
        try:
            if self.briefing_generator:
                text = await self.briefing_generator.generate_morning()
                if text:
                    await self._send_proactive(text, kind="morning_briefing")
                    return
            await self._send_proactive(
                "☀️ Доброе утро! Что сегодня в планах?",
                kind="morning_briefing",
            )
        except Exception as e:
            logger.error("Morning prompt failed: %s", e)

    async def _smart_question(self):
        """Midday: ask one targeted question to fill knowledge gaps."""
        try:
            if not self.smart_questions:
                return
            text = await self.smart_questions.generate_question()
            if text:
                await self._send_proactive(text, kind="smart_question")
        except Exception as e:
            logger.error("Smart question failed: %s", e)

    async def _check_decision_followups(self):
        """Check decisions due for follow-up. Push follow_up_at +14 days after sending
        so the same decision doesn't repeat every day until the LLM resolves it."""
        try:
            due = self.db.get_pending_decision_followups()
            for d in due:
                text = (
                    f"📋 Follow-up по решению:\n"
                    f"Ты решил: {d['description']}\n"
                )
                if d.get("context"):
                    text += f"Контекст: {d['context'][:200]}\n"
                text += "\nКак в итоге? Что получилось?"

                if await self._send_proactive(text, kind="decision_followup"):
                    self.db.push_decision_followup(d["id"], days=14)
                    logger.info("Decision follow-up sent: %s", d["id"])
        except Exception as e:
            logger.error("Decision follow-up failed: %s", e)

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

            await self._send_proactive(
                "🌙 Что важного было сегодня?",
                kind="evening_summary",
            )
        except Exception as e:
            logger.error("Evening prompt failed: %s", e)

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
            logger.error("Weekly review failed: %s", e)
