from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..core.config import Profile
from ..core.database import Database
from ..integrations.weather import WeatherClient
from .monitor import check_birthdays, check_reminders, check_weather_change

logger = logging.getLogger(__name__)


class ProactiveScheduler:
    """Manages all proactive scheduled tasks."""

    def __init__(self, db: Database, weather: WeatherClient | None,
                 profile: Profile, send_fn):
        self.db = db
        self.weather = weather
        self.profile = profile
        self.send_fn = send_fn
        self.scheduler = AsyncIOScheduler()
        self._last_forecast: dict | None = None
        # Set externally after creation
        self.briefing_generator = None
        self.weekly_reflection = None
        self.smart_questions = None

    def start(self):
        # Reminder checker — every 5 min
        self.scheduler.add_job(
            self._check_reminders, "interval", minutes=5,
            id="reminder_check", replace_existing=True,
        )

        # Birthday checker — daily at 09:00
        self.scheduler.add_job(
            self._check_birthdays, "cron", hour=9, minute=0,
            id="birthday_check", replace_existing=True,
        )

        # Weather monitor — every 60 min
        if self.weather:
            self.scheduler.add_job(
                self._check_weather, "interval", minutes=60,
                id="weather_monitor", replace_existing=True,
            )

        # Morning briefing — at wake_up
        wake_h, wake_m = map(int, self.profile.wake_up.split(":"))
        self.scheduler.add_job(
            self._morning_prompt, "cron", hour=wake_h, minute=wake_m,
            id="morning_prompt", replace_existing=True,
        )

        # Smart question — midday (13:00)
        self.scheduler.add_job(
            self._smart_question, "cron", hour=13, minute=0,
            id="smart_question", replace_existing=True,
        )

        # Decision follow-ups — daily at 10:00
        self.scheduler.add_job(
            self._check_decision_followups, "cron", hour=10, minute=0,
            id="decision_followup", replace_existing=True,
        )

        # Evening summary + diary — at 21:00
        self.scheduler.add_job(
            self._evening_prompt, "cron", hour=21, minute=0,
            id="evening_prompt", replace_existing=True,
        )

        # Weekly review — Sunday 20:00
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
        try:
            sent = await check_reminders(self.db, self.send_fn)
            if sent:
                logger.info("Sent %d reminders", sent)
        except Exception as e:
            logger.error("Reminder check failed: %s", e)

    async def _check_birthdays(self):
        try:
            sent = await check_birthdays(self.db, self.send_fn)
            if sent:
                logger.info("Sent %d birthday alerts", sent)
        except Exception as e:
            logger.error("Birthday check failed: %s", e)

    async def _check_weather(self):
        try:
            self._last_forecast = await check_weather_change(
                self.weather, self._last_forecast, self.send_fn,
            )
        except Exception as e:
            logger.error("Weather check failed: %s", e)

    async def _morning_prompt(self):
        try:
            if self.briefing_generator:
                text = await self.briefing_generator.generate_morning()
                if text:
                    await self.send_fn(text)
                    return
            await self.send_fn("☀️ Доброе утро! Что сегодня в планах?")
        except Exception as e:
            logger.error("Morning prompt failed: %s", e)

    async def _smart_question(self):
        """Midday: ask one targeted question to fill knowledge gaps."""
        try:
            if not self.smart_questions:
                return
            text = await self.smart_questions.generate_question()
            if text:
                await self.send_fn(text)
        except Exception as e:
            logger.error("Smart question failed: %s", e)

    async def _check_decision_followups(self):
        """Check for decisions that need follow-up."""
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
                await self.send_fn(text)
                logger.info("Decision follow-up sent: %s", d["id"])
        except Exception as e:
            logger.error("Decision follow-up failed: %s", e)

    async def _evening_prompt(self):
        """Evening: summary + diary entry."""
        try:
            # 1. Evening summary
            if self.briefing_generator:
                text = await self.briefing_generator.generate_evening()
                if text:
                    await self.send_fn(text)

                # 2. Auto-diary (generated quietly, saved to DB)
                diary = await self.briefing_generator.generate_diary()
                if diary:
                    await self.send_fn(f"📖 Запись в дневнике:\n\n{diary}")
                    logger.info("Diary entry saved.")
                return

            await self.send_fn("🌙 Что важного было сегодня?")
        except Exception as e:
            logger.error("Evening prompt failed: %s", e)

    async def _weekly_review(self):
        try:
            if not self.weekly_reflection:
                return
            text = await self.weekly_reflection.generate_weekly_review()
            if text:
                if len(text) > 4000:
                    await self.send_fn(text[:4000])
                    await self.send_fn(text[4000:])
                else:
                    await self.send_fn(text)
                logger.info("Weekly review sent.")
        except Exception as e:
            logger.error("Weekly review failed: %s", e)
