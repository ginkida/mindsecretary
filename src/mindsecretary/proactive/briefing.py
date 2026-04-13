from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ..core.config import Profile
from ..core.database import Database
from ..core.memory import Memory
from ..integrations.weather import WeatherClient
from ..learning.mood import analyze_mood, check_contact_frequency
from ..llm.prompts import BRIEFING_SYSTEM_PROMPT, DIARY_SYSTEM_PROMPT, EVENING_SYSTEM_PROMPT
from ..llm.router import ModelRouter

logger = logging.getLogger(__name__)

DAYS_RU = {
    0: "Понедельник", 1: "Вторник", 2: "Среда", 3: "Четверг",
    4: "Пятница", 5: "Суббота", 6: "Воскресенье",
}


class BriefingGenerator:
    def __init__(self, router: ModelRouter, memory: Memory, db: Database,
                 weather: WeatherClient | None, profile: Profile):
        self.router = router
        self.memory = memory
        self.db = db
        self.weather = weather
        self.profile = profile

    async def generate_morning(self) -> str | None:
        """Generate morning briefing text."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        # Gather data
        weather_text = "Погода недоступна."
        if self.weather:
            try:
                forecast = await self.weather.get_forecast(days=2)
                weather_text = self.weather.format_current(forecast)
            except Exception as e:
                logger.error("Weather for briefing failed: %s", e)

        events = self.db.get_events(today)
        events_text = "\n".join(
            f"- {e['start_at'][11:16] if len(e['start_at']) > 10 else '??:??'} {e['title']}"
            + (f" (с {e['related_person']})" if e.get("related_person") else "")
            for e in events
        ) or "Нет событий."

        reminders = self.db.get_pending_reminders()
        reminders_text = "\n".join(
            f"- {r['text']}" for r in reminders[:5]
        ) or "Нет напоминаний."

        birthdays = self.db.get_upcoming_birthdays(days=7)
        birthdays_text = "\n".join(
            f"- {c['name']}" + (f" ({c['relation']})" if c.get("relation") else "")
            + f" — {c['birthday']}"
            for c in birthdays
        ) or "Нет ближайших ДР."

        # Semantic search for relevant context
        context_query = f"важное на {DAYS_RU[now.weekday()]} {today}"
        memories = await self.memory.search(context_query, top_k=5)
        memories_text = "\n".join(
            f"- {m['content']}" for m in memories
        ) or "Нет релевантных воспоминаний."

        promises = await self.memory.search("незакрытые обещания", category="promise", top_k=5)
        promises_text = "\n".join(
            f"- {m['content']}" for m in promises
        ) or "Нет незакрытых обещаний."

        prompt = BRIEFING_SYSTEM_PROMPT.format(
            name=self.profile.name,
            style=self.profile.style,
            profile=self.profile.to_yaml_str(),
            date=today,
            day_of_week=DAYS_RU[now.weekday()],
            weather=weather_text,
            events=events_text,
            birthdays=birthdays_text,
            reminders=reminders_text,
            promises=promises_text,
            memories=memories_text,
        )

        try:
            response = await self.router.chat(
                system=prompt,
                messages=[{"role": "user", "content": "Сгенерируй утренний брифинг."}],
                max_tokens=800,
            )
            text = response.text
            if text:
                self.db.log_interaction(
                    direction="out", message_type="briefing", content=text,
                )
            return text
        except Exception as e:
            logger.error("Morning briefing generation failed: %s", e)
            return None

    async def generate_evening(self) -> str | None:
        """Generate evening summary text."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        # Today's interactions
        today_start = now.replace(hour=0, minute=0, second=0)
        interactions = self.db.get_interactions(since=today_start, limit=50)
        interactions_text = "\n".join(
            f"[{i['timestamp'][11:16]}] {'→' if i['direction'] == 'out' else '←'} "
            f"({i['message_type']}) {i['content'][:100]}"
            for i in interactions[:30]
        ) or "Нет взаимодействий."

        events = self.db.get_events(today)
        events_text = "\n".join(
            f"- {e['title']}" for e in events
        ) or "Нет событий."

        events_tomorrow = self.db.get_events(tomorrow)
        events_tomorrow_text = "\n".join(
            f"- {e['start_at'][11:16] if len(e['start_at']) > 10 else ''} {e['title']}"
            for e in events_tomorrow
        ) or "Нет событий."

        weather_tomorrow = "Недоступна."
        if self.weather:
            try:
                forecast = await self.weather.get_forecast(days=2)
                daily = forecast.get("daily", [])
                if len(daily) > 1:
                    d = daily[1]
                    weather_tomorrow = f"{d['temp_min']}..{d['temp_max']}°C, {d['condition']}"
            except Exception:
                pass

        # Count completed reminders
        completed = [i for i in interactions
                     if i.get("message_type") == "reminder" and i.get("direction") == "out"]

        # Daily goals
        goals = self.db.get_daily_goals(today)
        if goals:
            goals_lines = []
            for g in goals:
                status_label = {"pending": "не отмечена", "completed": "выполнена",
                                "skipped": "пропущена", "partial": "частично"}.get(g["status"], g["status"])
                line = f"- {g['title']} [{status_label}]"
                if g.get("reflection"):
                    line += f" — {g['reflection'][:100]}"
                goals_lines.append(line)
            goals_text = "\n".join(goals_lines)
        else:
            goals_text = "Цели не были поставлены."

        prompt = EVENING_SYSTEM_PROMPT.format(
            name=self.profile.name,
            date=today,
            interactions=interactions_text,
            events=events_text,
            completed=f"{len(completed)} напоминаний отправлено",
            daily_goals=goals_text,
            weather_tomorrow=weather_tomorrow,
            events_tomorrow=events_tomorrow_text,
        )

        try:
            response = await self.router.chat(
                system=prompt,
                messages=[{"role": "user", "content": "Сгенерируй вечерний итог."}],
                max_tokens=600,
            )
            text = response.text
            if text:
                self.db.log_interaction(
                    direction="out", message_type="briefing", content=text,
                )
            return text
        except Exception as e:
            logger.error("Evening summary generation failed: %s", e)
            return None

    async def generate_diary(self) -> str | None:
        """Generate auto-diary entry from the day's interactions."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        today_start = now.replace(hour=0, minute=0, second=0)

        interactions = self.db.get_interactions(since=today_start, limit=100)
        if len(interactions) < 3:
            return None

        # Mood analysis
        mood = analyze_mood(interactions)

        # People mentioned today
        contacts = self.db.get_contacts("")
        people_today = set()
        for i in interactions:
            content = (i.get("content") or "").lower()
            for c in contacts:
                if c["name"].lower() in content:
                    people_today.add(c["name"])

        # Events
        events = self.db.get_events(today)

        # Relationship alerts
        rel_alerts = check_contact_frequency(self.db)

        interactions_text = "\n".join(
            f"[{i['timestamp'][11:16]}] {'←' if i['direction'] == 'in' else '→'} {i['content'][:120]}"
            for i in interactions[:40]
        )

        events_text = "\n".join(f"- {e['title']}" for e in events) or "Нет"
        people_text = ", ".join(people_today) or "Никого"
        mood_text = f"Score: {mood['score']}, label: {mood['label']}, signals: {mood['signals']}"

        rel_text = ""
        if rel_alerts:
            rel_text = "\n".join(
                f"- {a['name']} ({a.get('relation', '?')}): не общались {a['days_since']} дней"
                for a in rel_alerts
            )

        try:
            response = await self.router.chat(
                system=DIARY_SYSTEM_PROMPT.format(
                    name=self.profile.name,
                    date=today,
                    day_of_week=DAYS_RU[now.weekday()],
                    interactions=interactions_text,
                    events=events_text,
                    people=people_text,
                    mood=mood_text,
                    relationship_alerts=rel_text or "Нет алертов.",
                ),
                messages=[{"role": "user", "content": "Напиши запись в дневник за сегодня."}],
                max_tokens=800,
            )
            text = response.text
            if text:
                self.db.save_diary_entry(
                    date=today, content=text,
                    mood=mood["label"],
                    people=", ".join(people_today),
                )
                self.db.log_interaction(
                    direction="out", message_type="diary", content=text,
                )
            return text
        except Exception as e:
            logger.error("Diary generation failed: %s", e)
            return None
