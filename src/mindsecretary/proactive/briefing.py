from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ..core import DAYS_RU, fmt_local_time, pluralize_ru, tz_now
from ..core.config import Profile
from ..core.database import Database
from ..core.memory import Memory
from ..core.prompt_safety import sanitize_for_context
from ..integrations.weather import WeatherClient
from ..learning.mood import analyze_mood, check_contact_frequency
from ..llm.prompts import BRIEFING_SYSTEM_PROMPT, DIARY_SYSTEM_PROMPT, EVENING_SYSTEM_PROMPT
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class BriefingGenerator:
    def __init__(self, llm: LLMClient, memory: Memory, db: Database,
                 weather: WeatherClient | None, profile: Profile):
        self.llm = llm
        self.memory = memory
        self.db = db
        self.weather = weather
        self.profile = profile

    @staticmethod
    def _format_reminder_line(reminder: dict) -> str:
        """Render one pending reminder for briefing prompts. Shows trigger
        time so Claude can sequence the day correctly — pre-fix the
        morning briefing emitted reminders without their times, leading
        to outputs like 'позвонить маме' with no hint when. trigger_at
        is profile-local naive, so the [:16] slice gives 'YYYY-MM-DD
        HH:MM' as the user actually scheduled it."""
        s = sanitize_for_context
        trigger = (reminder.get("trigger_at") or "")[:16] or "??"
        text = s(reminder.get("text") or "", 200)
        line = f"- {trigger} {text}"
        recurrence = reminder.get("recurrence")
        if recurrence:
            line += f" ({recurrence})"
        return line

    @staticmethod
    def _format_event_line(event: dict) -> str:
        """Render one event for briefing prompts: 'HH:MM Title (с Person, где: Loc)'.

        Consolidates three previously inconsistent inline formats (morning
        vs evening's today-events vs evening's tomorrow-events). Evening
        used to drop time entirely. Each user-origin field passes through
        sanitize_for_context — events end up in system prompts and an
        event title is the same channel a malicious forward could reach.
        """
        s = sanitize_for_context
        start = event.get("start_at") or ""
        time_str = start[11:16] if len(start) >= 16 else "??:??"
        title = s(event.get("title") or "", 200)
        line = f"- {time_str} {title}"
        extras: list[str] = []
        person = event.get("related_person")
        if person:
            extras.append(f"с {s(person, 100)}")
        location = event.get("location")
        if location:
            extras.append(f"где: {s(location, 100)}")
        if extras:
            line += f" ({', '.join(extras)})"
        return line

    @staticmethod
    def _format_age(days: int) -> str:
        """Render an age-in-days as a coarse anniversary label.

        We bucket so the briefing reads naturally — "год назад", not
        "365 дней назад". Buckets line up with how people actually talk
        about anniversaries: years if applicable, otherwise months.

        Russian plural forms via pluralize_ru — handles the 11-14 teens
        and 20+/30+/etc edge cases that the original inline `< 5` rule
        got wrong (e.g. "21 лет" → "21 год", "22 лет" → "22 года").
        """
        if days >= 365:
            years = days // 365
            return f"{years} {pluralize_ru(years, ('год', 'года', 'лет'))} назад"
        if days >= 30:
            months = days // 30
            return f"{months} {pluralize_ru(months, ('месяц', 'месяца', 'месяцев'))} назад"
        return f"{days} дн. назад"

    @staticmethod
    def _format_open_loops(snapshot: dict) -> str:
        counts = snapshot.get("counts", {})
        lines = []
        if counts.get("overdue_reminders"):
            lines.append(f"- Просроченные напоминания: {counts['overdue_reminders']}")
        if counts.get("due_today_reminders"):
            lines.append(f"- Напоминания до конца дня: {counts['due_today_reminders']}")
        if counts.get("pending_goals"):
            lines.append(f"- Незакрытые цели на сегодня: {counts['pending_goals']}")
        if counts.get("due_decisions"):
            lines.append(f"- Решения с просроченным follow-up: {counts['due_decisions']}")
        if snapshot.get("upcoming_events"):
            first = snapshot["upcoming_events"][0]
            # Event title is user-controlled → sanitize before it lands in the
            # briefing system prompt, matching the pattern used elsewhere.
            title = sanitize_for_context(first.get("title") or "", 200)
            lines.append(f"- Ближайшее событие: {first['start_at'][11:16]} {title}")
        return "\n".join(lines) or "Критичных хвостов нет."

    async def generate_morning(self) -> str | None:
        """Generate morning briefing text."""
        now = tz_now(self.profile.timezone)
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        s = sanitize_for_context

        # Gather data
        weather_text = "Погода недоступна."
        if self.weather:
            try:
                forecast = await self.weather.get_forecast(days=2)
                weather_text = self.weather.format_current(forecast)
            except Exception as e:
                logger.error("Weather for briefing failed: %s", type(e).__name__)

        events = self.db.get_events(today)
        events_text = "\n".join(
            self._format_event_line(e) for e in events
        ) or "Нет событий."

        reminders = self.db.get_pending_reminders()
        reminders_text = "\n".join(
            self._format_reminder_line(r) for r in reminders[:5]
        ) or "Нет напоминаний."

        birthdays = self.db.get_upcoming_birthdays(days=7)
        birthdays_text = "\n".join(
            f"- {s(c['name'], 80)}" + (f" ({s(c['relation'], 60)})" if c.get("relation") else "")
            + f" — {c['birthday']}"
            for c in birthdays
        ) or "Нет ближайших ДР."

        # Semantic search for relevant context
        context_query = f"важное на {DAYS_RU[now.weekday()]} {today}"
        memories = await self.memory.search(context_query, top_k=5)
        memories_text = "\n".join(
            f"- {s(m['content'])}" for m in memories
        ) or "Нет релевантных воспоминаний."

        promises = await self.memory.search("незакрытые обещания", category="promise", top_k=5)
        promises_text = "\n".join(
            f"- {s(m['content'])}" for m in promises
        ) or "Нет незакрытых обещаний."
        open_loops = self.db.get_open_loops(days_ahead=2, limit_per_section=3)
        open_loops_text = self._format_open_loops(open_loops)

        # Anniversary recall — items from this calendar date in past
        # months/years. Turns the bot's accumulated memory into living
        # context ("год назад ты решил сменить работу — получилось") so
        # the morning briefing isn't just forward-looking. The LLM
        # decides whether to weave it in; we just surface the data.
        anniversaries_text = "Нет совпадений по дате."
        try:
            anns = self.db.get_anniversaries(limit=5, min_age_days=30)
            if anns:
                lines = []
                for a in anns:
                    age = a.get("age_days", 0)
                    label = self._format_age(age)
                    if a["kind"] == "decision":
                        # Past decision with known outcome — most resonant
                        # form: "год назад решил X — получилось Y".
                        out = s(a.get("outcome") or "", 120)
                        senti = a.get("sentiment") or ""
                        senti_tag = f" [{senti}]" if senti else ""
                        line = (
                            f"- {label}: решил «{s(a['content'], 120)}» — "
                            f"{out}{senti_tag}"
                        )
                    else:
                        line = (
                            f"- {label} [{a.get('category', '?')}]: "
                            f"{s(a['content'], 200)}"
                        )
                    lines.append(line)
                anniversaries_text = "\n".join(lines)
        except Exception as e:
            logger.warning("Anniversaries lookup failed: %s", type(e).__name__)

        # Habits: morning gets the motivation framing — "don't break your
        # streak today" works while user is planning the day. Evening
        # (v0.13.6) gets the reflective framing — "you held it, here's
        # how today went". Different framings, same data source.
        habit_stats = self.db.get_habit_stats()
        if habit_stats:
            active_streaks = [h for h in habit_stats if h["streak"] >= 3]
            if active_streaks:
                streaks_str = ", ".join(
                    f"{s(h['name'], 60)} — {h['streak']}д"
                    for h in active_streaks
                )
                habits_text = f"🔥 Серии: {streaks_str}"
            else:
                habits_text = "Серий нет, всё с нуля."
        else:
            habits_text = "Привычки не отслеживаются."

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
            habits=habits_text,
            anniversaries=anniversaries_text,
            promises=promises_text,
            open_loops=open_loops_text,
            memories=memories_text,
        )

        try:
            response = await self.llm.chat(
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
            logger.error("Morning briefing generation failed: %s", type(e).__name__)
            return None

    async def generate_evening(self) -> str | None:
        """Generate evening summary text."""
        now = tz_now(self.profile.timezone)
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        s = sanitize_for_context

        # Today's interactions — `since` must be UTC (interactions.timestamp
        # is stored via SQLite's `datetime('now')` = UTC). Use the UTC
        # equivalent of local midnight today so "today" matches the user's
        # calendar, not the system clock.
        start_utc_s, _ = self.db._local_day_utc_bounds()
        today_start = datetime.strptime(start_utc_s, "%Y-%m-%d %H:%M:%S")
        interactions = self.db.get_interactions(since=today_start, limit=50)
        # timestamp is UTC-naive; render in profile TZ so Claude sees the
        # times the user actually lived, not UTC wall-clock.
        today_local_str = now.strftime("%Y-%m-%d")
        interactions_text = "\n".join(
            f"[{fmt_local_time(i['timestamp'], self.profile.timezone, today_local_str)}] "
            f"{'→' if i['direction'] == 'out' else '←'} "
            f"({i['message_type']}) {s(i['content'], 100)}"
            for i in interactions[:30]
        ) or "Нет взаимодействий."

        events = self.db.get_events(today)
        events_text = "\n".join(
            self._format_event_line(e) for e in events
        ) or "Нет событий."

        events_tomorrow = self.db.get_events(tomorrow)
        events_tomorrow_text = "\n".join(
            self._format_event_line(e) for e in events_tomorrow
        ) or "Нет событий."

        weather_tomorrow = "Недоступна."
        if self.weather:
            try:
                forecast = await self.weather.get_forecast(days=2)
                daily = forecast.get("daily", [])
                if len(daily) > 1:
                    d = daily[1]
                    weather_tomorrow = f"{d['temp_min']}..{d['temp_max']}°C, {d['condition']}"
            except Exception as e:
                logger.warning("Weather for evening summary failed: %s", type(e).__name__)

        # Count completed reminders
        completed = [i for i in interactions
                     if i.get("message_type") == "reminder" and i.get("direction") == "out"]

        # Habits: active streaks and what wasn't logged today.
        # Habit names are user-origin (created via log_habit from voice/text)
        # so they go through sanitize_for_context like other free-text fields.
        habit_stats = self.db.get_habit_stats()
        if habit_stats:
            active_streaks = [h for h in habit_stats if h["streak"] >= 3]
            unlogged = [h for h in habit_stats if not h.get("logged_today")]
            habit_lines: list[str] = []
            if active_streaks:
                streaks_str = ", ".join(
                    f"{s(h['name'], 60)} — {h['streak']}д"
                    for h in active_streaks
                )
                habit_lines.append(f"🔥 Серии: {streaks_str}")
            if unlogged:
                unlogged_str = ", ".join(s(h["name"], 60) for h in unlogged[:5])
                habit_lines.append(f"Не отмечено сегодня: {unlogged_str}")
            habits_text = "\n".join(habit_lines) or "Всё отмечено, серий нет."
        else:
            habits_text = "Привычки не отслеживаются."

        # Daily goals (sanitize user-origin text before prompt injection)
        goals = self.db.get_daily_goals(today)
        if goals:
            goals_lines = []
            for g in goals:
                status_label = {"pending": "не отмечена", "completed": "выполнена",
                                "skipped": "пропущена", "partial": "частично"}.get(g["status"], g["status"])
                title = s(g["title"] or "", 150)
                line = f"- {title} [{status_label}]"
                if g.get("reflection"):
                    line += f" — {s(g['reflection'] or '', 100)}"
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
            habits=habits_text,
            weather_tomorrow=weather_tomorrow,
            events_tomorrow=events_tomorrow_text,
        )

        try:
            response = await self.llm.chat(
                system=prompt,
                messages=[{"role": "user", "content": "Сгенерируй вечерний итог."}],
                max_tokens=800,
            )
            text = response.text
            if text:
                self.db.log_interaction(
                    direction="out", message_type="briefing", content=text,
                )
            return text
        except Exception as e:
            logger.error("Evening summary generation failed: %s", type(e).__name__)
            return None

    async def generate_diary(self) -> str | None:
        """Generate auto-diary entry from the day's interactions."""
        now = tz_now(self.profile.timezone)
        today = now.strftime("%Y-%m-%d")
        # UTC bound of local midnight today — matches `datetime('now')` storage.
        start_utc_s, _ = self.db._local_day_utc_bounds()
        today_start = datetime.strptime(start_utc_s, "%Y-%m-%d %H:%M:%S")
        s = sanitize_for_context

        interactions = self.db.get_interactions(since=today_start, limit=100)
        if len(interactions) < 3:
            return None
        # At least one inbound message — proactive notifications and bot
        # replies don't count. Without this guard, a day with 5 reminders
        # / briefings and 0 user messages would generate a "diary" purely
        # from the bot's own outputs (nonsense + wasted LLM call).
        if not any(i.get("direction") == "in" for i in interactions):
            logger.info("Diary skipped: no inbound messages today")
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

        # Render stored-UTC timestamps in profile TZ so the diary matches
        # the user's clock, not the system's.
        interactions_text = "\n".join(
            f"[{fmt_local_time(i['timestamp'], self.profile.timezone, today)}] "
            f"{'←' if i['direction'] == 'in' else '→'} {s(i['content'], 120)}"
            for i in interactions[:40]
        )

        events_text = "\n".join(
            self._format_event_line(e) for e in events
        ) or "Нет"
        people_text = ", ".join(s(p, 80) for p in people_today) or "Никого"
        mood_text = f"Score: {mood['score']}, label: {mood['label']}, signals: {mood['signals']}"

        rel_text = ""
        if rel_alerts:
            rel_text = "\n".join(
                f"- {s(a['name'], 80)} ({s(a.get('relation', '?'), 60)}): не общались {a['days_since']} дней"
                for a in rel_alerts
            )

        try:
            response = await self.llm.chat(
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
            logger.error("Diary generation failed: %s", type(e).__name__)
            return None
