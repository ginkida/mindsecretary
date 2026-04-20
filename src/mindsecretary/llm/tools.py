from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime
from typing import Any

from ..core.database import Database
from ..core.enums import Priority, Sentiment
from ..core.memory import Memory
from ..integrations.weather import WeatherClient

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "name": "save_memory",
        "description": (
            "Сохранить факт, переживание, обещание или наблюдение. "
            "Для эмоций, настроения, личных переживаний используй "
            "category='emotional'. Вызывай для КАЖДОГО нового факта или "
            "важного переживания из сообщения."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Что запомнить. Включи контекст: кто, когда, связь.",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "contact", "health", "work", "personal",
                        "promise", "preference", "location", "learning",
                        "emotional",
                    ],
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "1=мелочь, 5=полезно, 8=важно, 10=критично",
                },
                "related_person": {"type": "string"},
                "related_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["content", "category", "importance"],
        },
    },
    {
        "name": "search_memory",
        "description": "Семантический поиск по памяти — прошлое, люди, обещания, переживания.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": [
                        "contact", "health", "work", "personal",
                        "promise", "preference", "location", "learning",
                        "emotional",
                    ],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_event",
        "description": "Создать событие в календаре.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_at": {"type": "string", "description": "YYYY-MM-DDTHH:MM"},
                "end_at": {"type": "string"},
                "location": {"type": "string"},
                "related_person": {"type": "string", "description": "Кто участвует"},
                "description": {"type": "string"},
            },
            "required": ["title", "start_at"],
        },
    },
    {
        "name": "get_events",
        "description": "Получить события за период.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string"},
            },
            "required": ["date_from"],
        },
    },
    {
        "name": "create_reminder",
        "description": "Создать напоминание. Для повторяющихся — укажи recurrence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "trigger_at": {"type": "string", "description": "YYYY-MM-DDTHH:MM"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "recurrence": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly"],
                    "description": "Повторение: daily/weekly/monthly (опционально)",
                },
            },
            "required": ["text", "trigger_at"],
        },
    },
    {
        "name": "update_contact",
        "description": (
            "Создать или обновить контакт. Вызывай при ЛЮБОМ упоминании "
            "нового факта о человеке: переезд, работа, семья, интересы."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "relation": {"type": "string"},
                "birthday": {"type": "string"},
                "notes": {"type": "string", "description": "Новые факты о человеке"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_contacts",
        "description": "Найти контакты.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Прогноз погоды.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "days": {"type": "integer", "maximum": 7},
            },
        },
    },
    {
        "name": "log_habit",
        "description": "Отметить выполнение привычки.",
        "input_schema": {
            "type": "object",
            "properties": {
                "habit_name": {"type": "string"},
                "date": {"type": "string"},
                "done": {"type": "boolean"},
            },
            "required": ["habit_name", "done"],
        },
    },
    {
        "name": "track_decision",
        "description": (
            "Track a decision the user is making or considering. "
            "Use when user says 'I'm thinking of...', 'decided to...', 'should I...'. "
            "Also search past decisions for similar situations and mention outcomes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "What the decision is about"},
                "context": {"type": "string", "description": "Relevant context: why considering, pros/cons"},
                "follow_up_days": {
                    "type": "integer", "description": "Days until follow-up (default 30)",
                    "minimum": 1, "maximum": 365,
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "set_daily_goal",
        "description": (
            "Записать цель на сегодня. Вызывай когда пользователь ставит себе "
            "задачу или цель на день: 'хочу сегодня...', 'план на день...', "
            "'нужно сделать...'. Для КАЖДОЙ цели вызывай отдельно."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Короткое название цели"},
                "description": {"type": "string", "description": "Детали (если есть)"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["title"],
        },
    },
    {
        "name": "complete_daily_goal",
        "description": (
            "Отметить цель дня как выполненную, пропущенную или частично "
            "выполненную. Вызывай когда пользователь говорит что сделал или "
            "не сделал что-то из целей дня."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal_hint": {
                    "type": "string",
                    "description": "Ключевое слово из названия цели (напр. 'зал', 'отчёт')",
                },
                "status": {
                    "type": "string",
                    "enum": ["completed", "skipped", "partial"],
                },
                "reflection": {
                    "type": "string",
                    "description": "Что помогло или помешало (если озвучено)",
                },
            },
            "required": ["goal_hint", "status"],
        },
    },
    {
        "name": "resolve_decision",
        "description": (
            "Отметить решение как закрытое с итогом. Вызывай когда пользователь "
            "рассказывает чем закончилось решение, которое отслеживалось. "
            "Совпадение по ключевому слову из описания решения."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description_hint": {
                    "type": "string",
                    "description": "Keyword/phrase from the original decision (e.g. 'велосипед')",
                },
                "outcome": {
                    "type": "string",
                    "description": "What actually happened",
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                },
            },
            "required": ["description_hint", "outcome"],
        },
    },
    # Anthropic native server-side web search. Claude runs the search itself
    # and embeds results + citations into its text response. No client-side
    # handler needed. Billed separately ($10 / 1000 searches) — not tracked
    # in api_costs (which only captures token costs).
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    },
]


MAX_STR_LEN = 5000
VALID_CATEGORIES = {
    "contact", "health", "work", "personal",
    "promise", "preference", "location", "learning", "emotional",
}
VALID_PRIORITIES = {p.value for p in Priority}


def _truncate(val: str | None, max_len: int = MAX_STR_LEN) -> str | None:
    if val is None:
        return None
    return str(val)[:max_len]


def _sanitize_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize tool arguments from LLM output."""
    clean = {}
    for k, v in args.items():
        if isinstance(v, str):
            clean[k] = _truncate(v)
        elif isinstance(v, (int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = v
        else:
            clean[k] = str(v)[:MAX_STR_LEN]

    # Tool-specific validation
    if name == "save_memory":
        cat = clean.get("category", "")
        if cat not in VALID_CATEGORIES:
            clean["category"] = "personal"
        imp = clean.get("importance", 5)
        clean["importance"] = max(1, min(10, int(imp)))

    if name in ("create_event", "create_reminder"):
        priority = clean.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            clean["priority"] = "medium"

    # Validate datetime fields — try to parse, normalize to space-separated format
    _DT_FIELDS = {
        "create_event": ("start_at", "end_at"),
        "create_reminder": ("trigger_at",),
    }
    for field in _DT_FIELDS.get(name, ()):
        val = clean.get(field)
        if val:
            try:
                parsed = datetime.fromisoformat(val.replace(" ", "T"))
                clean[field] = parsed.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                pass  # Leave as-is, DB will store the raw string

    return clean


class ToolExecutor:
    def __init__(self, db: Database, memory: Memory,
                 weather: WeatherClient | None = None):
        self.db = db
        self.memory = memory
        self.weather = weather

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        handler = getattr(self, f"_handle_{name}", None)
        if handler is None:
            logger.warning("Unknown tool called: %s", name)
            return f"Unknown tool: {name}"
        try:
            safe_args = _sanitize_args(name, arguments)
            result = handler(**safe_args)
            # Support both sync and async handlers (weather is async)
            if inspect.isawaitable(result):
                result = await result
            logger.info("Tool %s OK", name)
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", name, type(e).__name__)
            return f"Error executing {name}: {type(e).__name__}"

    async def _handle_save_memory(self, content: str, category: str, importance: int,
                                  related_person: str | None = None,
                                  related_date: str | None = None) -> str:
        mid = await self.memory.save(content, category, importance,
                                     related_person, related_date)
        return f"Saved memory {mid}: {content[:60]}..."

    async def _handle_search_memory(self, query: str,
                                    category: str | None = None) -> str:
        results = await self.memory.search(query, top_k=5, category=category)
        if not results:
            return "Nothing found in memory."
        lines = []
        for r in results:
            lines.append(f"- [{r['category']}] {r['content']} (score={r['final_score']:.2f})")
        return "\n".join(lines)

    def _handle_create_event(self, title: str, start_at: str,
                             end_at: str | None = None,
                             location: str | None = None,
                             related_person: str | None = None,
                             description: str | None = None) -> str:
        event = self.db.create_event(title, start_at, end_at, location,
                                     description, related_person)
        return f"Event created: {title} at {start_at}"

    def _handle_get_events(self, date_from: str,
                           date_to: str | None = None) -> str:
        events = self.db.get_events(date_from, date_to)
        if not events:
            return f"No events for {date_from}"
        lines = []
        for e in events:
            time_str = e["start_at"][11:16] if len(e["start_at"]) > 10 else ""
            lines.append(f"- {time_str} {e['title']}")
        return "\n".join(lines)

    def _handle_create_reminder(self, text: str, trigger_at: str,
                                priority: str = "medium",
                                recurrence: str | None = None) -> str:
        if recurrence and recurrence not in ("daily", "weekly", "monthly"):
            recurrence = None
        self.db.create_reminder(text, trigger_at, priority, recurrence)
        rec_str = f" (повтор: {recurrence})" if recurrence else ""
        return f"Reminder set: {text} at {trigger_at}{rec_str}"

    def _handle_update_contact(self, name: str, relation: str | None = None,
                               birthday: str | None = None,
                               notes: str | None = None) -> str:
        contact = self.db.upsert_contact(name, relation, birthday, notes)
        return f"Contact updated: {name}"

    def _handle_get_contacts(self, query: str) -> str:
        contacts = self.db.get_contacts(query)
        if not contacts:
            return f"No contacts matching '{query}'"
        lines = []
        for c in contacts:
            parts = [c["name"]]
            if c.get("relation"):
                parts.append(f"({c['relation']})")
            if c.get("birthday"):
                parts.append(f"ДР: {c['birthday']}")
            if c.get("notes"):
                parts.append(f"— {c['notes'][:80]}")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    async def _handle_get_weather(self, date: str | None = None,
                                  days: int | None = None) -> str:
        if not self.weather:
            return "Weather not configured."
        forecast = await self.weather.get_forecast(days=days or 1)
        return self.weather.format_daily(forecast)

    def _handle_log_habit(self, habit_name: str, done: bool,
                          date: str | None = None) -> str:
        result = self.db.log_habit(habit_name, done, date)
        status = "done" if done else "skipped"
        return f"Habit '{habit_name}' {status} for {result['date']}"

    def _handle_track_decision(self, description: str,
                               context: str | None = None,
                               follow_up_days: int = 30) -> str:
        # Check for similar past decisions
        words = description.split() if description and description.strip() else []
        past = self.db.get_past_decisions(words[0] if words else "", limit=3)
        decision = self.db.create_decision(description, context, follow_up_days)

        result = f"Decision tracked: {description}. Follow-up in {follow_up_days} days."
        if past:
            result += "\n\nSimilar past decisions:"
            for p in past:
                sentiment = p.get('outcome_sentiment', '?')
                result += f"\n- {p['description'][:80]} → {(p.get('outcome') or 'no outcome')[:80]} ({sentiment})"
        return result

    def _handle_set_daily_goal(self, title: str, description: str | None = None,
                               priority: str = "medium") -> str:
        goal = self.db.create_daily_goal(title, description, priority)
        prio_ru = {"high": "высокий", "medium": "средний", "low": "низкий"}.get(priority, priority)
        return f"Goal set: {title} (приоритет: {prio_ru})"

    def _handle_complete_daily_goal(self, goal_hint: str, status: str = "completed",
                                    reflection: str | None = None) -> str:
        if not goal_hint or not goal_hint.strip():
            return "complete_daily_goal requires a non-empty goal_hint"
        result = self.db.complete_daily_goal_by_hint(goal_hint.strip(), status, reflection)
        if not result:
            return f"No pending goal found matching '{goal_hint}' for today"
        status_ru = {"completed": "выполнена", "skipped": "пропущена", "partial": "частично"}.get(status, status)
        return f"Goal '{result['title']}' marked as {status_ru}"

    def _handle_resolve_decision(self, description_hint: str, outcome: str,
                                 sentiment: str = "neutral") -> str:
        if not description_hint or not description_hint.strip():
            return "resolve_decision requires a non-empty description_hint"
        if not outcome or not outcome.strip():
            return "resolve_decision requires a non-empty outcome"
        if sentiment not in (Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE):
            sentiment = Sentiment.NEUTRAL
        resolved = self.db.resolve_decision_by_hint(
            description_hint.strip(), outcome, sentiment,
        )
        if not resolved:
            return f"No pending decision found matching '{description_hint}'"
        return f"Resolved decision: {resolved['description'][:80]} → {outcome[:80]}"
