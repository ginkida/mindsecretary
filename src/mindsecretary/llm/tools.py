from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from datetime import datetime, timedelta, timezone as _dt_timezone
from typing import Any
from zoneinfo import ZoneInfo

_tz_utc = _dt_timezone.utc

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
        "name": "delete_memory",
        "description": (
            "Удалить (soft-delete, восстанавливается через /undo) один факт "
            "из памяти. Вызывай когда {name} говорит «забудь что X», "
            "«это уже не актуально», «удали факт про Y». Если совпадает "
            "несколько записей — удаление НЕ выполняется, будут возвращены "
            "найденные и количество. Уточни hint и вызови снова. Не "
            "используй для исправления факта — для этого update_memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_hint": {
                    "type": "string",
                    "description": (
                        "Подстрока из содержимого факта "
                        "(напр. 'Yandex', 'шахматы')."
                    ),
                },
            },
            "required": ["text_hint"],
        },
    },
    {
        "name": "update_memory",
        "description": (
            "Заменить содержимое одного факта в памяти. Вызывай когда "
            "пользователь корректирует ранее сохранённое: «больше не работаю "
            "в Yandex, теперь в Сбере», «Петя теперь муж Маши, а не друг». "
            "Если совпадает несколько записей по hint, обновление НЕ "
            "выполняется — будет возвращён список найденных и количество, "
            "уточни hint и вызови снова. Не используй для добавления новых "
            "фактов — для этого save_memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_hint": {
                    "type": "string",
                    "description": (
                        "Подстрока из старого содержимого факта "
                        "(напр. 'Yandex', 'друг Маши')."
                    ),
                },
                "new_content": {
                    "type": "string",
                    "description": "Новое содержимое факта целиком.",
                },
            },
            "required": ["text_hint", "new_content"],
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
        "name": "get_recent_memories",
        "description": (
            "Показать недавние воспоминания с источником, уверенностью и временем. "
            "Используй если пользователь спрашивает что ты помнишь или почему."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "contact", "health", "work", "personal",
                        "promise", "preference", "location", "learning",
                        "emotional",
                    ],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
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
        "name": "search_events",
        "description": (
            "Найти будущие события по подстроке — title / description / "
            "location / related_person, case-insensitive, Cyrillic-aware. "
            "Вызывай когда {name} спрашивает «когда у меня встреча с "
            "Машей?», «есть ли что в зале на этой неделе?», «куда мы "
            "идём с Олегом?». Работает БЕЗ необходимости угадывать "
            "диапазон дат — get_events нужен когда уже известны даты, "
            "search_events когда известно ключевое слово."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Подстрока для поиска."},
                "days_ahead": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Окно вперёд в днях (по умолчанию 30).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Сколько матчей (по умолчанию 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "cancel_event",
        "description": (
            "Удалить будущее событие из календаря. Вызывай когда "
            "пользователь говорит «отмени встречу с Машей в пятницу», "
            "«я не пойду», «убери из календаря». Ищет по подстроке в "
            "title или description (case-insensitive, Cyrillic-aware) "
            "среди будущих событий. Если совпадает несколько — удалит "
            "ближайшее по времени и сообщит сколько ещё похожих."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_hint": {
                    "type": "string",
                    "description": "Подстрока из title/description (напр. 'Маша', 'дантист').",
                },
            },
            "required": ["text_hint"],
        },
    },
    {
        "name": "reschedule_event",
        "description": (
            "Перенести будущее событие на новое время. Вызывай когда "
            "пользователь говорит «перенеси встречу на 17:00», «сдвинь "
            "ужин на завтра», «давай не в пятницу, а в субботу». "
            "Ищет по подстроке в title/description среди будущих "
            "событий. Если совпадает несколько — перенесёт ближайшее "
            "и сообщит сколько ещё похожих. end_at можно не указывать "
            "если длительность не меняется."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_hint": {"type": "string"},
                "new_start_at": {"type": "string", "description": "YYYY-MM-DDTHH:MM"},
                "new_end_at": {"type": "string", "description": "Опционально"},
            },
            "required": ["text_hint", "new_start_at"],
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
        "name": "reschedule_reminder",
        "description": (
            "Перенести pending-напоминание на новое время. Вызывай когда "
            "пользователь говорит «перенеси на завтра», «давай в 18:00 "
            "вместо 14:00». Поиск по подстроке в тексте — кириллица "
            "учитывается. Если совпадает несколько, переносится ближайшее "
            "по времени; в ответе будет указано сколько ещё осталось похожих. "
            "Для recurring серии переносится только следующая instance — "
            "серия продолжается от нового времени. Не используй для уже "
            "отправленных (status='sent') — создай новое."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_hint": {
                    "type": "string",
                    "description": (
                        "Ключевое слово или фраза из текста напоминания "
                        "(напр. 'дантист', 'Маше позвонить')."
                    ),
                },
                "new_trigger_at": {
                    "type": "string",
                    "description": "Новое время в формате YYYY-MM-DDTHH:MM",
                },
            },
            "required": ["text_hint", "new_trigger_at"],
        },
    },
    {
        "name": "cancel_reminder",
        "description": (
            "Отменить отложенное напоминание (status pending → cancelled). "
            "Вызывай когда пользователь говорит «отмени напоминание про X», "
            "«не надо больше про Y». Поиск по подстроке в тексте — кириллица "
            "учитывается. Если совпадает несколько, отменяется ближайшее по "
            "времени; в ответе будет указано сколько ещё осталось похожих, "
            "чтобы ты мог уточнить или отменить остальные. Для recurring "
            "напоминаний это останавливает серию — следующее не создаётся."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_hint": {
                    "type": "string",
                    "description": (
                        "Ключевое слово или фраза из текста напоминания "
                        "(напр. 'дантист', 'Маше позвонить')."
                    ),
                },
            },
            "required": ["text_hint"],
        },
    },
    {
        "name": "get_reminders",
        "description": (
            "Перечислить отложенные напоминания (status=pending). Используй, "
            "когда пользователь спрашивает «что у меня в напоминаниях», "
            "«какие на этой неделе напоминания» или нужно сослаться на ранее "
            "поставленное напоминание перед его изменением. Включает "
            "просроченные. days_ahead ограничивает окно вперёд (по умолчанию "
            "все). Не путай с get_open_loops — там агрегат по всем хвостам."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "description": (
                        "Сколько дней вперёд от сегодня показывать (опционально). "
                        "Просроченные пендинг-напоминания включаются всегда."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Максимум записей (по умолчанию 20).",
                },
            },
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
        "name": "get_open_loops",
        "description": (
            "Получить незакрытые хвосты: просроченные напоминания, цели, решения, "
            "ближайшие события. Используй если пользователь спрашивает что висит или на контроле."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {"type": "integer", "minimum": 1, "maximum": 7},
            },
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
        "name": "get_habits",
        "description": (
            "Показать привычки {name} с текущим streak, выполнением за "
            "последние 7 дней и датой последнего выполнения. Вызывай когда "
            "пользователь спрашивает про свои привычки: «сколько уже "
            "бегаю?», «когда последний раз качался?», «какие у меня "
            "привычки?». Без аргументов возвращает все."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
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
        "name": "get_decisions",
        "description": (
            "Показать активные (pending) решения {name}: что обдумывает, "
            "что в процессе. Вызывай когда пользователь спрашивает «что я "
            "там обдумываю?», «какие у меня открытые вопросы?», «что "
            "решал?». Возвращает описание + контекст + дата создания. "
            "Решения, которые уже resolved, не показываются."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Сколько последних показать (по умолчанию 10).",
                },
            },
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
    {
        "name": "search_conversations",
        "description": (
            "Поиск по старым сообщениям диалога (и твоим, и пользователя) "
            "по ключевому слову. Вызывай когда пользователь ссылается на "
            "разговор старше последних ~20 ходов, которые ты видишь в "
            "диалоге напрямую: «что я тебе говорил про X», «мы обсуждали Y», "
            "«ты писал мне про Z». Ищет подстроку в content, не семантика."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Ключевое слово или короткая фраза",
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Сколько дней назад смотреть (по умолчанию 30)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Максимум результатов (по умолчанию 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "set_ephemeral_state",
        "description": (
            "Запомнить текущее состояние юзера с авто-истечением. Вызывай, "
            "когда он упоминает свою локацию ('на работе', 'дома', 'еду'), "
            "здоровье ('болею', 'устал'), занятость ('свободен до 18', 'весь "
            "день в meetings'), энергию или текущее занятие. TTL выбирай по "
            "смыслу: 'на работе' → 8-10ч, 'болею' → 24ч, 'в отпуске до 15' → "
            "до этой даты в часах. INSERT OR REPLACE — та же key перезапишет. "
            "НЕ используй для долгих фактов — для них save_memory. Макс TTL 72ч."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": ["location", "health", "availability", "energy", "activity"],
                    "description": (
                        "Категория: location=где, health=самочувствие, "
                        "availability=занятость, energy=силы, activity=чем занят сейчас."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "Текущее значение, коротко.",
                },
                "ttl_hours": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 72,
                    "description": "Через сколько часов это перестанет быть актуальным.",
                },
            },
            "required": ["key", "value", "ttl_hours"],
        },
    },
]


MAX_STR_LEN = 5000
VALID_CATEGORIES = {
    "contact", "health", "work", "personal",
    "promise", "preference", "location", "learning", "emotional",
}
VALID_EPHEMERAL_KEYS = {"location", "health", "availability", "energy", "activity"}
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

    if name == "get_recent_memories":
        clean["limit"] = max(1, min(10, int(clean.get("limit", 5))))

    if name == "get_open_loops":
        clean["days_ahead"] = max(1, min(7, int(clean.get("days_ahead", 2))))

    if name == "cancel_reminder":
        hint = clean.get("text_hint") or ""
        # Tool-side cap independent of MAX_STR_LEN — hint goes into LIKE
        # so a 5000-char value is just wasted matcher work.
        clean["text_hint"] = str(hint)[:200]

    if name == "update_memory":
        hint = clean.get("text_hint") or ""
        clean["text_hint"] = str(hint)[:200]
        # new_content gets the global MAX_STR_LEN cap from the generic
        # truncate above; nothing extra needed here.

    if name == "delete_memory":
        hint = clean.get("text_hint") or ""
        clean["text_hint"] = str(hint)[:200]

    if name == "reschedule_reminder":
        hint = clean.get("text_hint") or ""
        clean["text_hint"] = str(hint)[:200]

    if name == "cancel_event":
        hint = clean.get("text_hint") or ""
        clean["text_hint"] = str(hint)[:200]

    if name == "reschedule_event":
        hint = clean.get("text_hint") or ""
        clean["text_hint"] = str(hint)[:200]

    if name == "get_reminders":
        if clean.get("days_ahead") is not None:
            clean["days_ahead"] = max(1, min(365, int(clean["days_ahead"])))
        clean["limit"] = max(1, min(50, int(clean.get("limit", 20))))

    if name == "search_conversations":
        clean["days"] = max(1, min(365, int(clean.get("days", 30))))
        clean["limit"] = max(1, min(30, int(clean.get("limit", 10))))

    if name == "get_decisions":
        clean["limit"] = max(1, min(30, int(clean.get("limit", 10))))

    if name == "search_events":
        if clean.get("days_ahead") is not None:
            clean["days_ahead"] = max(1, min(365, int(clean["days_ahead"])))
        else:
            clean["days_ahead"] = 30
        clean["limit"] = max(1, min(30, int(clean.get("limit", 10))))

    if name == "set_ephemeral_state":
        key = clean.get("key", "")
        if key not in VALID_EPHEMERAL_KEYS:
            clean["key"] = "activity"  # safe default
        # value length cap tighter than global MAX_STR_LEN — state lines go
        # into the system prompt of every message.
        val = clean.get("value") or ""
        clean["value"] = str(val)[:200] or "активно"
        ttl = float(clean.get("ttl_hours", 8.0))
        clean["ttl_hours"] = max(0.5, min(72.0, ttl))

    if name in ("create_event", "create_reminder"):
        priority = clean.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            clean["priority"] = "medium"

    # Validate datetime fields — try to parse, normalize to space-separated format
    _DT_FIELDS = {
        "create_event": ("start_at", "end_at"),
        "create_reminder": ("trigger_at",),
        "reschedule_reminder": ("new_trigger_at",),
        "reschedule_event": ("new_start_at", "new_end_at"),
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

    @staticmethod
    def _default_memory_confidence(request_context: dict[str, Any] | None) -> float:
        source_type = (request_context or {}).get("source_type")
        return {
            "text": 0.95,
            "forward": 0.9,
            "voice": 0.82,
            "photo": 0.78,
        }.get(source_type, 0.9)

    @staticmethod
    def _format_memory_source(source_type: str | None, source_ref: str | None) -> str:
        label = {
            "text": "text",
            "voice": "voice",
            "photo": "photo",
            "forward": "forward",
        }.get(source_type or "", source_type or "unknown")
        if source_ref:
            return f"{label}#{source_ref[:8]}"
        return label

    @staticmethod
    def _format_open_loops(snapshot: dict) -> str:
        counts = snapshot.get("counts", {})
        lines = [
            f"Open loops: overdue_reminders={counts.get('overdue_reminders', 0)}, "
            f"today_reminders={counts.get('due_today_reminders', 0)}, "
            f"upcoming_events={counts.get('upcoming_events', 0)}, "
            f"pending_goals={counts.get('pending_goals', 0)}, "
            f"due_decisions={counts.get('due_decisions', 0)}",
        ]

        if snapshot.get("overdue_reminders"):
            lines.append("Overdue reminders:")
            lines.extend(
                f"- {r['trigger_at']}: {r['text']}"
                for r in snapshot["overdue_reminders"][:3]
            )
        if snapshot.get("upcoming_events"):
            lines.append("Upcoming events:")
            lines.extend(
                f"- {e['start_at']}: {e['title']}"
                for e in snapshot["upcoming_events"][:3]
            )
        if snapshot.get("pending_goals"):
            lines.append("Pending goals today:")
            lines.extend(
                f"- {g['title']} ({g['priority']})"
                for g in snapshot["pending_goals"][:3]
            )
        if snapshot.get("due_decisions"):
            lines.append("Decision follow-ups due:")
            lines.extend(
                f"- {d['description'][:120]}"
                for d in snapshot["due_decisions"][:3]
            )

        return "\n".join(lines)

    async def execute(self, name: str, arguments: dict[str, Any],
                      request_context: dict[str, Any] | None = None) -> str:
        handler = getattr(self, f"_handle_{name}", None)
        if handler is None:
            logger.warning("Unknown tool called: %s", name)
            return f"Unknown tool: {name}"
        # Wall-clock timing covers async waits too — tools like search_memory
        # spend most of their time inside Voyage embed calls, and only
        # `time.monotonic()` (not CPU time) reflects what the user sees.
        # Logged on both branches so ops can correlate slow LLM rounds with
        # the offending tool without adding tracing infrastructure.
        start = time.monotonic()
        try:
            safe_args = _sanitize_args(name, arguments)
            if "request_context" in inspect.signature(handler).parameters:
                result = handler(request_context=request_context, **safe_args)
            else:
                result = handler(**safe_args)
            # Support both sync and async handlers (weather is async)
            if inspect.isawaitable(result):
                result = await result
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info("Tool %s OK (%.0fms)", name, elapsed_ms)
            return result
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "Tool %s failed (%.0fms): %s",
                name, elapsed_ms, type(e).__name__,
            )
            return f"Error executing {name}: {type(e).__name__}"

    async def _handle_save_memory(self, content: str, category: str, importance: int,
                                  related_person: str | None = None,
                                  related_date: str | None = None,
                                  request_context: dict[str, Any] | None = None) -> str:
        source_type = (request_context or {}).get("source_type")
        source_ref = (request_context or {}).get("source_ref")
        mid = await self.memory.save(content, category, importance,
                                     related_person, related_date,
                                     source_type=source_type,
                                     source_ref=source_ref,
                                     confidence=self._default_memory_confidence(request_context))
        return f"Saved memory {mid}: {content[:60]}..."

    async def _handle_delete_memory(self, text_hint: str) -> str:
        result = await self.memory.delete_by_hint(text_hint)
        status = result.get("status")
        if status == "invalid":
            return "delete_memory requires non-empty text_hint"
        if status == "not_found":
            return f"Не нашёл в памяти ничего по '{text_hint[:80]}'"
        if status == "ambiguous":
            count = result.get("count", 0)
            samples = result.get("samples") or []
            sample_lines = "\n".join(
                f"  - [{s.get('id')}] {s.get('content', '')}"
                for s in samples
            )
            return (
                f"По '{text_hint[:80]}' нашлось {count} записей — слишком "
                f"много, удаление не выполнено. Уточни hint и вызови снова. "
                f"Примеры найденного:\n{sample_lines}"
            )
        if status == "ok":
            mem = result.get("memory") or {}
            mem_id = mem.get("id", "?")
            content = (mem.get("content") or "")[:120]
            cat = mem.get("category", "?")
            return f"Удалено [{mem_id}] [{cat}]: {content} (можно восстановить через /undo)"
        return f"delete_memory unknown status: {status}"

    async def _handle_update_memory(self, text_hint: str,
                                    new_content: str) -> str:
        result = await self.memory.update_by_hint(text_hint, new_content)
        status = result.get("status")
        if status == "invalid":
            return "update_memory requires non-empty text_hint and new_content"
        if status == "not_found":
            return f"Не нашёл в памяти ничего по '{text_hint[:80]}'"
        if status == "ambiguous":
            count = result.get("count", 0)
            samples = result.get("samples") or []
            sample_lines = "\n".join(
                f"  - [{s.get('id')}] {s.get('content', '')}"
                for s in samples
            )
            return (
                f"По '{text_hint[:80]}' нашлось {count} записей — слишком "
                f"много, обновление не выполнено. Уточни hint и вызови снова. "
                f"Примеры найденного:\n{sample_lines}"
            )
        if status == "embed_failed":
            return (
                "Voyage embed недоступен — память не обновлена, "
                "старая запись осталась. Попробуй позже."
            )
        if status == "ok":
            mem = result.get("memory") or {}
            mem_id = mem.get("id", "?")
            content = (mem.get("content") or "")[:120]
            cat = mem.get("category", "?")
            return f"Обновлено [{mem_id}] [{cat}]: {content}"
        return f"update_memory unknown status: {status}"

    async def _handle_search_memory(self, query: str,
                                    category: str | None = None) -> str:
        results = await self.memory.search(query, top_k=5, category=category)
        if not results:
            return "Nothing found in memory."
        lines = []
        for r in results:
            lines.append(
                f"- [{r['category']}] {r['content']} "
                f"(score={r['final_score']:.2f}, reason={r['match_reason']}, "
                f"source={self._format_memory_source(r.get('source_type'), r.get('source_ref'))}, "
                f"confidence={r.get('confidence', 0.0):.2f})"
            )
        return "\n".join(lines)

    def _handle_get_recent_memories(self, category: str | None = None,
                                    limit: int = 5) -> str:
        memories = self.memory.list_recent(limit=limit, category=category)
        if not memories:
            return "No recent memories."
        lines = []
        for m in memories:
            lines.append(
                f"- [{m['category']}] {m['content']} "
                f"(source={self._format_memory_source(m.get('source_type'), m.get('source_ref'))}, "
                f"confidence={float(m.get('confidence') or 0.0):.2f}, created={m.get('created_at', '')})"
            )
        return "\n".join(lines)

    def _handle_create_event(self, title: str, start_at: str,
                             end_at: str | None = None,
                             location: str | None = None,
                             related_person: str | None = None,
                             description: str | None = None) -> str:
        event = self.db.create_event(title, start_at, end_at, location,
                                     description, related_person)
        return f"Event created: {title} at {start_at}"

    _EVENTS_OUTPUT_CAP = 30

    def _handle_get_events(self, date_from: str,
                           date_to: str | None = None) -> str:
        events = self.db.get_events(date_from, date_to)
        if not events:
            return f"No events for {date_from}"

        # When the query spans multiple days, prefix each line with the
        # date so the LLM (and ultimately the user) can tell which day the
        # event falls on. Single-day queries skip the date — it'd be
        # repeated noise on every line.
        multi_day = bool(date_to and date_to != date_from)

        truncated = events[: self._EVENTS_OUTPUT_CAP]
        lines = []
        for e in truncated:
            start = e.get("start_at") or ""
            date_part = start[:10]
            time_part = start[11:16] if len(start) >= 16 else ""
            end_part = ""
            end_at = e.get("end_at") or ""
            # Render "HH:MM-HH:MM" only if end_at is on the same day —
            # otherwise the dash hides a day boundary and reads wrong.
            if end_at and len(end_at) >= 16 and end_at[:10] == date_part:
                end_part = f"-{end_at[11:16]}"

            prefix = f"{date_part} {time_part}{end_part}" if multi_day else f"{time_part}{end_part}"
            line = f"- {prefix} {e.get('title', '')}".rstrip()

            extras = []
            loc = (e.get("location") or "").strip()
            if loc:
                extras.append(f"📍 {loc[:80]}")
            person = (e.get("related_person") or "").strip()
            title_lower = (e.get("title") or "").lower()
            # Avoid duplicating the person when they're already in the title
            # — see _format_event_alert in monitor.py for the same heuristic.
            if person and person.lower()[:3] not in title_lower:
                extras.append(f"👤 {person[:80]}")
            if extras:
                line += " | " + " ".join(extras)
            lines.append(line)

        if len(events) > len(truncated):
            lines.append(
                f"... и ещё {len(events) - len(truncated)} (сузь диапазон дат)"
            )
        return "\n".join(lines)

    def _handle_search_events(self, query: str, days_ahead: int = 30,
                              limit: int = 10) -> str:
        rows = self.db.search_events(query, days_ahead=days_ahead, limit=limit)
        if not rows:
            return f"Не нашёл будущих событий по '{query}' в ближайшие {days_ahead} дн."
        lines = []
        for e in rows:
            # Show full date+time; user often asks about events farther out
            # than today so HH:MM alone (as in get_events) hides the day.
            start = (e.get("start_at") or "?")[:16]
            title = (e.get("title") or "")[:120]
            line = f"- {start} {title}"
            extras = []
            loc = (e.get("location") or "").strip()
            if loc:
                extras.append(f"📍 {loc[:80]}")
            person = (e.get("related_person") or "").strip()
            if person and person.lower()[:3] not in title.lower():
                extras.append(f"👤 {person[:80]}")
            if extras:
                line += " | " + " ".join(extras)
            lines.append(line)
        return "\n".join(lines)

    def _handle_cancel_event(self, text_hint: str) -> str:
        if not text_hint or not text_hint.strip():
            return "cancel_event requires a non-empty text_hint"
        # Count first so we can disclose ambiguity in the response —
        # single-threaded SQLite, no race with the cancel below.
        total = self.db.count_future_events_matching(text_hint)
        cancelled = self.db.cancel_event_by_hint(text_hint)
        if not cancelled:
            return f"Не нашёл будущих событий по '{text_hint[:80]}'"
        title = (cancelled.get("title") or "")[:120]
        start = (cancelled.get("start_at") or "?")[:16]
        msg = f"Отменено: {title} ({start})"
        remaining = total - 1
        if remaining > 0:
            msg += f". Похожих ещё {remaining} — уточни если нужно отменить и их."
        return msg

    def _handle_reschedule_event(self, text_hint: str, new_start_at: str,
                                 new_end_at: str | None = None) -> str:
        if not text_hint or not text_hint.strip():
            return "reschedule_event requires a non-empty text_hint"
        if not new_start_at or not new_start_at.strip():
            return "reschedule_event requires new_start_at"
        total = self.db.count_future_events_matching(text_hint)
        updated = self.db.reschedule_event_by_hint(text_hint, new_start_at, new_end_at)
        if not updated:
            return f"Не нашёл будущих событий по '{text_hint[:80]}'"
        title = (updated.get("title") or "")[:120]
        msg = f"Перенесено: {title} → {new_start_at[:16]}"
        remaining = total - 1
        if remaining > 0:
            msg += f". Похожих ещё {remaining} — уточни если нужно перенести и их."
        return msg

    def _handle_create_reminder(self, text: str, trigger_at: str,
                                priority: str = "medium",
                                recurrence: str | None = None) -> str:
        if recurrence and recurrence not in ("daily", "weekly", "monthly"):
            recurrence = None
        self.db.create_reminder(text, trigger_at, priority, recurrence)
        rec_str = f" (повтор: {recurrence})" if recurrence else ""
        return f"Reminder set: {text} at {trigger_at}{rec_str}"

    def _handle_reschedule_reminder(self, text_hint: str,
                                    new_trigger_at: str) -> str:
        if not text_hint or not text_hint.strip():
            return "reschedule_reminder requires a non-empty text_hint"
        if not new_trigger_at or not new_trigger_at.strip():
            return "reschedule_reminder requires new_trigger_at"
        # Count first so we can disclose ambiguity in the response.
        total = self.db.count_pending_reminders_matching(text_hint)
        updated = self.db.reschedule_reminder_by_hint(text_hint, new_trigger_at)
        if not updated:
            return f"Не нашёл pending-напоминаний по '{text_hint[:80]}'"
        text = (updated.get("text") or "")[:120]
        msg = f"Перенесено: {text} → {new_trigger_at[:16]}"
        remaining = total - 1
        if remaining > 0:
            msg += f". Похожих ещё {remaining} — уточни если нужно перенести и их."
        return msg

    def _handle_cancel_reminder(self, text_hint: str) -> str:
        if not text_hint or not text_hint.strip():
            return "cancel_reminder requires a non-empty text_hint"
        # Count first so we can disclose ambiguity in the response —
        # single-threaded SQLite, no race with the cancel below.
        total = self.db.count_pending_reminders_matching(text_hint)
        cancelled = self.db.cancel_reminder_by_hint(text_hint)
        if not cancelled:
            return f"Не нашёл pending-напоминаний по '{text_hint[:80]}'"
        text = (cancelled.get("text") or "")[:120]
        trigger = (cancelled.get("trigger_at") or "?")[:16]
        msg = f"Отменено: {text} ({trigger})"
        remaining = total - 1
        if remaining > 0:
            msg += f". Похожих ещё {remaining} — уточни если нужно отменить и их."
        return msg

    def _handle_get_reminders(self, days_ahead: int | None = None,
                              limit: int = 20) -> str:
        rows = self.db.get_pending_reminders()
        if not rows:
            return "Нет отложенных напоминаний."

        # `reminders.trigger_at` is profile-local naive (per CLAUDE.md TZ
        # convention) — same source as db.local_now_naive(), so compare
        # directly without TZ conversion.
        now_local = self.db.local_now_naive()
        upper_bound: datetime | None = None
        if days_ahead is not None:
            upper_bound = now_local + timedelta(days=days_ahead)

        filtered: list[dict] = []
        for r in rows:
            trigger_str = r.get("trigger_at") or ""
            # Always keep overdue (trigger <= now); window only restricts
            # future ones, so user still sees what's already late.
            if upper_bound is not None:
                try:
                    trig = datetime.fromisoformat(trigger_str.replace(" ", "T"))
                except (ValueError, TypeError):
                    filtered.append(r)  # unparseable — keep, don't silently drop
                    continue
                if trig > now_local and trig > upper_bound:
                    continue
            filtered.append(r)

        if not filtered:
            return f"Нет отложенных напоминаний на ближайшие {days_ahead} дн."

        lines: list[str] = []
        for r in filtered[:limit]:
            ts = (r.get("trigger_at") or "?")[:16]
            prio = r.get("priority") or "medium"
            rec = r.get("recurrence")
            rec_tag = f" ({rec})" if rec else ""
            text = (r.get("text") or "")[:200]
            lines.append(f"- {ts} [{prio}]{rec_tag}: {text}")

        if len(filtered) > limit:
            lines.append(f"... и ещё {len(filtered) - limit} (увеличь limit чтобы увидеть)")
        return "\n".join(lines)

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

    def _handle_get_open_loops(self, days_ahead: int = 2) -> str:
        snapshot = self.db.get_open_loops(days_ahead=days_ahead, limit_per_section=5)
        return self._format_open_loops(snapshot)

    _SEARCH_KIND_LABELS = {
        "morning_briefing": "брифинг",
        "evening_summary": "вечер",
        "diary": "дневник",
        "weekly_review": "неделя",
        "smart_question": "вопрос",
        "open_loops_nudge": "контроль",
        "decision_followup": "решение",
        "birthday_alert": "день рождения",
        "weather_alert": "погода",
        "reminder": "напоминание",
    }

    @staticmethod
    def _ts_utc_to_local_str(ts: str, tz_name: str | None) -> str:
        """DB timestamps are UTC (SQLite's `datetime('now')`). Convert to
        profile timezone so Claude reads times that match the user's clock
        instead of a silent UTC offset (e.g. '05:30' vs actual 10:30)."""
        if not ts or not tz_name:
            return ts[:16] if ts else "?"
        try:
            utc_naive = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            local = utc_naive.replace(tzinfo=_tz_utc).astimezone(ZoneInfo(tz_name))
            return local.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, KeyError):
            return ts[:16]

    def _handle_search_conversations(self, query: str, days: int = 30,
                                     limit: int = 10) -> str:
        rows = self.db.search_past_conversations(query, days=days, limit=limit)
        if not rows:
            return f"Ничего не нашёл по '{query}' за последние {days} дн."
        tz_name = getattr(self.db, "_timezone", None)
        lines = []
        for r in rows:
            ts = self._ts_utc_to_local_str(r.get("timestamp") or "", tz_name)
            if r.get("direction") == "in":
                who = "Ты"
            elif r.get("message_type") == "notification":
                kind = None
                meta_raw = r.get("metadata")
                if meta_raw:
                    try:
                        kind = json.loads(meta_raw).get("kind")
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Malformed interactions.metadata JSON; kind label will fall back",
                        )
                who = f"[{self._SEARCH_KIND_LABELS.get(kind, 'уведомление')}]"
            else:
                who = "Бот"
            raw_content = r.get("content") or ""
            content = raw_content[:300]
            # Signal truncation so Claude knows the quote is partial.
            if len(raw_content) > 300:
                content += "…"
            lines.append(f"- {ts} {who}: {content}")
        return "\n".join(lines)

    def _handle_set_ephemeral_state(self, key: str, value: str,
                                    ttl_hours: float) -> str:
        self.db.set_ephemeral_state(key, value, float(ttl_hours))
        return f"Ephemeral state saved: {key}={value} (TTL {ttl_hours}h)"

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

    def _handle_get_habits(self) -> str:
        stats = self.db.get_habit_stats()
        if not stats:
            return "Привычек пока нет."
        lines = []
        for s in stats:
            today_mark = "✓ сегодня" if s["logged_today"] else "ещё не сегодня"
            last = s.get("last_done_date") or "никогда"
            lines.append(
                f"- {s['name']}: streak {s['streak']} дн., "
                f"за 7 дней {s['week_done']}/7 ({s['week_rate']}%), "
                f"последний раз: {last}, {today_mark}"
            )
        return "\n".join(lines)

    def _handle_get_decisions(self, limit: int = 10) -> str:
        rows = self.db.get_pending_decisions(limit=limit)
        if not rows:
            return "Нет активных решений в процессе."
        # Format: created_at[:10] is the YYYY-MM-DD prefix; works for both
        # SQLite-written UTC (datetime('now')) and any future local-write
        # since we only show the date, not the clock.
        lines = []
        for r in rows:
            created = (r.get("created_at") or "?")[:10]
            desc = (r.get("description") or "")[:200]
            ctx = (r.get("context") or "").strip()
            line = f"- [{created}] {desc}"
            if ctx:
                line += f" — {ctx[:200]}"
            lines.append(line)
        return "\n".join(lines)

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
