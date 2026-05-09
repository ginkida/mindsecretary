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

from ..core import (
    NOTIFICATION_KIND_LABELS,
    fmt_utc_to_local,
    is_person_in_title,
    pluralize_ru,
    tz_now,
)
from ..core.database import Database
from ..core.enums import Priority, Sentiment, Status
from ..core.memory import Memory

# Forms reused across the ambiguity-disclosure messages and diary truncation
# hint — keep in one place so a future translation pass touches one tuple.
_RECORDS_FORMS = ("запись", "записи", "записей")
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
        "name": "update_event",
        "description": (
            "Изменить НЕ-временные поля будущего события "
            "(title/description/location/related_person). Для переноса "
            "времени используй reschedule_event. Вызывай когда {name} "
            "говорит «переименуй встречу», «добавь локацию», «добавь "
            "Сашу как участника». Поля, которые не передал, не меняются. "
            "Передай пустую строку чтобы очистить поле."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_hint": {"type": "string", "description": "Подстрока из title/description."},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "related_person": {"type": "string"},
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
        "description": (
            "Прогноз погоды. Передай date='YYYY-MM-DD' для конкретного "
            "дня (макс +6 дней от сегодня), либо days=N для прогноза на "
            "N дней вперёд. Без аргументов — сегодня."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "days": {
                    "type": "integer", "minimum": 1, "maximum": 7,
                    "description": "Сколько дней вперёд показать.",
                },
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
        "name": "get_daily_goals",
        "description": (
            "Получить цели на конкретный день со статусами "
            "(completed/pending/skipped/partial). Без даты — сегодня. "
            "Вызывай когда {name} спрашивает «что я хотел сделать "
            "сегодня?», «какие цели?», «что я не успел вчера?» "
            "(передай date='YYYY-MM-DD' для прошлых дат)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "YYYY-MM-DD. Без аргумента — сегодня.",
                },
            },
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
        "name": "get_diary_entries",
        "description": (
            "Получить недавние записи дневника {name}. Дневник создаётся "
            "автоматически каждый вечер — содержит резюме дня, настроение и "
            "людей. Вызывай когда {name} спрашивает «что я писал на прошлой "
            "неделе?», «какое было настроение в среду?», «что я отметил "
            "вчера?». days=7 по умолчанию."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 90,
                    "description": "Сколько дней назад смотреть (по умолчанию 7).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Максимум записей в выводе (по умолчанию 5).",
                },
            },
        },
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


def _safe_int(val: Any, default: int) -> int:
    """Coerce LLM-provided value to int with a default fallback.

    Pre-fix `int(clean.get("limit", 5))` raised ValueError when the LLM
    hallucinated "five" or "all" instead of a real integer. Sanitize_args
    propagated the exception, tool_executor caught it as a generic
    "Error executing X: ValueError" and the LLM had no path to recover.
    With the fallback the call still goes through with sane defaults
    and the user gets a real answer.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        # bool is a subclass of int — int(True)=1 is meaningless as a
        # "limit" or "days_ahead". Treat as garbage and fall back.
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def _strip_or_none(val: Any) -> str | None:
    """Trim whitespace; collapse empty/None to None.

    Free-text fields (related_person, location, description, etc.) used
    to land in the DB with leading/trailing whitespace from LLM-side
    formatting. That broke is_person_in_title's stem match ("  м" not
    in "встреча с машей") and surfaced as "📍   кафе  " in user-facing
    output. Bouncing empty strings to None matches the DB nullable
    contract too (so Brain.section_events skips empty fields cleanly).
    """
    if val is None:
        return None
    stripped = str(val).strip()
    return stripped or None


def _safe_float(val: Any, default: float) -> float:
    """Float counterpart to _safe_int — same fallback contract.

    Used for ttl_hours / fractional numeric fields where the LLM
    might hallucinate a non-numeric value ("forever", "пять"). bool
    is bounced to default for the same reason as _safe_int.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


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
        clean["importance"] = max(1, min(10, _safe_int(clean.get("importance"), 5)))

    if name == "search_memory":
        # Drop invalid categories instead of letting them through to the SQL
        # filter — pre-fix, a typo'd or hallucinated category like 'tasks'
        # would silently restrict the search to zero rows and the LLM would
        # tell the user "ничего не нашёл" even when the fact existed in the
        # right category. None falls back to "search across all categories".
        cat = clean.get("category")
        if cat is not None and cat not in VALID_CATEGORIES:
            clean["category"] = None

    if name == "track_decision":
        # Schema enforces 1-365 but defensive clamp catches drift. Negative
        # follow_up_days makes follow_up_at land in the past, so the
        # decision-followup scheduler at 10:00 the next morning would
        # prompt the user about a decision they just created — looks like
        # the bot lost track of when "now" is.
        if clean.get("follow_up_days") is not None:
            clean["follow_up_days"] = max(1, min(365, _safe_int(clean["follow_up_days"], 30)))

    if name == "get_recent_memories":
        clean["limit"] = max(1, min(10, _safe_int(clean.get("limit"), 5)))
        # Same guard as search_memory: invalid category drops to None
        # (no filter) instead of letting bogus value through to the SQL
        # WHERE clause and producing a misleading empty result.
        cat = clean.get("category")
        if cat is not None and cat not in VALID_CATEGORIES:
            clean["category"] = None

    if name == "get_open_loops":
        clean["days_ahead"] = max(1, min(7, _safe_int(clean.get("days_ahead"), 2)))

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

    if name == "update_event":
        hint = clean.get("text_hint") or ""
        clean["text_hint"] = str(hint)[:200]

    if name == "get_reminders":
        if clean.get("days_ahead") is not None:
            clean["days_ahead"] = max(1, min(365, _safe_int(clean["days_ahead"], 7)))
        clean["limit"] = max(1, min(50, _safe_int(clean.get("limit"), 20)))

    if name == "search_conversations":
        clean["days"] = max(1, min(365, _safe_int(clean.get("days"), 30)))
        clean["limit"] = max(1, min(30, _safe_int(clean.get("limit"), 10)))

    if name == "get_decisions":
        clean["limit"] = max(1, min(30, _safe_int(clean.get("limit"), 10)))

    if name == "search_events":
        if clean.get("days_ahead") is not None:
            clean["days_ahead"] = max(1, min(365, _safe_int(clean["days_ahead"], 30)))
        else:
            clean["days_ahead"] = 30
        clean["limit"] = max(1, min(30, _safe_int(clean.get("limit"), 10)))

    if name == "get_diary_entries":
        clean["days"] = max(1, min(90, _safe_int(clean.get("days"), 7)))
        clean["limit"] = max(1, min(30, _safe_int(clean.get("limit"), 5)))

    if name == "get_weather":
        # Schema declares max 7 but no min; sanitizer is the safety net.
        # Open-Meteo rejects forecast_days <= 0 and silently caps high
        # values, so the cap below 7 is mostly cosmetic. The min=1 floor
        # is the one that actually prevents an API error.
        if clean.get("days") is not None:
            clean["days"] = max(1, min(7, _safe_int(clean["days"], 1)))

    if name == "set_ephemeral_state":
        key = clean.get("key", "")
        if key not in VALID_EPHEMERAL_KEYS:
            clean["key"] = "activity"  # safe default
        # value length cap tighter than global MAX_STR_LEN — state lines go
        # into the system prompt of every message.
        val = clean.get("value") or ""
        clean["value"] = str(val)[:200] or "активно"
        # _safe_float defends against LLM hallucinating "forever" or
        # "пять" — the prior float() raised ValueError, sanitize_args
        # propagated it, tool_executor returned opaque "Error executing
        # set_ephemeral_state: ValueError". Now falls back to 8h.
        ttl = _safe_float(clean.get("ttl_hours"), 8.0)
        clean["ttl_hours"] = max(0.5, min(72.0, ttl))

    if name in ("create_event", "create_reminder", "set_daily_goal"):
        # All three accept priority and store it on the DB row. Sanitize
        # in the same place so handlers can rely on a valid enum value
        # without each adding its own coercion. Pre-fix set_daily_goal
        # was missing — DB clamped to "medium" but the handler rendered
        # the LLM's raw value to the user (e.g. "(приоритет: urgent)"
        # while the row actually said medium).
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
        # Empty content → Voyage embed of "" produces a near-zero vector,
        # the memory row stores nothing useful, and search would return
        # the empty row whenever cosine drops to chance. Reject up front
        # like every other create handler now does.
        if not content or not content.strip():
            return "save_memory requires a non-empty content"
        content = content.strip()
        # Strip ancillary free-text fields too — LLM occasionally passes
        # "  Маша  " for related_person, which then breaks
        # is_person_in_title's 3-char stem (becomes "  м", never matches
        # the title's lowercase form). Empty after strip → None to
        # match the DB nullable contract.
        related_person = _strip_or_none(related_person)
        related_date = _strip_or_none(related_date)
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
            records_word = pluralize_ru(count, _RECORDS_FORMS)
            return (
                f"По '{text_hint[:80]}' нашлось {count} {records_word} — "
                f"слишком много, удаление не выполнено. Уточни hint и "
                f"вызови снова. Примеры найденного:\n{sample_lines}"
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
            records_word = pluralize_ru(count, _RECORDS_FORMS)
            return (
                f"По '{text_hint[:80]}' нашлось {count} {records_word} — "
                f"слишком много, обновление не выполнено. Уточни hint и "
                f"вызови снова. Примеры найденного:\n{sample_lines}"
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
        # memories.created_at is UTC. Convert to local so the LLM doesn't
        # tell the user "saved at 22:00" for a memory they actually saved
        # at 03:00 local (Asia/Almaty).
        tz_name = getattr(self.db, "_timezone", None)
        lines = []
        for m in memories:
            created_local = self._ts_utc_to_local_str(
                m.get("created_at", ""), tz_name,
            )
            lines.append(
                f"- [{m['category']}] {m['content']} "
                f"(source={self._format_memory_source(m.get('source_type'), m.get('source_ref'))}, "
                f"confidence={float(m.get('confidence') or 0.0):.2f}, created={created_local})"
            )
        return "\n".join(lines)

    @staticmethod
    def _check_iso_datetime(value: str, tool: str, field: str) -> str | None:
        """Return an LLM-facing error string if `value` doesn't parse as
        ISO datetime, else None. Sanitizer attempts the same parse and
        normalizes on success but silently leaves bad input as-is — this
        is the boundary check that surfaces the failure to the caller
        with a format hint, so the row never lands in the DB with
        garbage that date()/datetime() queries silently skip.
        """
        try:
            datetime.fromisoformat(value.replace(" ", "T"))
        except (ValueError, TypeError):
            return (
                f"{tool}: invalid {field} {value!r} — "
                f"use YYYY-MM-DDTHH:MM (e.g. 2026-04-15T14:00)"
            )
        return None

    @staticmethod
    def _check_iso_date(value: str, tool: str, field: str) -> str | None:
        """Date-only counterpart to _check_iso_datetime.

        get_events / get_daily_goals etc. expect YYYY-MM-DD. SQLite's
        date(?) silently returns NULL for unparseable input, the WHERE
        clause then matches nothing, and the LLM tells the user "нет
        событий" while real events sit in the calendar. Strictly check
        the YYYY-MM-DD shape via strptime so the LLM gets a format
        hint instead of empty-look-alike-success.
        """
        if not value or not value.strip():
            return f"{tool}: {field} is required (use YYYY-MM-DD)"
        try:
            datetime.strptime(value.strip(), "%Y-%m-%d")
        except (ValueError, TypeError):
            return (
                f"{tool}: invalid {field} {value!r} — "
                f"use YYYY-MM-DD (e.g. 2026-04-15)"
            )
        return None

    @staticmethod
    def _check_birthday(value: str, tool: str, field: str = "birthday") -> str | None:
        """Birthday columns accept either full date (YYYY-MM-DD) or
        year-less (MM-DD). get_upcoming_birthdays does substr(birthday,
        -5) and matches against the next 7 days' MM-DD list — invalid
        input silently never matches, so a typo'd birthday goes into
        the DB and the user never gets reminded ("я же добавил Машу!"
        but no alert ever fires)."""
        if not value or not value.strip():
            return f"{tool}: {field} is required if provided"
        v = value.strip()
        for fmt in ("%Y-%m-%d", "%m-%d"):
            try:
                datetime.strptime(v, fmt)
                return None
            except ValueError:
                continue
        return (
            f"{tool}: invalid {field} {value!r} — "
            f"use YYYY-MM-DD (e.g. 1990-04-15) or MM-DD (e.g. 04-15)"
        )

    def _handle_create_event(self, title: str, start_at: str,
                             end_at: str | None = None,
                             location: str | None = None,
                             related_person: str | None = None,
                             description: str | None = None) -> str:
        # Defensive validation. Schema is NOT NULL on both, but an LLM
        # passing whitespace would store a row that looks broken in
        # /events output (empty title) or vanishes from date queries
        # (empty start_at — date('') is NULL, never matches a real date).
        if not title or not title.strip():
            return "create_event requires a non-empty title"
        if not start_at or not start_at.strip():
            return "create_event requires a non-empty start_at (YYYY-MM-DDTHH:MM)"
        if (err := self._check_iso_datetime(start_at, "create_event", "start_at")):
            return err
        if end_at and (err := self._check_iso_datetime(end_at, "create_event", "end_at")):
            return err
        # If both provided, end_at must be after start_at. Pre-fix a
        # transposed pair ("с 14:00 до 13:00") was stored as-is and
        # rendered as a nonsense range in get_events output.
        if end_at:
            try:
                start_dt = datetime.fromisoformat(start_at.replace(" ", "T"))
                end_dt = datetime.fromisoformat(end_at.replace(" ", "T"))
                if end_dt <= start_dt:
                    return (
                        f"create_event: end_at ({end_at[:16]}) must be after "
                        f"start_at ({start_at[:16]})"
                    )
            except (ValueError, TypeError):
                pass  # parse failures already caught above
        # Strip free-text fields so " кафе " doesn't render as "📍  кафе "
        # in /events output and so is_person_in_title's stem matcher
        # works on "  Маша  " → "Маша". Empty after strip → None
        # so Brain.section_events / formatters skip the field entirely
        # instead of emitting "(с )".
        event = self.db.create_event(
            title.strip(), start_at, end_at,
            _strip_or_none(location),
            _strip_or_none(description),
            _strip_or_none(related_person),
        )
        return f"Event created: {title} at {start_at}"

    _EVENTS_OUTPUT_CAP = 30

    def _handle_get_events(self, date_from: str,
                           date_to: str | None = None) -> str:
        if (err := self._check_iso_date(date_from, "get_events", "date_from")):
            return err
        if date_to and (err := self._check_iso_date(date_to, "get_events", "date_to")):
            return err
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
            if person and not is_person_in_title(person, e.get("title") or ""):
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
            if person and not is_person_in_title(person, title):
                extras.append(f"👤 {person[:80]}")
            if extras:
                line += " | " + " ".join(extras)
            lines.append(line)
        return "\n".join(lines)

    def _handle_update_event(self, text_hint: str,
                             title: str | None = None,
                             description: str | None = None,
                             location: str | None = None,
                             related_person: str | None = None) -> str:
        if not text_hint or not text_hint.strip():
            return "update_event requires a non-empty text_hint"
        # At least one field must be specified — otherwise the call is
        # a no-op masquerading as a fix.
        if all(v is None for v in (title, description, location, related_person)):
            return (
                "update_event requires at least one of: "
                "title, description, location, related_person"
            )
        # Pre-validate empty title — the schema is NOT NULL so
        # update_event_by_hint silently returns None on whitespace input.
        # Without this guard the response was "Не нашёл будущих событий"
        # even when an event WAS matched, just rejected for empty title.
        # Empty string for the other fields is a documented "clear" sentinel
        # and stays valid.
        if title is not None and not title.strip():
            return (
                "update_event: title cannot be empty (other fields can "
                "be cleared with empty string, but title is required)"
            )
        total = self.db.count_future_events_matching(text_hint)
        updated = self.db.update_event_by_hint(
            text_hint, title=title, description=description,
            location=location, related_person=related_person,
        )
        if not updated:
            return f"Не нашёл будущих событий по '{text_hint[:80]}'"
        new_title = (updated.get("title") or "")[:120]
        start = (updated.get("start_at") or "?")[:16]
        # Brief diff summary so the LLM can echo what changed
        changed = []
        if title is not None:
            changed.append("title")
        if description is not None:
            changed.append("description")
        if location is not None:
            changed.append("location")
        if related_person is not None:
            changed.append("related_person")
        msg = f"Обновлено [{', '.join(changed)}]: {new_title} ({start})"
        remaining = total - 1
        if remaining > 0:
            msg += f". Похожих ещё {remaining} — уточни если нужно изменить и их."
        return msg

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
        if (err := self._check_iso_datetime(new_start_at, "reschedule_event", "new_start_at")):
            return err
        if new_end_at and (err := self._check_iso_datetime(new_end_at, "reschedule_event", "new_end_at")):
            return err
        # If both provided, new_end_at must be after new_start_at — same
        # rationale as iter 49 for create_event. Pre-fix a transposed
        # pair stored a broken row and reschedule_event_by_hint dutifully
        # preserved it.
        if new_end_at:
            try:
                start_dt = datetime.fromisoformat(new_start_at.replace(" ", "T"))
                end_dt = datetime.fromisoformat(new_end_at.replace(" ", "T"))
                if end_dt <= start_dt:
                    return (
                        f"reschedule_event: new_end_at ({new_end_at[:16]}) "
                        f"must be after new_start_at ({new_start_at[:16]})"
                    )
            except (ValueError, TypeError):
                pass
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
        # Defensive validation. Empty text would surface as "⏰ Напоминание:"
        # in monitor.py with nothing after the colon, leaving the user
        # confused about what was supposed to remind. Empty trigger_at
        # makes the row invisible to the time-based reminder check.
        if not text or not text.strip():
            return "create_reminder requires a non-empty text"
        if not trigger_at or not trigger_at.strip():
            return "create_reminder requires a non-empty trigger_at"
        if (err := self._check_iso_datetime(trigger_at, "create_reminder", "trigger_at")):
            return err
        if recurrence and recurrence not in ("daily", "weekly", "monthly"):
            recurrence = None
        self.db.create_reminder(text.strip(), trigger_at, priority, recurrence)
        rec_str = f" (повтор: {recurrence})" if recurrence else ""
        return f"Reminder set: {text} at {trigger_at}{rec_str}"

    def _handle_reschedule_reminder(self, text_hint: str,
                                    new_trigger_at: str) -> str:
        if not text_hint or not text_hint.strip():
            return "reschedule_reminder requires a non-empty text_hint"
        if not new_trigger_at or not new_trigger_at.strip():
            return "reschedule_reminder requires new_trigger_at"
        if (err := self._check_iso_datetime(new_trigger_at, "reschedule_reminder", "new_trigger_at")):
            return err
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
        # Empty name → upsert_contact would create a row with name=""
        # invisible in /people, /about, and get_contacts (LIKE '%%' matches
        # but the rendered output is '— (relation)'). Defend at the boundary.
        if not name or not name.strip():
            return "update_contact requires a non-empty name"
        # Strip padded fields so " 04-15 " doesn't slip past
        # substr(birthday, -5) — that slice would return "-15 " and the
        # MM-DD lookup window in get_upcoming_birthdays never matches.
        # Same risk for relation ("(  друг  )" in /people) and notes
        # (whitespace accumulating on each upsert append).
        relation = _strip_or_none(relation)
        birthday = _strip_or_none(birthday)
        notes = _strip_or_none(notes)
        # Validate birthday shape so a typo doesn't silently break
        # get_upcoming_birthdays — it uses substr(birthday, -5) for the
        # MM-DD match, which never lines up with bogus input. User adds
        # "Маша 1990-13-99", waits for the alert that never comes.
        if birthday and (err := self._check_birthday(birthday, "update_contact")):
            return err
        contact = self.db.upsert_contact(name.strip(), relation, birthday, notes)
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

    # Source of truth is core.NOTIFICATION_KIND_LABELS. Re-aliased to
    # preserve `self._SEARCH_KIND_LABELS.get(...)` callers.
    _SEARCH_KIND_LABELS = NOTIFICATION_KIND_LABELS

    # Backward-compat alias — implementation moved to core.fmt_utc_to_local
    # so telegram.py can share the same conversion. Existing callers
    # (`self._ts_utc_to_local_str(...)`) keep working.
    _ts_utc_to_local_str = staticmethod(fmt_utc_to_local)

    _DIARY_CONTENT_CHAR_CAP = 600

    def _handle_get_diary_entries(self, days: int = 7, limit: int = 5) -> str:
        rows = self.db.get_diary_entries(days=days)
        if not rows:
            return f"Записей в дневнике за последние {days} дн. нет."
        # Cap content per entry to keep the LLM round bounded — a 30-day
        # dump of full diary text would dominate the token budget. Recent
        # entries are usually most informative for "what did I do?" queries
        # so we keep DB ordering (newest first) and just truncate count.
        truncated = rows[:limit]
        lines = []
        for entry in truncated:
            date = entry.get("date") or "?"
            mood = entry.get("mood")
            people = (entry.get("people") or "").strip()
            content = (entry.get("content") or "")[: self._DIARY_CONTENT_CHAR_CAP]
            header_parts = [f"📖 {date}"]
            if mood:
                header_parts.append(f"настроение: {mood}")
            if people:
                header_parts.append(f"люди: {people[:120]}")
            line = " | ".join(header_parts) + f"\n{content}"
            if len(entry.get("content") or "") > self._DIARY_CONTENT_CHAR_CAP:
                line += "…"
            lines.append(line)
        if len(rows) > limit:
            extra = len(rows) - limit
            lines.append(
                f"... и ещё {extra} {pluralize_ru(extra, _RECORDS_FORMS)}"
            )
        return "\n\n".join(lines)

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

    _WEATHER_FORECAST_HORIZON = 7  # Open-Meteo limit

    async def _handle_get_weather(self, date: str | None = None,
                                  days: int | None = None) -> str:
        """Resolve `date` first if both are passed: 'погода в субботу?'
        is a much more common LLM output than 'погода на 3 дня вперёд',
        and pre-fix the date was silently ignored — caller saw today's
        weather no matter what.

        Strategy:
          - date: parse YYYY-MM-DD, derive N = (target - today) + 1, fetch
            that many days, filter the daily list to just the matching row.
          - days: legacy multi-day forecast, no filtering.
          - neither: today only.
        """
        if not self.weather:
            return "Weather not configured."

        target_date: str | None = None
        if date:
            # Reject malformed dates early — pre-fix this caught the
            # ValueError, set target=None, and fell through to "today
            # only", so an LLM passing date="tomorrow" got today's
            # weather instead and reported it as tomorrow's. Now the
            # LLM sees a format hint and can retry with a real date.
            if (err := self._check_iso_date(date, "get_weather", "date")):
                return err
            target = datetime.strptime(date, "%Y-%m-%d").date()
            today = tz_now(self.weather.tz).date()
            delta = (target - today).days
            if delta < 0:
                return f"Дата {date} в прошлом — прогноз не доступен."
            if delta >= self._WEATHER_FORECAST_HORIZON:
                return (
                    f"Дата {date} слишком далеко — Open-Meteo даёт "
                    f"максимум {self._WEATHER_FORECAST_HORIZON} дней."
                )
            days = delta + 1
            target_date = date

        forecast = await self.weather.get_forecast(days=days or 1)

        if target_date:
            matching = next(
                (d for d in forecast.get("daily", [])
                 if d.get("date") == target_date),
                None,
            )
            if not matching:
                return f"Не удалось получить прогноз на {target_date}."
            return self.weather.format_daily({"daily": [matching]})

        return self.weather.format_daily(forecast)

    def _handle_log_habit(self, habit_name: str, done: bool,
                          date: str | None = None) -> str:
        # Empty habit_name would store a row in habits with name="" — invisible
        # in /habits stats and impossible to log against again (the dedup
        # lookup matches against name). Defend at the boundary.
        if not habit_name or not habit_name.strip():
            return "log_habit requires a non-empty habit_name"
        # Validate the date string so a stray "вчера" / "yesterday" passed
        # by the LLM doesn't land in habit_log with garbage that
        # get_habit_stats then silently skips (Cyrillic > '2026' lexically,
        # so `date <= '2026-05-07'` filters it out and the user thinks
        # the habit wasn't logged). Mirrors get_events / get_daily_goals.
        if date and (err := self._check_iso_date(date, "log_habit", "date")):
            return err
        result = self.db.log_habit(habit_name.strip(), done, date)
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
        # decisions.created_at is UTC (`DEFAULT (datetime('now'))`).
        # Slicing [:10] would give the UTC date, off by a day for users
        # whose local time crosses the UTC date boundary at the hour
        # they make decisions (Asia/Almaty +5 — a 03:00 May 8 decision
        # is stored as "2026-05-07 22:00" UTC, so the bot would render
        # it as "[2026-05-07]" while the user thinks "today".
        tz_name = getattr(self.db, "_timezone", None)
        lines = []
        for r in rows:
            created_local = self._ts_utc_to_local_str(
                r.get("created_at") or "", tz_name,
            )
            created = created_local[:10]
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
        # Empty description → row in decisions with description="" —
        # invisible to get_decisions (rendered as "[date]  — context"),
        # and /loops "due_decisions" shows blank entries. Reject up front.
        if not description or not description.strip():
            return "track_decision requires a non-empty description"
        description = description.strip()
        # Save first, search second. Pre-fix the order was reversed and
        # any failure in get_past_decisions (DB hiccup, weird LIKE input)
        # would propagate up the execute() wrapper as "Error executing
        # track_decision" — and the user's intent never made it into the
        # decisions table. The "similar past decisions" hint is nice but
        # strictly secondary; capturing what the user said is the
        # commitment we can't lose.
        decision = self.db.create_decision(description, context, follow_up_days)

        result = f"Decision tracked: {description}. Follow-up in {follow_up_days} days."

        # Best-effort similar-decisions lookup. A failure here is logged
        # and swallowed so the create above still surfaces to the LLM
        # cleanly. Empty/whitespace description → skip the search since
        # there's no useful keyword to extract.
        words = description.split() if description and description.strip() else []
        if words:
            try:
                past = self.db.get_past_decisions(words[0], limit=3)
            except Exception as e:
                logger.warning(
                    "track_decision: past lookup failed (%s) — decision still saved",
                    type(e).__name__,
                )
                past = []
            if past:
                result += "\n\nSimilar past decisions:"
                for p in past:
                    sentiment = p.get('outcome_sentiment', '?')
                    result += f"\n- {p['description'][:80]} → {(p.get('outcome') or 'no outcome')[:80]} ({sentiment})"
        return result

    def _handle_get_daily_goals(self, date: str | None = None) -> str:
        if date and (err := self._check_iso_date(date, "get_daily_goals", "date")):
            return err
        rows = self.db.get_daily_goals(date)
        when = date or "сегодня"
        if not rows:
            return f"Целей на {when} нет."
        # Status labels in Russian so the LLM (and user-facing rendering)
        # stays consistent — same vocabulary the complete_daily_goal handler
        # uses on resolution.
        status_ru = {
            "pending": "не отмечена",
            "completed": "выполнена",
            "skipped": "пропущена",
            "partial": "частично",
        }
        prio_ru = {"high": "высокий", "medium": "средний", "low": "низкий"}
        lines = []
        for g in rows:
            title = (g.get("title") or "")[:200]
            status = status_ru.get(g.get("status") or "pending", g.get("status") or "?")
            prio = prio_ru.get(g.get("priority") or "medium", g.get("priority") or "?")
            line = f"- [{status}] {title} (приоритет: {prio})"
            reflection = (g.get("reflection") or "").strip()
            if reflection:
                line += f" — {reflection[:150]}"
            lines.append(line)
        return "\n".join(lines)

    def _handle_set_daily_goal(self, title: str, description: str | None = None,
                               priority: str = "medium") -> str:
        # Empty title → daily_goals row with title="" — /goals output reads
        # "⬜ (приоритет: medium)" with no actionable label, and
        # complete_daily_goal_by_hint can't match it back.
        if not title or not title.strip():
            return "set_daily_goal requires a non-empty title"
        goal = self.db.create_daily_goal(title.strip(), description, priority)
        prio_ru = {"high": "высокий", "medium": "средний", "low": "низкий"}.get(priority, priority)
        return f"Goal set: {title} (приоритет: {prio_ru})"

    def _handle_complete_daily_goal(self, goal_hint: str, status: str = "completed",
                                    reflection: str | None = None) -> str:
        if not goal_hint or not goal_hint.strip():
            return "complete_daily_goal requires a non-empty goal_hint"
        # Coerce invalid status BEFORE the DB call so the rendered Russian
        # label matches what's actually stored. Pre-fix the DB silently
        # defaulted bad input to 'completed' but the handler rendered the
        # LLM's raw value, so a status="done" call would read "marked as
        # done" while the row actually said 'completed'. Mirror of the
        # sentiment coercion in _handle_resolve_decision.
        if status not in (Status.COMPLETED, Status.SKIPPED, Status.PARTIAL):
            status = Status.COMPLETED
        # Count first so we can disclose ambiguity ("matched 2, marked
        # the soonest"). Single-threaded SQLite, no race with the mark
        # below.
        total = self.db.count_pending_goals_matching(goal_hint)
        result = self.db.complete_daily_goal_by_hint(goal_hint.strip(), status, reflection)
        if not result:
            return f"No pending goal found matching '{goal_hint}' for today"
        status_ru = {"completed": "выполнена", "skipped": "пропущена", "partial": "частично"}.get(status, status)
        msg = f"Goal '{result['title']}' marked as {status_ru}"
        remaining = total - 1
        if remaining > 0:
            msg += f". Похожих ещё {remaining} — уточни если нужно отметить и их."
        return msg

    def _handle_resolve_decision(self, description_hint: str, outcome: str,
                                 sentiment: str = "neutral") -> str:
        if not description_hint or not description_hint.strip():
            return "resolve_decision requires a non-empty description_hint"
        if not outcome or not outcome.strip():
            return "resolve_decision requires a non-empty outcome"
        if sentiment not in (Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE):
            sentiment = Sentiment.NEUTRAL
        # Count first so we can disclose ambiguity in the response —
        # single-threaded SQLite, no race with the resolve below.
        total = self.db.count_pending_decisions_matching(description_hint)
        resolved = self.db.resolve_decision_by_hint(
            description_hint.strip(), outcome, sentiment,
        )
        if not resolved:
            return f"No pending decision found matching '{description_hint}'"
        msg = f"Resolved decision: {resolved['description'][:80]} → {outcome[:80]}"
        remaining = total - 1
        if remaining > 0:
            msg += f". Похожих ещё {remaining} — уточни если нужно закрыть и их."
        return msg
