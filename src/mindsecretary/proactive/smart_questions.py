from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta

from ..core.database import Database
from ..core.memory import Memory
from ..llm.router import ModelRouter

logger = logging.getLogger(__name__)

# Prompts for generating smart questions
GAPS_PROMPT = """\
Ты — подсистема MindSecretary, которая ищет пробелы в знаниях о пользователе.

Вот что мы знаем (контакты):
{contacts}

Вот что в памяти:
{memories}

Последние взаимодействия:
{recent}

Задача: найди 1 самый полезный вопрос, который стоит задать пользователю.

Типы вопросов (выбери один):
1. ПРОБЕЛ — упоминали человека без деталей ("Кто такой Петя — друг, коллега?")
2. FOLLOW-UP — упоминали событие/проблему, но не знаем чем закончилось ("Как прошёл визит к врачу?")
3. НАМЕРЕНИЕ — упоминали желание 2+ раз, но не начали ("Ты 3 раза говорил про английский. Есть план?")
4. ЗДОРОВЬЕ — упоминали недомогание, прошло время ("На прошлой неделе болела голова. Как сейчас?")
5. УГЛУБЛЕНИЕ — знаем факт поверхностно ("Ты работаешь в IT. Чем именно занимаешься?")

Правила:
- Вопрос должен быть естественным, не допросом
- Не повторяй то, что уже знаешь
- Не спрашивай про очевидное
- Максимум 1-2 предложения

Ответь ТОЛЬКО текстом вопроса, без пояснений. Если нечего спрашивать — ответь "SKIP".\
"""


class SmartQuestions:
    """Generate targeted questions to fill knowledge gaps."""

    def __init__(self, router: ModelRouter, memory: Memory, db: Database,
                 min_interactions: int = 5):
        self.router = router
        self.memory = memory
        self.db = db
        self.min_interactions = min_interactions
        self._last_asked: datetime | None = None

    async def generate_question(self) -> str | None:
        """Generate one smart question. Returns None if nothing to ask."""
        # Don't ask more than once per 8 hours
        if self._last_asked and datetime.now() - self._last_asked < timedelta(hours=8):
            return None

        # Need at least some interactions before asking
        recent = self.db.get_interactions(
            since=datetime.now() - timedelta(days=7), limit=30,
        )
        if len(recent) < self.min_interactions:
            return None

        contacts = self.db.get_contacts("")
        contacts_text = "\n".join(
            f"- {c['name']}" + (f" ({c['relation']})" if c.get('relation') else " (связь неизвестна)")
            + (f" заметки: {c['notes'][:100]}" if c.get('notes') else "")
            for c in contacts[:20]
        ) or "Контактов нет."

        memories = await self.memory.search("важные факты о пользователе", top_k=15)
        memories_text = "\n".join(
            f"- [{m['category']}] {m['content'][:120]}"
            for m in memories
        ) or "Память пуста."

        recent_text = "\n".join(
            f"- {i['content'][:100]}" for i in recent[:15]
            if i.get('direction') == 'in'
        )

        try:
            response = await self.router.chat(
                system=GAPS_PROMPT.format(
                    contacts=contacts_text,
                    memories=memories_text,
                    recent=recent_text,
                ),
                messages=[{"role": "user", "content": "Сгенерируй вопрос."}],
                max_tokens=200,
            )
            text = (response.text or "").strip()
            if not text or text == "SKIP":
                return None

            self._last_asked = datetime.now()
            self.db.log_interaction(
                direction="out", message_type="smart_question", content=text,
            )
            return f"🤔 {text}"
        except Exception as e:
            logger.error("Smart question generation failed: %s", e)
            return None
