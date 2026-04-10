# MindSecretary — Архитектура

## Что это и зачем

MindSecretary — **внешний мозг**, а не ремайндер. Telegram-бот, которому ты говоришь голосом всё, что в голове, а он:

- **Связывает факты** — ты жаловался на спину 3 раза за 3 месяца, в зал ходил 4 раза → "может дело в кресле, а не в зале?"
- **Знает твоих людей** — перед встречей с Димой: "вы не виделись 4 месяца, он переезжал, жена рожала, ты обещал контакт юриста"
- **Видит паттерны** — "ты опаздываешь по понедельникам утром", "импульсивные покупки по пятницам после тяжёлых дней"
- **Упреждает проблемы** — "годовщина через 2 недели, в прошлом году подарок покупал за день — начни сейчас?"

**Voice-first:** основной ввод — голосовые сообщения. 20 секунд потока сознания → бот извлекает события, факты, обещания, настроение. Никаких форм, кнопок, ручного ввода.

---

## Ключевые решения

| Решение | Выбор | Почему |
|---------|-------|--------|
| LLM (основная) | MiniMax M-2.7 | $0.30/$1.20 за M tokens, tool use, agentic |
| LLM (fallback/аналитика) | Claude Sonnet 4.6 | Fallback + weekly review, лучший русский |
| STT (голос → текст) | Groq Whisper | $0.0011/мин, быстро, отличный русский |
| Embeddings | Voyage AI (voyage-3) | Лучшие embeddings, ~бесплатно |
| Хранилище | SQLite (одна БД) | Всё в одном файле: vectors, events, contacts, interactions |
| Vector search | numpy cosine similarity | <10мс для тысяч записей, без лишних зависимостей |
| Interface | Telegram Bot | Голосовые, push, реакции, пересылка |
| Scheduler | APScheduler | Проактивные задачи |

**Стоимость: ~$1.20/мес.** Не внешние календари, не облака — бот и есть твой органайзер.

---

## Поток данных

```
                    ВХОД (всё через Telegram)
                    ─────────────────────────
    🎤 Голосовые         💬 Текст         ↪️ Пересылка
    (основной)           (быстрые)        (из других чатов)
         │                   │                  │
         ▼                   │                  │
    ┌──────────┐             │                  │
    │  Groq    │             │                  │
    │  Whisper │             │                  │
    │  (STT)   │             │                  │
    └────┬─────┘             │                  │
         │ текст             │                  │
         └───────────┬───────┘──────────────────┘
                     ▼
              ┌─────────────┐
              │    BRAIN     │
              │              │
              │ 1. Voyage    │──→ SQLite: cosine search по memories
              │    embed     │
              │ 2. SQLite    │──→ Календарь, контакты, ремайндеры
              │    query     │
              │ 3. Контекст  │
              │ 4. MiniMax   │──→ Ответ + tool calls
              │ 5. Execute   │
              │ 6. Сохранить │──→ Voyage embed → SQLite (BLOB)
              └──────┬──────┘
                     ▼
              Ответ в Telegram

    ─────────────────────────────────────────────
                🤖 Бот спрашивает сам (3x/день)
    ─────────────────────────────────────────────
    07:00  "Что сегодня в планах?"
    после события: "Как прошла встреча с Петей?"
    21:00  "Что важного было сегодня?"
```

### Четыре канала ввода

| Канал | Что | Усилие | Объём данных |
|-------|-----|--------|-------------|
| 🎤 Голосовые | Поток сознания, рассказы, планы | **Низкое** — 15-20 сек | Высокий — 2-4 факта за сообщение |
| 🤖 Бот спрашивает | Планы утром, итоги вечером, follow-up после событий | **Минимальное** — ответ на вопрос | Высокий — структурированный |
| 💬 Текст | Быстрые команды: "напомни", "что завтра?" | Среднее | Низкий — 1 действие |
| ↪️ Пересылка | Договорённости из других чатов | **Минимальное** — один tap | Средний — контекст из чужих сообщений |

**~1 минута голосовых в день → 90+ записей в месяц → 200-300 извлечённых фактов.**

---

## Голосовой пайплайн

### Обработка голосового сообщения

```python
async def handle_voice(update: Update, context: CallbackContext):
    """Главный обработчик голосовых сообщений."""
    voice = update.message.voice or update.message.audio

    # 1. Скачиваем .ogg файл из Telegram
    file = await context.bot.get_file(voice.file_id)
    ogg_bytes = await file.download_as_bytearray()

    # 2. Транскрибируем через Groq Whisper
    transcript = await stt.transcribe(ogg_bytes)

    # 3. Дальше как обычный текст → Brain
    response = await brain.process(
        user_message=transcript,
        message_type="voice",
        metadata={"duration_sec": voice.duration},
    )

    # 4. Ответ пользователю
    await update.message.reply_text(response.text)
```

### STT-клиент (Groq Whisper)

```python
class GroqSTT:
    """Speech-to-Text через Groq Whisper API."""

    def __init__(self, api_key: str):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=api_key)

    async def transcribe(self, audio_bytes: bytes, language: str = "ru") -> str:
        # Groq принимает файл напрямую
        from io import BytesIO
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "voice.ogg"

        response = await self.client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language=language,
            response_format="text",
        )
        return response.strip()
```

### Извлечение данных из потока сознания

Ключевая задача — из свободного голоса извлечь структурированные действия.

```
🎤 "Сегодня был у врача, сказал что МРТ надо сделать, записался
    на двадцатое. Дима звонил, зовёт на рыбалку в выходные,
    надо подумать. Жена просила купить молоко"

→ MiniMax получает system prompt:
  "Извлеки из сообщения ВСЕ факты, события, обещания, задачи.
   Для каждого вызови подходящий инструмент."

→ Tool calls:
  save_memory(content="Был у врача, направление на МРТ", category="health", importance=8)
  create_event(title="МРТ", start_at="2026-04-20T09:00")
  save_memory(content="Дима зовёт на рыбалку в выходные, нужно решить", category="personal",
              related_person="Дима")
  create_reminder(text="Решить насчёт рыбалки с Димой", trigger_at="2026-04-09T19:00")
  create_reminder(text="Купить молоко по дороге домой", trigger_at="2026-04-07T18:00")

→ Бот: "Записал: МРТ 20-го, Дима зовёт на рыбалку (напомню решить в четверг),
        молоко — напомню в 18:00."
```

**Одно голосовое 30 секунд → 5 структурированных действий.** В этом сила voice-first.

---

## Персональный CRM

Каждый упомянутый человек автоматически попадает в граф контактов. Со временем накапливается контекст.

### Как это работает

```
Месяц 1:
  🎤 "Встретил Олега, он переехал в Питер"
  → contacts: Олег | память: "переехал в Питер"

  🎤 "Олег скинул вакансию, интересная, но далеко"
  → память: "Олег прислал вакансию, показалось далеко"

Месяц 3:
  🎤 "Завтра обедаю с Олегом"
  → Бот (поиск по памяти):
    "Олег. Вы не виделись ~2 месяца.
     Последнее: он переехал в Питер, присылал вакансию.
     Можешь спросить как устроился."
```

### Брифинг перед встречей

Когда у тебя в календаре (SQLite) событие с человеком, бот автоматически ищет по памяти всё, что знает о нём:

```python
async def pre_meeting_briefing(event: dict):
    """Генерирует брифинг перед встречей с человеком."""
    person = event.get("related_person")
    if not person:
        return

    # Всё, что знаем о человеке
    person_memories = memory.search(person, top_k=10)
    contact = db.get_contact_by_name(person)

    # Незакрытые обещания этому человеку
    promises = memory.search(
        f"обещания {person}",
        category="promise",
        top_k=5,
    )

    # LLM генерирует брифинг
    response = await router.chat(
        system=PRE_MEETING_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Встреча с: {person}\n"
                       f"Контакт: {format_contact(contact)}\n"
                       f"Память: {format_memories(person_memories)}\n"
                       f"Обещания: {format_memories(promises)}"
        }],
        max_tokens=500,
    )

    await telegram.send_message(user_id, f"📋 Перед встречей с {person}:\n\n{response.text}")
```

### Автоматическое извлечение контактов

MiniMax автоматически обновляет контакты из разговоров:

```python
# В списке tools:
{
    "name": "update_contact",
    "description": "Создать или обновить информацию о человеке. "
                   "Вызывай когда пользователь упоминает новый факт о ком-то: "
                   "переезд, новая работа, семейные события, интересы.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "relation": {"type": "string", "description": "друг, коллега, жена, врач..."},
            "birthday": {"type": "string"},
            "notes": {"type": "string", "description": "Новые факты о человеке"}
        },
        "required": ["name"]
    }
}
```

---

## Система инсайтов

### Связывание фактов (Voyage + Sonnet, еженедельно)

Voyage embeddings позволяют находить **неочевидные связи** между фактами:

```python
async def find_insights(memory: Memory, router: ModelRouter):
    """Еженедельный поиск связей между фактами."""

    # 1. Берём последние воспоминания
    recent = memory.search("важные события и факты за неделю", top_k=20)

    # 2. Для каждого ищем связанные старые воспоминания
    connections = []
    for mem in recent:
        related = memory.search(mem["content"], top_k=5)
        # Фильтруем: только если связь неочевидна (разные категории, разное время)
        for r in related:
            if r["id"] != mem["id"] and r["category"] != mem["category"]:
                connections.append((mem, r))

    # 3. Sonnet анализирует связи (раз в неделю — можно качественно)
    response = await router.chat(
        system="""Проанализируй связи между фактами из жизни пользователя.
Найди неочевидные паттерны и инсайты.
Для каждого инсайта:
- Что заметил (конкретно)
- На чём основано (какие факты связал)
- Что предлагаешь (конкретное действие)
Только значимые инсайты. Не выдумывай — если связи нет, не придумывай.""",
        messages=[{"role": "user", "content": format_connections(connections)}],
        use_fallback=True,  # Sonnet
        max_tokens=1500,
    )

    return response.text
```

### Поведенческие паттерны (Sonnet, еженедельно)

```python
async def analyze_patterns(db: Database, memory: Memory, router: ModelRouter):
    """Анализ поведенческих паттернов за неделю."""

    interactions = db.get_interactions(since=now - timedelta(days=7))
    events = db.get_events(date_from=week_ago, date_to=today)
    learnings = memory.get_by_category("learning")

    response = await router.chat(
        system="""Ты — аналитик поведенческих паттернов.
Проанализируй неделю пользователя:

1. Время: когда активен, когда молчит, когда опаздывает
2. Люди: с кем общался, кого давно не видел
3. Обещания: что выполнил, что забросил
4. Привычки: прогресс по трекеру
5. Настроение: по тону сообщений (не спрашивай — выводи из данных)
6. Нетривиальные связи: что с чем коррелирует

Сравни с прошлыми learnings. Подтверди или опровергни.
Только конкретные наблюдения с числами. Не общие фразы.""",
        messages=[{
            "role": "user",
            "content": f"Взаимодействия:\n{format_interactions(interactions)}\n\n"
                       f"События:\n{format_events(events)}\n\n"
                       f"Текущие learnings:\n{format_learnings(learnings)}"
        }],
        use_fallback=True,  # Sonnet для глубокого анализа
        max_tokens=2000,
    )

    return response.text
```

### Пример недельного обзора (генерирует Sonnet)

```
📈 Неделя 12:

👤 Люди:
  • Общался с 6 людьми. Чаще всего — Петя (3 встречи).
  • Давно не общался: Дима (3 мес), мама (2 недели — раньше звонил чаще).

💡 Инсайты:
  • Ты жалуешься на усталость каждый вторник вечером.
    Все 4 вторника — 3+ встреч. Может разгрузить вторник?
  • Ты обещал 5 вещей, выполнил 2. Оба невыполненных — обещания
    "по дороге" (аптека, молоко). Ты забываешь дела-по-дороге.
    Начну напоминать за 15 мин до выхода.
  • Третий раз упоминаешь "надо бы заняться английским".
    Ни разу не начал. Напомнить конкретно или забить?

📊 Привычки:
  • Спорт: 3/5 дней ✓ (прошлая неделя: 2/5, прогресс)
  • Ранний подъём: 4/5 (в пятницу проспал)
```

---

## Проактивная система

### Расписание

| Задача | Когда | Движок | Что делает |
|--------|-------|--------|------------|
| **morning_prompt** | wake_up | Без LLM | "Что сегодня в планах?" — запрашивает данные |
| **morning_briefing** | wake_up + 5 мин (после ответа) | MiniMax | Погода + ответ пользователя → брифинг |
| **pre_meeting** | за 30 мин до события | MiniMax | Брифинг по человеку из CRM |
| **post_meeting** | через 30 мин после события | Без LLM | "Как прошла встреча?" — запрос follow-up |
| **reminder_check** | каждые 5 мин | Без LLM | `if now >= trigger_at` → отправить |
| **weather_monitor** | каждые 60 мин | Без LLM | Сравнить прогноз, при резком изменении → MiniMax |
| **evening_prompt** | 21:00 | Без LLM | "Что важного было сегодня?" |
| **evening_summary** | 21:15 (после ответа) | MiniMax | Итоги дня |
| **weekly_review** | воскресенье 20:00 | **Sonnet** | Паттерны + инсайты + CRM |
| **learning_cycle** | воскресенье 22:00 | **Sonnet** | Анализ → новые learnings |
| **birthday_check** | ежедневно 09:00 | Без LLM | SQL-запрос по контактам |

**Половина задач работает без LLM** — чистая логика, SQL-запросы, простые проверки.

### Follow-up после событий

```python
async def schedule_post_event_followup(event: dict):
    """Запланировать follow-up после события."""
    end_time = parse_datetime(event["end_at"] or event["start_at"]) + timedelta(minutes=30)

    scheduler.add_job(
        ask_followup,
        trigger="date",
        run_date=end_time,
        args=[event],
    )

async def ask_followup(event: dict):
    """Спросить как прошло событие."""
    person = event.get("related_person", "")
    title = event["title"]

    text = f"Как прошло: {title}?"
    if person:
        text += f"\nЧто обсудили с {person}? Договорились о чём-нибудь?"

    await telegram.send_message(user_id, text)
    # Ответ пользователя (текст или голосовое) обработается обычным путём
    # Brain извлечёт факты, обещания, обновит контакт
```

---

## LLM-клиент: абстракция над провайдерами

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[dict]       # [{"name": "...", "arguments": {...}}]
    usage: dict                   # {"input_tokens": N, "output_tokens": N}

class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...

class MiniMaxClient(LLMClient):
    """Основная модель. OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str = "https://api.minimax.chat/v1"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = "MiniMax-M2.7"

    async def chat(self, system, messages, tools=None, max_tokens=1024):
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }} for t in tools
            ]

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message

        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return LLMResponse(
            text=choice.content,
            tool_calls=tool_calls,
            usage={"input_tokens": resp.usage.prompt_tokens,
                   "output_tokens": resp.usage.completion_tokens},
        )

class AnthropicClient(LLMClient):
    """Fallback + weekly review."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6-20250514"):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, system, messages, tools=None, max_tokens=1024):
        kwargs = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        resp = await self.client.messages.create(**kwargs)

        text = None
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input})

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage={"input_tokens": resp.usage.input_tokens,
                   "output_tokens": resp.usage.output_tokens},
        )

class ModelRouter:
    """Выбирает модель: primary (MiniMax) или fallback (Sonnet)."""

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self.primary = primary
        self.fallback = fallback

    async def chat(self, system, messages, tools=None, max_tokens=1024,
                   use_fallback=False):
        client = self.fallback if use_fallback else self.primary
        try:
            return await client.chat(system, messages, tools, max_tokens)
        except Exception as e:
            if client is self.primary:
                logger.warning(f"MiniMax failed: {e}, falling back to Sonnet")
                return await self.fallback.chat(system, messages, tools, max_tokens)
            raise
```

---

## Система памяти (SQLite + Voyage)

### Одна БД на всё

```
mindsecretary.db
├── memories     — семантическая память (embedding как BLOB)
├── events       — календарь (бот = твой календарь)
├── reminders    — напоминания
├── contacts     — персональный CRM
├── interactions — лог взаимодействий (для learning)
├── preferences  — learned preferences
├── habits       — трекинг привычек
└── habit_log    — история привычек
```

Один файл. Бэкап = `cp mindsecretary.db backup.db`.

### Модуль памяти

```python
import numpy as np
import voyageai
import sqlite3

class Memory:
    def __init__(self, db_path: str, voyage_api_key: str):
        self.db = sqlite3.connect(db_path)
        self.voyage = voyageai.Client(api_key=voyage_api_key)
        self._ensure_table()

    def _ensure_table(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                category TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                related_person TEXT,
                related_date TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_mem_cat ON memories(category)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status)")
        self.db.commit()

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        result = self.voyage.embed(texts, model="voyage-3", input_type="document")
        return [np.array(e, dtype=np.float32) for e in result.embeddings]

    def _embed_query(self, text: str) -> np.ndarray:
        result = self.voyage.embed([text], model="voyage-3", input_type="query")
        return np.array(result.embeddings[0], dtype=np.float32)

    def save(self, content: str, category: str, importance: int = 5,
             related_person: str = None, related_date: str = None) -> str:
        import uuid
        memory_id = str(uuid.uuid4())[:8]
        embedding = self._embed([content])[0]

        self.db.execute(
            "INSERT INTO memories (id, content, embedding, category, importance, "
            "related_person, related_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, content, embedding.tobytes(), category, importance,
             related_person, related_date)
        )
        self.db.commit()
        return memory_id

    def search(self, query: str, top_k: int = 10,
               category: str = None, min_importance: int = 0) -> list[dict]:
        query_emb = self._embed_query(query)

        where = "WHERE status = 'active'"
        params = []
        if category:
            where += " AND category = ?"
            params.append(category)
        if min_importance > 0:
            where += " AND importance >= ?"
            params.append(min_importance)

        rows = self.db.execute(
            f"SELECT id, content, embedding, category, importance, "
            f"related_person, related_date, created_at FROM memories {where}",
            params
        ).fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            emb = np.frombuffer(row[2], dtype=np.float32)
            score = float(np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8
            ))
            results.append({
                "id": row[0], "content": row[1], "score": score,
                "category": row[3], "importance": row[4],
                "related_person": row[5], "related_date": row[6],
                "created_at": row[7],
            })

        for r in results:
            r["final_score"] = r["score"] * 0.6 + (r["importance"] / 10) * 0.4
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def get_by_category(self, category: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, content, category, importance, related_person, related_date "
            "FROM memories WHERE category = ? AND status = 'active' ORDER BY importance DESC",
            (category,)
        ).fetchall()
        return [{"id": r[0], "content": r[1], "category": r[2],
                 "importance": r[3], "related_person": r[4], "related_date": r[5]}
                for r in rows]
```

---

## Схема базы данных

```sql
-- Семантическая память
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    category TEXT NOT NULL,       -- contact|health|work|personal|promise|preference|location|learning
    importance INTEGER DEFAULT 5, -- 1-10
    related_person TEXT,
    related_date TEXT,
    status TEXT DEFAULT 'active', -- active|fulfilled|expired|deleted
    created_at TEXT DEFAULT (datetime('now')),
    last_accessed TEXT
);

-- Календарь (бот = твой календарь)
CREATE TABLE events (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
    title TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT,
    location TEXT,
    description TEXT,
    related_person TEXT,          -- для pre-meeting briefing
    recurring TEXT,               -- NULL|daily|weekly|monthly|yearly
    source TEXT DEFAULT 'voice',  -- voice|text|forward
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_events_start ON events(start_at);

-- Напоминания
CREATE TABLE reminders (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
    text TEXT NOT NULL,
    trigger_at TEXT NOT NULL,
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'acknowledged', 'dismissed')),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_reminders_trigger ON reminders(trigger_at);
CREATE INDEX idx_reminders_status ON reminders(status);

-- Персональный CRM
CREATE TABLE contacts (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
    name TEXT NOT NULL,
    aliases TEXT,                  -- JSON: ["Саша", "Александр"]
    relation TEXT,                 -- друг, коллега, жена...
    birthday TEXT,                 -- MM-DD или YYYY-MM-DD
    phone TEXT,
    notes TEXT,                    -- последние заметки
    last_contact TEXT,             -- когда последний раз общались
    contact_frequency INTEGER,     -- раз в сколько дней обычно общаемся
    mention_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Взаимодействия (для learning + аналитики)
CREATE TABLE interactions (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
    timestamp TEXT DEFAULT (datetime('now')),
    direction TEXT CHECK (direction IN ('in', 'out')),
    message_type TEXT,             -- voice|text|forward|briefing|notification|reminder|prompt
    content TEXT NOT NULL,
    voice_duration_sec REAL,       -- длительность голосового (если есть)
    feedback TEXT,                  -- positive|negative|ignored|NULL
    read_at TEXT,
    response_time_sec REAL,
    metadata TEXT                   -- JSON
);

CREATE INDEX idx_interactions_ts ON interactions(timestamp);

-- Learned preferences
CREATE TABLE preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT 'default' CHECK (source IN ('default', 'explicit', 'learned')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Привычки
CREATE TABLE habits (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
    name TEXT NOT NULL UNIQUE,
    target TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE habit_log (
    habit_id TEXT REFERENCES habits(id),
    date TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 1,
    notes TEXT,
    PRIMARY KEY (habit_id, date)
);
```

---

## System prompt

```
Ты — MindSecretary, персональный AI-секретарь для {name}.
Ты — внешний мозг, который помнит, связывает факты и думает за пользователя.

## Твоя роль
- Из каждого сообщения (особенно голосового) извлекай ВСЕ факты, события, обещания, упоминания людей
- Связывай новые факты с тем, что уже знаешь
- Если знаешь что-то релевантное, что не спрашивали — упомяни
- Если упоминают человека — обнови контакт
- Будь кратким

## Профиль
{profile_yaml}

## Контекст
Дата: {date}, {day_of_week}
Время: {time}

## Релевантные воспоминания
{retrieved_memories}

## События сегодня
{today_events}

## Последние сообщения
{recent_messages}

## Правила
1. Из голосовых извлекай КАЖДЫЙ факт и действие — люди наговаривают много за раз
2. Отвечай на русском, кратко
3. При упоминании человека — save_memory + update_contact
4. При упоминании даты/плана — create_event или create_reminder
5. При упоминании обещания — save_memory с category="promise"
6. Если не уверен во времени — уточни
7. В конце ответа кратко перечисли что записал
```

---

## Tool definitions

```python
TOOLS = [
    {
        "name": "save_memory",
        "description": "Сохранить факт, обещание или наблюдение. Вызывай для КАЖДОГО "
                       "нового факта из сообщения пользователя. Из одного голосового "
                       "может быть несколько вызовов.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Что запомнить. Включи контекст: кто, когда, связь."
                },
                "category": {
                    "type": "string",
                    "enum": ["contact", "health", "work", "personal",
                             "promise", "preference", "location", "learning"],
                },
                "importance": {
                    "type": "integer", "minimum": 1, "maximum": 10,
                    "description": "1=мелочь, 5=полезно, 8=важно, 10=критично"
                },
                "related_person": {"type": "string"},
                "related_date": {"type": "string", "description": "YYYY-MM-DD"}
            },
            "required": ["content", "category", "importance"]
        }
    },
    {
        "name": "search_memory",
        "description": "Семантический поиск по памяти. Используй для вопросов "
                       "о прошлом, людях, обещаниях.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["contact", "health", "work", "personal",
                             "promise", "preference", "location", "learning"],
                }
            },
            "required": ["query"]
        }
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
                "related_person": {"type": "string", "description": "Кто участвует — для CRM briefing"},
                "description": {"type": "string"}
            },
            "required": ["title", "start_at"]
        }
    },
    {
        "name": "get_events",
        "description": "Получить события за период.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string"}
            },
            "required": ["date_from"]
        }
    },
    {
        "name": "create_reminder",
        "description": "Создать напоминание.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "trigger_at": {"type": "string", "description": "YYYY-MM-DDTHH:MM"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]}
            },
            "required": ["text", "trigger_at"]
        }
    },
    {
        "name": "update_contact",
        "description": "Создать или обновить контакт. Вызывай при ЛЮБОМ упоминании "
                       "нового факта о человеке: переезд, работа, семья, интересы.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "relation": {"type": "string"},
                "birthday": {"type": "string"},
                "notes": {"type": "string", "description": "Новые факты о человеке (добавляются к существующим)"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_contacts",
        "description": "Найти контакты.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "Прогноз погоды.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "days": {"type": "integer", "maximum": 7}
            }
        }
    },
    {
        "name": "log_habit",
        "description": "Отметить выполнение привычки.",
        "input_schema": {
            "type": "object",
            "properties": {
                "habit_name": {"type": "string"},
                "date": {"type": "string"},
                "done": {"type": "boolean"}
            },
            "required": ["habit_name", "done"]
        }
    }
]
```

---

## Структура проекта

```
mindsecretary/
├── pyproject.toml
├── .env                          # API keys
├── config/
│   ├── profile.yaml
│   └── settings.yaml
│
├── src/
│   └── mindsecretary/
│       ├── __init__.py
│       ├── app.py                # Entry point
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── brain.py          # Оркестратор: voice/text → context → LLM → tools
│       │   ├── memory.py         # Voyage + SQLite: embed, save, search
│       │   ├── database.py       # SQLite: events, reminders, contacts, interactions
│       │   └── config.py         # Profile + settings + env
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py         # LLMClient ABC + MiniMaxClient + AnthropicClient
│       │   ├── router.py         # ModelRouter: primary/fallback
│       │   ├── prompts.py        # System prompts
│       │   └── tools.py          # Tool definitions + execution
│       │
│       ├── voice/
│       │   ├── __init__.py
│       │   └── stt.py            # Groq Whisper: ogg → текст
│       │
│       ├── interfaces/
│       │   ├── __init__.py
│       │   └── telegram.py       # Bot: voice handler, text handler, forwarded handler
│       │
│       ├── proactive/
│       │   ├── __init__.py
│       │   ├── scheduler.py      # APScheduler: все задачи
│       │   ├── briefing.py       # Morning/evening/pre-meeting briefings
│       │   └── monitor.py        # Weather, reminders, birthdays
│       │
│       ├── learning/
│       │   ├── __init__.py
│       │   ├── tracker.py        # Feedback tracking
│       │   └── reflection.py     # Weekly: patterns + insights (Sonnet)
│       │
│       └── integrations/
│           ├── __init__.py
│           └── weather.py        # Open-Meteo API
│
├── data/
│   └── mindsecretary.db          # Одна БД на всё
│
└── tests/
    ├── test_memory.py
    ├── test_brain.py
    ├── test_voice.py
    └── test_llm_client.py
```

---

## Стоимость

```
MiniMax M-2.7:    $0.42/мес   (ежедневный чат, брифинги, tool use)
Sonnet 4.6:       $0.57/мес   (weekly review, insights, fallback)
Groq Whisper:     $0.17/мес   (~5 мин голосовых в день)
Voyage AI:        $0.01/мес   (embeddings)
─────────────────────────────
ИТОГО:            ~$1.20/мес
```

---

## Roadmap

### Фаза 1: Голосовой секретарь с памятью (неделя 1-2)

- [ ] Skeleton: pyproject.toml, структура, .env
- [ ] `config/profile.yaml`
- [ ] `core/config.py` — загрузка конфига
- [ ] `core/database.py` — все таблицы
- [ ] `core/memory.py` — Voyage + SQLite: save/search
- [ ] `voice/stt.py` — Groq Whisper
- [ ] `llm/client.py` — MiniMaxClient + AnthropicClient
- [ ] `llm/router.py` — ModelRouter
- [ ] `llm/tools.py` — save_memory, search_memory, update_contact, create_event, create_reminder
- [ ] `core/brain.py` — оркестратор: voice → STT → context → LLM → tools
- [ ] `interfaces/telegram.py` — voice handler, text handler, forward handler
- [ ] `app.py` — запуск

**Результат:** Бот, которому говоришь голосом. Он запоминает, создаёт события, обновляет контакты.

### Фаза 2: Проактивный секретарь (неделя 3-4)

- [ ] `proactive/scheduler.py` — APScheduler
- [ ] `proactive/briefing.py` — morning prompt + briefing, evening prompt + summary
- [ ] `proactive/monitor.py` — reminder check, birthday check
- [ ] `integrations/weather.py` — Open-Meteo
- [ ] Post-event follow-up ("Как прошла встреча?")
- [ ] Pre-meeting CRM briefing

**Результат:** Бот сам спрашивает, напоминает, готовит к встречам.

### Фаза 3: Умный секретарь (неделя 5-6)

- [ ] `learning/tracker.py` — feedback tracking
- [ ] `learning/reflection.py` — weekly insights + patterns (Sonnet)
- [ ] Habit tracking
- [ ] Preference learning
- [ ] Weather monitor с умными уведомлениями

**Результат:** Недельные обзоры с инсайтами. Система учится.

---

## Зависимости

```toml
[project]
name = "mindsecretary"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "openai>=1.50.0",            # MiniMax (OpenAI-compatible API)
    "anthropic>=0.42.0",          # Sonnet fallback + weekly
    "groq>=0.11.0",               # Whisper STT
    "voyageai>=0.3.0",            # Embeddings
    "numpy>=1.26.0",              # Cosine similarity
    "python-telegram-bot>=21.0",  # Telegram (voice + text)
    "apscheduler>=3.10.0",        # Scheduled jobs
    "httpx>=0.27.0",              # Weather API
    "pyyaml>=6.0",                # Config
    "python-dotenv>=1.0.0",       # .env
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

10 зависимостей. Один файл БД. ~$1.20/мес. Voice-first.
