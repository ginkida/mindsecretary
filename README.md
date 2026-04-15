<p align="center">
  <img src="assets/mascot.png" alt="MindSecretary mascot — a secretarybird holding a stopwatch next to a calendar" width="480">
</p>

<h1 align="center">MindSecretary</h1>

<p align="center">
  Voice-first personal AI secretary and companion as a Telegram bot.
  <br>
  Speak freely — it remembers everything, helps you think, tracks your goals, and analyzes your week.
</p>

<p align="center">
  <a href="https://github.com/ginkida/mindsecretary/releases"><img src="https://img.shields.io/github/v/release/ginkida/mindsecretary?color=4c1" alt="Release"></a>
  <a href="https://github.com/ginkida/mindsecretary/pkgs/container/mindsecretary"><img src="https://img.shields.io/badge/ghcr.io-mindsecretary-4c1" alt="Container"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/ginkida/mindsecretary" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
</p>

---

Send voice messages, text, photos, or forwards — the bot extracts facts, events, promises, and contact updates, stores them in semantic memory, and proactively reminds, briefs, and analyzes patterns in your life. When you just want to talk — it listens, responds with specificity, and connects what you say to what it already knows about you.

Single-user. Runs in Docker. Long polling (no webhooks, no public IP needed). All data in one SQLite file. ~$2/month.

## How It Works

```
Input (Telegram)                    Processing                         Output
─────────────────                   ──────────────────                  ──────────────
Voice message ──→ Groq Whisper ──→ transcript ─┐
Text message ────────────────────────────────────┤
Photo + caption ─────────────────────────────────┤
Forwarded message ───────────────────────────────┘
                                                 ▼
                                          Brain.process()
                                    1. Voyage AI embed(message)
                                       → SQLite cosine search
                                       → relevant memories
                                    2. SQLite query → events, goals,
                                       reminders, mood, themes
                                    3. Build system prompt with
                                       profile + full context
                                    4. Claude Sonnet + tool calls
                                    5. Execute tools (save_memory,
                                       create_event, set_daily_goal...)
                                    6. Loop until done (max 5 rounds)
                                                 ▼
                                          Response → Telegram
                                          Log interaction + cost
```

One 30-second voice message → 2-5 structured actions (memories, events, reminders, contact updates).

## Features

| Feature | Description |
|---------|-------------|
| **Companion mode** | Dual personality: secretary for tasks, warm companion for personal conversations. Responds with specificity, not clichés. Can gently disagree when it sees contradictions with your past. |
| **Voice-first input** | Speak freely; Groq Whisper transcribes, Claude extracts every fact and action |
| **Photo input** | Send photos (business cards, receipts, screenshots) — Claude vision extracts data |
| **Daily goals** | Set goals in the morning, track throughout the day, review together in the evening |
| **Semantic memory** | Voyage AI embeddings in SQLite; cosine similarity finds relevant context even with different phrasing |
| **Personal CRM** | Tracks people: last contact, topics, promises, relationship context. Alerts on drifting relationships |
| **Calendar** | Bot IS the calendar — events stored in SQLite, no external sync needed |
| **Decision tracker** | Tracks decisions with follow-ups (+14 days); surfaces past similar decisions and outcomes |
| **Auto-diary** | Daily diary entry generated from interactions, mood analysis, relationship alerts |
| **Smart questions** | Midday proactive question to fill knowledge gaps about your life |
| **Morning briefing** | Weather + events + goals prompt + promises + birthdays + relevant memories |
| **Evening summary** | What happened, goal review, what's tomorrow, auto-diary entry |
| **Weekly review** | Behavioral patterns, promise tracking, habit progress, insights saved as learnings |
| **Mood tracking** | Keyword-based Russian sentiment detection; 3-day trend visible to LLM in every conversation |
| **Theme clusters** | Automatically groups recent memories by person/topic to show what's on your mind |
| **Weather monitoring** | Alerts when rain appears in forecast |
| **Birthday alerts** | Upcoming birthdays with 7-day deduplication per contact |
| **Habit tracking** | Daily habit log with weekly review |
| **Quiet hours** | Proactive messages respect `PROFILE_QUIET_HOURS` (reminders still fire) |
| **Notification limit** | Daily cap via `PROFILE_NOTIFY_LIMIT` (default 10) |
| **Feedback loop** | Thumbs up/down on every response; used in weekly analysis |

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **LLM** | Claude Sonnet 4.6 (`anthropic` SDK) | All chat, tool use, briefings, weekly review |
| **STT** | Groq Whisper large-v3 (`groq` SDK) | Voice → text, optimized for Russian |
| **Embeddings** | Voyage AI voyage-3 (`voyageai` SDK) | Semantic memory search |
| **Vector search** | numpy vectorized cosine similarity | Batch matrix ops + O(n) partial sort for top-k |
| **Database** | SQLite (WAL mode) | Single file: vectors, events, contacts, goals, everything |
| **Bot** | python-telegram-bot v22 | Async handlers for voice, text, photo, forwards. Long polling |
| **Scheduler** | APScheduler (AsyncIO) | 8 proactive scheduled jobs (each toggleable via config) |
| **Weather** | Open-Meteo (free, no key) | Forecast via `httpx` |
| **Language** | Python 3.10+ | Fully async (asyncio) |

## Quick Start

### Option A: From pre-built image (recommended)

```bash
# 1. Create a directory and download configs
mkdir mindsecretary && cd mindsecretary
curl -LO https://raw.githubusercontent.com/ginkida/mindsecretary/main/.env.example
curl -LO https://raw.githubusercontent.com/ginkida/mindsecretary/main/docker-compose.yaml
mkdir -p config
curl -Lo config/profile.yaml https://raw.githubusercontent.com/ginkida/mindsecretary/main/config/profile.yaml
curl -Lo config/settings.yaml https://raw.githubusercontent.com/ginkida/mindsecretary/main/config/settings.yaml

# 2. Configure
cp .env.example .env
# Edit .env — fill in API keys and profile

# 3. Run
docker compose up -d
```

No webhooks, public IP, or SSL needed — the bot uses long polling.

### Option B: Build from source

```bash
git clone https://github.com/ginkida/mindsecretary.git
cd mindsecretary
cp .env.example .env
# Edit .env — fill in API keys and profile
docker compose up -d --build
```

### Required Environment Variables

```env
ANTHROPIC_API_KEY=...          # https://console.anthropic.com
GROQ_API_KEY=...               # https://console.groq.com
VOYAGE_API_KEY=...             # https://dash.voyageai.com
TELEGRAM_TOKEN=...             # @BotFather in Telegram
TELEGRAM_USER_ID=...           # @userinfobot in Telegram
PROFILE_NAME=...               # Your name (required)
```

### Optional Profile Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROFILE_CITY` | Москва | City for weather |
| `PROFILE_TIMEZONE` | Europe/Moscow | Timezone |
| `PROFILE_HOME_COORDS` | 55.7558,37.6173 | Home coordinates (lat,lon) |
| `PROFILE_WORK_COORDS` | = home | Work coordinates |
| `PROFILE_WAKE_UP` | 07:00 | Morning briefing time |
| `PROFILE_WORK_START` | 09:00 | Work start |
| `PROFILE_WORK_END` | 18:00 | Work end |
| `PROFILE_SLEEP` | 23:00 | Sleep time |
| `PROFILE_COMMUTE` | метро | Commute method |
| `PROFILE_COMMUTE_MIN` | 45 | Commute minutes |
| `PROFILE_STYLE` | кратко, по делу | Bot communication style |
| `PROFILE_LANGUAGE` | ru | Language |
| `PROFILE_NOTIFY_LIMIT` | 10 | Max proactive notifications/day |
| `PROFILE_QUIET_HOURS` | 23:00,07:00 | Do not disturb window (proactive only; reminders still fire) |
| `PROFILE_PRIORITIES` | здоровье,семья,работа,развитие | Life priorities |
| `PROFILE_DISLIKES` | опаздывать,пустая болтовня,лишние уведомления | Things to avoid |

Profile can also be defined in `config/profile.yaml` (env vars take priority).
Model and proactive job settings are in `config/settings.yaml`.

### Proactive Job Toggles (`config/settings.yaml`)

Each proactive job can be disabled independently:

```yaml
proactive:
  morning_briefing: true
  evening_summary: true
  smart_questions: true
  decision_followups: true
  weekly_review: true
  weather_monitor: true
  birthday_alerts: true
```

Reminder checking is always on (core feature, not toggleable).

### Tunable Thresholds (`config/settings.yaml`)

Intervals, timeouts, and detection thresholds are configurable:

```yaml
tuning:
  reminder_check_minutes: 5       # How often to check for due reminders
  weather_check_minutes: 60       # Weather polling interval
  process_timeout_sec: 90         # Max time for LLM processing per message
  quiet_contact_days: 30          # Days without contact before alert
  quiet_contact_min_mentions: 3   # Min past mentions to trigger alert
  smart_question_min_interactions: 5  # Min interactions before asking questions
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message with usage guide |
| `/stats` | Cost, memory count, contacts, interactions today |
| `/diary` | Last 3 diary entries |
| `/people` | Contact list with last contact dates |
| `/review` | Trigger weekly review on demand |
| `/forget <query>` | Delete the closest matching memory |

## LLM Tools

The LLM has access to 13 tools, called automatically based on message content:

| Tool | Description |
|------|-------------|
| `save_memory` | Store a fact or feeling with category, importance (1-10), related person/date |
| `search_memory` | Semantic search over all memories, optionally filtered by category |
| `create_event` | Calendar event with title, time, location, related person |
| `get_events` | Query events by date range |
| `create_reminder` | Time-triggered reminder with priority |
| `update_contact` | Create/update person: relation, birthday, notes (appended) |
| `get_contacts` | Search contacts by name or relation |
| `get_weather` | Weather forecast (1-7 days) |
| `log_habit` | Mark habit done/skipped for a date |
| `track_decision` | Track a decision with context; surfaces similar past decisions |
| `resolve_decision` | Close a tracked decision with outcome and sentiment |
| `set_daily_goal` | Create a goal for today with title, priority, description |
| `complete_daily_goal` | Mark a daily goal as completed/skipped/partial with reflection |

Memory categories: `contact`, `health`, `work`, `personal`, `promise`, `preference`, `location`, `learning`, `emotional`.

## Scheduled Jobs

All proactive jobs respect quiet hours and the daily notification limit.
Each can be toggled off in `config/settings.yaml`. Reminders always fire.

| Job | Schedule | Uses LLM | Description |
|-----|----------|----------|-------------|
| `reminder_check` | Every 5 min | No | Send due reminders (bypasses quiet hours) |
| `birthday_check` | Daily 09:00 | No | Alert on upcoming birthdays (7-day dedup per contact) |
| `weather_monitor` | Every 60 min | No | Alert when rain appears in forecast |
| `morning_prompt` | At `PROFILE_WAKE_UP` | Yes | Briefing + "What are your goals today?" |
| `smart_question` | Daily 13:00 | Yes | One targeted question to fill knowledge gaps |
| `decision_followup` | Daily 10:00 | No | Follow up on past decisions (+14 day repeat) |
| `evening_prompt` | Daily 21:00 | Yes | Day summary + goal review + auto-diary entry |
| `weekly_review` | Sunday 20:00 | Yes | Pattern analysis → insights → learnings saved to memory |

## Database Schema

Single SQLite file: `data/mindsecretary.db`

| Table | Key columns | Purpose |
|-------|-------------|---------|
| `memories` | content, embedding (BLOB), category, importance, related_person, status | Semantic memory with Voyage AI vectors |
| `events` | title, start_at, end_at, location, related_person, recurring | Calendar (bot is the calendar) |
| `reminders` | text, trigger_at, priority, status | Time-triggered reminders |
| `contacts` | name, relation, birthday, notes, last_contact, mention_count, last_birthday_alert | Personal CRM |
| `interactions` | direction, message_type, content, feedback, voice_duration_sec | Full message log (in/out) |
| `decisions` | description, context, outcome, outcome_sentiment, follow_up_at, status | Decision tracking with follow-ups |
| `daily_goals` | date, title, description, priority, status, reflection, completed_at | Daily goal tracking |
| `diary_entries` | date, content, mood, people | Auto-generated daily diary |
| `preferences` | key, value, confidence, source | Learned user preferences |
| `habits` | name, target | Habit definitions |
| `habit_log` | habit_id, date, done, notes | Daily habit tracking |
| `api_costs` | provider, input_tokens, output_tokens, cost_usd | Token usage and cost |

## Project Structure

```
mindsecretary/
├── pyproject.toml                 # Package config, dependencies, entry point
├── Dockerfile                     # Python 3.12-slim, non-root user
├── docker-compose.yaml            # Single service with data/config volumes
├── .env.example                   # Template for API keys and profile
├── SECURITY.md                    # Security policy
├── config/
│   ├── profile.yaml               # User profile (YAML fallback, env vars override)
│   └── settings.yaml              # Model, STT, embedding, memory, proactive toggles
├── scripts/
│   ├── backup.sh                  # SQLite online backup with 30-day rotation
│   └── healthcheck.py             # Docker healthcheck: process alive + DB accessible
├── .github/
│   ├── dependabot.yml             # Weekly pip dependency updates
│   └── workflows/
│       └── docker-publish.yml     # Build & push to ghcr.io on release
└── src/mindsecretary/
    ├── app.py                     # Entry point: wires all components, runs bot
    ├── core/
    │   ├── brain.py               # Orchestrator: context → LLM → tool execution loop
    │   ├── config.py              # Profile (env/yaml) + Settings dataclasses
    │   ├── database.py            # SQLite: all tables, CRUD, cost tracking, goals
    │   ├── enums.py               # Status, Priority, Sentiment, Feedback enums
    │   └── memory.py              # Voyage AI embeddings + vectorized cosine search
    ├── llm/
    │   ├── client.py              # AnthropicClient (Claude Sonnet via Anthropic SDK)
    │   ├── router.py              # Model router (single client wrapper)
    │   ├── prompts.py             # System prompts: main, briefing, evening, weekly, diary
    │   └── tools.py               # 13 tool definitions + argument validation + executor
    ├── voice/
    │   └── stt.py                 # Groq Whisper large-v3 speech-to-text
    ├── interfaces/
    │   └── telegram.py            # Telegram bot: voice/text/photo/forward handlers, commands
    ├── proactive/
    │   ├── scheduler.py           # APScheduler: 8 jobs, quiet hours, notification limit
    │   ├── briefing.py            # Morning/evening/diary generation via LLM
    │   ├── monitor.py             # Reminder checks (no LLM)
    │   └── smart_questions.py     # Midday knowledge-gap questions via LLM
    ├── learning/
    │   ├── tracker.py             # Feedback tracking (positive/negative/response time)
    │   ├── reflection.py          # Weekly review: patterns → insights → learnings to memory
    │   └── mood.py                # Russian keyword-based mood analysis + contact frequency
    └── integrations/
        └── weather.py             # Open-Meteo API client (free, no key)
tests/
├── conftest.py                    # Shared fixtures (temp database)
├── test_brain.py                  # Sanitization tests
├── test_database.py               # CRUD + timestamp format tests
├── test_mood.py                   # Mood analysis tests
├── test_scheduler.py              # Quiet hours logic tests
└── test_tools.py                  # Argument sanitization tests
```

## Operations

```bash
docker compose up -d               # Start
docker compose logs -f              # Logs
docker compose down                 # Stop
docker compose pull && docker compose up -d  # Update to latest
./scripts/backup.sh                 # Backup SQLite (safe while running)
```

## Security

- **Auth**: Telegram user ID check on every handler (including callback queries)
- **SQL injection**: Parameterized queries everywhere, column name whitelists for dynamic queries, LIKE wildcard escaping
- **Prompt injection**: Instruction-like patterns (EN + RU) stripped from memory/event/goal content before system prompt injection. Role-lock in system prompt against memory-based hijacking.
- **Input limits**: Voice 25 MB / 10 min, photo 10 MB, text 10K chars, `/forget` query 500 chars, processing timeout configurable (default 90s)
- **Proactive limits**: Quiet hours enforcement, daily notification cap, birthday dedup (7 days)
- **Docker**: Non-root user, read-only config volume
- **Secrets**: `.env` gitignored, no secrets in logs
- **GitHub**: Secret scanning + push protection, Dependabot alerts + auto-fix, branch protection on main

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Cost Estimate

| Provider | Price | Typical monthly cost |
|----------|-------|---------------------|
| Claude Sonnet 4.6 | $3 / $15 per M tokens | ~$1.00 |
| Groq Whisper | $0.0011/min | ~$0.17 |
| Voyage AI | $0.06 per M tokens | ~$0.01 |
| Open-Meteo | Free | $0.00 |
| **Total** | | **~$1.20/month** |

## License

MIT
