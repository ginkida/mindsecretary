# MindSecretary

Voice-first personal AI secretary as a Telegram bot. Speak freely — it remembers everything, creates events, tracks promises, prepares you for meetings, and analyzes your week.

## Overview

MindSecretary is an external brain that connects facts from your life and surfaces insights you'd miss on your own. It turns 20-second voice messages into structured memories, events, reminders, and contact updates — all stored locally in a single SQLite database.

### Core Capabilities

- **Voice-first input** — speak naturally; Groq Whisper transcribes, MiniMax M-2.7 extracts every fact, event, and promise
- **Semantic memory** — Voyage AI embeddings stored in SQLite; cosine similarity search finds relevant memories even when phrasing differs
- **Personal CRM** — tracks people in your life: last contact, topics discussed, promises made, relationship context. Generates pre-meeting briefings
- **Proactive scheduler** — morning briefing, evening summary, weekly review, reminders, birthday alerts, weather monitoring
- **Behavioral insights** — weekly analysis detects patterns (e.g. "you're late on Mondays", "you forget errands-on-the-way")
- **Learning loop** — weekly reflection extracts learnings from interactions, stores them as memories that influence future responses

### Architecture

```
Telegram (voice / text / forwarded messages)
    │
    ├─ Groq Whisper ─── speech-to-text
    │
    ▼
  Brain (orchestrator)
    │
    ├─ Voyage AI ─────── embed query ──→ SQLite (cosine search over BLOB embeddings)
    │
    ├─ MiniMax M-2.7 ── LLM with tool use (OpenAI-compatible API)
    │   ├── save_memory       — store facts with category + importance + related person/date
    │   ├── search_memory     — semantic search over all memories
    │   ├── create_event      — calendar (bot IS the calendar, no external sync)
    │   ├── get_events        — query events by date range
    │   ├── create_reminder   — time-triggered reminders
    │   ├── update_contact    — upsert contact with notes, relation, birthday
    │   ├── get_contacts      — search contacts
    │   ├── get_weather       — Open-Meteo forecast
    │   └── log_habit         — habit tracking
    │
    ├─ ModelRouter ───── primary (MiniMax) / fallback (Anthropic Claude, optional)
    │
    └─ APScheduler ──── 6 scheduled jobs (see below)
```

### Data Flow

```
Voice message ──→ Groq Whisper (STT) ──→ transcript
Text message  ──→ (direct)
Forwarded msg ──→ extract sender + content
                            │
                            ▼
                    Brain.process()
                      1. Voyage embed(message) → SQLite cosine search → relevant memories
                      2. SQLite query → today's events, pending reminders
                      3. Build system prompt with profile + memories + events + recent messages
                      4. MiniMax M-2.7 chat completion with tool definitions
                      5. Execute tool calls (save_memory, create_event, etc.)
                      6. Repeat until no more tool calls (max 5 rounds)
                      7. Return final text → Telegram
                      8. Log interaction + API cost
```

### Database Schema (single SQLite file)

| Table | Purpose |
|-------|---------|
| `memories` | Semantic memory with Voyage embeddings (BLOB), category, importance, related person/date |
| `events` | Calendar events (bot is the calendar) |
| `reminders` | Time-triggered reminders with priority and status |
| `contacts` | Personal CRM: name, aliases, relation, birthday, notes, last contact, mention count |
| `interactions` | Full log of all messages (in/out) with type, feedback, response time |
| `preferences` | Learned preferences with confidence scores |
| `habits` / `habit_log` | Habit definitions and daily tracking |
| `api_costs` | Token usage and cost per provider |

### Scheduled Jobs

| Job | Schedule | Engine | Description |
|-----|----------|--------|-------------|
| `reminder_check` | Every 5 min | No LLM | `if now >= trigger_at` → send |
| `birthday_check` | Daily 09:00 | No LLM | SQL query on contacts.birthday |
| `weather_monitor` | Every 60 min | No LLM | Open-Meteo diff → alert on new rain |
| `morning_prompt` | At wake_up time | MiniMax | Briefing: weather + events + promises + memories |
| `evening_prompt` | 21:00 | MiniMax | Day summary + tomorrow preview |
| `weekly_review` | Sunday 20:00 | MiniMax | Pattern analysis → insights → learnings saved to memory |

### Security

- Input validation: tool arguments sanitized (length limits, category whitelist, type checks)
- SQL injection protection: column name whitelist in dynamic queries, parameterized everywhere
- Prompt injection mitigation: instruction-like patterns stripped from memory content before system prompt injection
- DoS protection: voice file size limit (25MB), duration limit (10min), download timeout (30s), processing timeout (60s), text length limit (10K chars)
- Single-user auth: Telegram user ID check on every handler
- Secrets: `.env` file gitignored, no secrets in logs (only error types logged, no content)
- `<think>` tag stripping: MiniMax M-2.7 reasoning tokens removed from user-facing responses

### Cost

| Provider | Price | Typical monthly cost |
|----------|-------|---------------------|
| MiniMax M-2.7 | $0.30/$1.20 per M tokens | ~$0.42 |
| Claude Sonnet (optional fallback) | $3/$15 per M tokens | ~$0.57 |
| Groq Whisper | $0.0011/min | ~$0.17 |
| Voyage AI | $0.06/M tokens | ~$0.01 |
| **Total** | | **~$1.20/month** |

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM (primary) | MiniMax M-2.7 | $0.30/$1.20 per M tokens, OpenAI-compatible API, tool use |
| LLM (fallback) | Claude Sonnet 4.6 | Optional, better Russian, used for weekly analysis |
| STT | Groq Whisper large-v3 | $0.0011/min, fast, excellent Russian |
| Embeddings | Voyage AI voyage-3 | Best retrieval embeddings, near-free |
| Vector search | numpy cosine similarity | <10ms for thousands of records, no extra dependency |
| Database | SQLite (WAL mode) | Single file, embeddings as BLOB, zero config |
| Bot framework | python-telegram-bot v21+ | Async, voice/text/forward handlers |
| Scheduler | APScheduler (AsyncIO) | Cron + interval jobs |
| Weather | Open-Meteo | Free, no API key needed |
| HTTP | httpx | Async HTTP client |

## Quick Start

```bash
git clone https://github.com/ginkida/mindsecretary.git
cd mindsecretary
cp .env.example .env
# Fill in .env: API keys + profile
docker compose up -d
```

## Configuration

Everything via `.env` — no config files to edit:

```env
# API Keys
MINIMAX_API_KEY=...
GROQ_API_KEY=...
VOYAGE_API_KEY=...
TELEGRAM_TOKEN=...          # from @BotFather
TELEGRAM_USER_ID=...        # from @userinfobot
ANTHROPIC_API_KEY=          # optional, leave empty for MiniMax-only

# Profile
PROFILE_NAME=John
PROFILE_CITY=London
PROFILE_TIMEZONE=Europe/London
PROFILE_HOME_COORDS=51.5074,-0.1278
PROFILE_WAKE_UP=07:30
PROFILE_WORK_START=09:00
PROFILE_WORK_END=18:00
PROFILE_STYLE=concise and practical
```

<details>
<summary>All profile variables</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `PROFILE_NAME` | — | Your name (required) |
| `PROFILE_CITY` | Москва | City |
| `PROFILE_TIMEZONE` | Europe/Moscow | Timezone |
| `PROFILE_HOME_COORDS` | 55.7558,37.6173 | Home coordinates (for weather) |
| `PROFILE_WORK_COORDS` | = home coords | Work coordinates |
| `PROFILE_WAKE_UP` | 07:00 | Wake up time (morning briefing trigger) |
| `PROFILE_WORK_START` | 09:00 | Work start |
| `PROFILE_WORK_END` | 18:00 | Work end |
| `PROFILE_SLEEP` | 23:00 | Sleep time |
| `PROFILE_COMMUTE` | метро | Commute method |
| `PROFILE_COMMUTE_MIN` | 45 | Commute duration (minutes) |
| `PROFILE_STYLE` | кратко, по делу | Bot communication style |
| `PROFILE_LANGUAGE` | ru | Language |
| `PROFILE_NOTIFY_LIMIT` | 5 | Max notifications per day |
| `PROFILE_QUIET_HOURS` | 23:00,07:00 | Do not disturb window |
| `PROFILE_PRIORITIES` | здоровье,семья,работа,развитие | Life priorities |
| `PROFILE_DISLIKES` | опаздывать,пустая болтовня,лишние уведомления | Things to avoid |

</details>

## Usage

### Voice (primary input)

Send a voice message with anything on your mind:

> "Had a doctor's appointment today, need to get an MRI, scheduled for the 20th. Dima called, wants to go fishing this weekend. Wife asked me to buy milk on the way home."

Result: 1 event (MRI), 2 reminders (fishing decision, milk), 1 contact update (Dima), 2 memories saved.

### Text

```
remind me to call mom on Saturday
what do I have tomorrow?
what did I promise Sasha?
schedule a meeting with Petya on Friday at 3pm
```

### Forward

Forward any message from another chat — the bot extracts agreements and facts.

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/stats` | Usage stats: costs, memory count, contacts, interactions today |

## Project Structure

```
mindsecretary/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yaml
├── .env.example
├── config/
│   ├── profile.yaml          # YAML fallback (env vars take priority)
│   └── settings.yaml         # Model and scheduler settings
├── scripts/
│   └── backup.sh             # SQLite online backup with rotation
└── src/mindsecretary/
    ├── app.py                # Entry point: wires everything, runs bot
    ├── core/
    │   ├── brain.py          # Orchestrator: context → LLM → tool loop
    │   ├── config.py         # Profile (env/yaml) + settings loader
    │   ├── database.py       # SQLite: all tables + CRUD + cost tracking
    │   └── memory.py         # Voyage embeddings + cosine search in SQLite
    ├── llm/
    │   ├── client.py         # MiniMaxClient + AnthropicClient + <think> stripping
    │   ├── router.py         # Primary/fallback model routing
    │   ├── prompts.py        # System prompts (main, briefing, evening, weekly, pre-meeting)
    │   └── tools.py          # 9 tool definitions + validation + async executor
    ├── voice/
    │   └── stt.py            # Groq Whisper STT
    ├── interfaces/
    │   └── telegram.py       # Voice/text/forward handlers + /stats
    ├── proactive/
    │   ├── scheduler.py      # APScheduler: 6 jobs
    │   ├── briefing.py       # Morning/evening briefing generation
    │   └── monitor.py        # Reminder/birthday/weather checks
    ├── learning/
    │   ├── tracker.py        # Feedback tracking (positive/negative/response time)
    │   └── reflection.py     # Weekly analysis → extract learnings → save to memory
    └── integrations/
        └── weather.py        # Open-Meteo API client
```

## Operations

```bash
# Start
docker compose up -d

# Logs
docker compose logs -f

# Stop
docker compose down

# Backup
./scripts/backup.sh

# Update
git pull && docker compose up -d --build
```

## API Keys

| Service | Get it at |
|---------|-----------|
| MiniMax | https://platform.minimax.io |
| Groq | https://console.groq.com |
| Voyage AI | https://dash.voyageai.com |
| Telegram Bot Token | @BotFather in Telegram |
| Telegram User ID | @userinfobot in Telegram |

## License

MIT
