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
| **Semantic memory** | Voyage AI embeddings in SQLite; vectorized cosine similarity, automatic dedup (>0.92 threshold), recency decay on old memories |
| **Personal CRM** | Tracks people: last contact, topics, promises, relationship context. Alias fuzzy matching. Alerts on drifting relationships |
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
| **Habit tracking** | Daily habit log with streak tracking and weekly completion rate (`/habits`) |
| **Data export** | `/export` sends all memories, contacts, diary, events as JSON file |
| **Quiet hours** | Proactive messages respect `PROFILE_QUIET_HOURS` (reminders still fire) |
| **Notification limit** | Daily cap via `PROFILE_NOTIFY_LIMIT` (default 10) |
| **Feedback loop** | Thumbs up/down + 📌 pin on every response; used in weekly analysis |
| **Notification awareness** | Shows count in `/start`; warns when approaching daily limit |

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

Prerequisites: Docker + Docker Compose. Nothing else — no webhooks, public IP, or SSL needed (long polling).

### 1. Get API keys (5 minutes)

| Service | Where | Free tier |
|---------|-------|-----------|
| Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) → API Keys | $5 free credit |
| Groq (STT) | [console.groq.com](https://console.groq.com) → API Keys | Generous free tier |
| Voyage AI (embeddings) | [dash.voyageai.com](https://dash.voyageai.com) → API Keys | 200M free tokens |
| Telegram bot | Message [@BotFather](https://t.me/BotFather) → `/newbot` | Free |
| Your Telegram ID | Message [@userinfobot](https://t.me/userinfobot) → it replies with your numeric ID | Free |

### 2. Set up the project (Option A — pre-built image)

```bash
# Create a directory for the bot
mkdir mindsecretary && cd mindsecretary

# Download the compose file + env template + default configs
curl -LO https://raw.githubusercontent.com/ginkida/mindsecretary/main/docker-compose.yaml
curl -LO https://raw.githubusercontent.com/ginkida/mindsecretary/main/.env.example
mkdir -p config data
curl -Lo config/profile.yaml https://raw.githubusercontent.com/ginkida/mindsecretary/main/config/profile.yaml
curl -Lo config/settings.yaml https://raw.githubusercontent.com/ginkida/mindsecretary/main/config/settings.yaml

# Configure
cp .env.example .env
chmod 600 .env              # protect your API keys from other local users
nano .env                   # fill in the 6 required values (see below)

# Start
docker compose up -d
docker compose logs -f      # follow startup logs, ctrl-c to detach
```

### 2. Set up the project (Option B — build from source)

```bash
git clone https://github.com/ginkida/mindsecretary.git
cd mindsecretary
cp .env.example .env
chmod 600 .env
nano .env                   # fill in the 6 required values
docker compose up -d --build
```

### 3. Verify

```bash
docker compose ps                  # mindsecretary should be "running (healthy)"
docker compose logs | tail -30     # look for "Proactive scheduler started with N jobs"
```

Then open Telegram, find your bot, send `/start`. You should see the welcome message with available commands.

If something went wrong, see [Troubleshooting](#troubleshooting) below.

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

Intervals, timeouts, detection thresholds, and safety limits are configurable:

```yaml
tuning:
  reminder_check_minutes: 5       # How often to check for due reminders
  weather_check_minutes: 60       # Weather polling interval
  process_timeout_sec: 90         # Max time for LLM processing per message
  quiet_contact_days: 30          # Days without contact before alert
  quiet_contact_min_mentions: 3   # Min past mentions to trigger alert
  smart_question_min_interactions: 5  # Min interactions before midday question

  # Safety limits
  daily_cost_limit_usd: 5.0       # Bot refuses LLM calls once daily spend crosses this
  rate_limit_per_minute: 20       # Max inbound messages/min (DoS protection)
  data_retention_days: 90         # Weekly cleanup deletes interactions/api_costs older than this
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message with usage guide |
| `/stats` | Cost, memory count, contacts, interactions today |
| `/diary` | Last 3 diary entries |
| `/people` | Contact list with last contact dates |
| `/review` | Trigger weekly review on demand |
| `/goals` | Today's daily goals with status |
| `/habits` | Habit streaks and weekly completion rate |
| `/search <query>` | Direct semantic memory search with scores |
| `/memory [query]` | Inspect what's remembered — match reason, confidence, source |
| `/loops` | Show current open items (overdue reminders, upcoming events, pending goals, due decisions) |
| `/export` | Export all data as JSON file |
| `/undo` | Restore the last deleted memory |
| `/forget <query>` | Delete the closest matching memory (with confirmation) |

## LLM Tools

The LLM has 16 tools at its disposal (15 custom + 1 Anthropic-native), called automatically based on message content:

| Tool | Description |
|------|-------------|
| `save_memory` | Store a fact or feeling with category, importance (1-10), related person/date |
| `search_memory` | Semantic search over all memories, optionally filtered by category |
| `get_recent_memories` | List recent memories with source info (used for "what do you remember?") |
| `get_open_loops` | Snapshot of overdue reminders, upcoming events, pending goals, due decisions |
| `create_event` | Calendar event with title, time, location, related person |
| `get_events` | Query events by date range |
| `create_reminder` | Time-triggered reminder with optional recurrence (daily/weekly/monthly) |
| `update_contact` | Create/update person: relation, birthday, notes (appended) |
| `get_contacts` | Search contacts by name or relation |
| `get_weather` | Weather forecast (1-7 days) |
| `log_habit` | Mark habit done/skipped for a date |
| `track_decision` | Track a decision with context; surfaces similar past decisions |
| `resolve_decision` | Close a tracked decision with outcome and sentiment |
| `set_daily_goal` | Create a goal for today with title, priority, description |
| `complete_daily_goal` | Mark a daily goal as completed/skipped/partial with reflection |
| `web_search` | Anthropic native server-side web search — for time-sensitive queries (currency, news, live data). Billed separately ($10 / 1000 searches) |

Memory categories: `contact`, `health`, `work`, `personal`, `promise`, `preference`, `location`, `learning`, `emotional`.

Every saved memory carries provenance — `source_type` (text / voice / photo / forward), `source_ref` (pointer to originating interaction), and a `confidence` score by source.

## Scheduled Jobs

All proactive jobs respect quiet hours and the daily notification limit.
Each can be toggled off in `config/settings.yaml`. Reminders always fire.

| Job | Schedule | Uses LLM | Description |
|-----|----------|----------|-------------|
| `reminder_check` | Every 5 min | No | Send due reminders (bypasses quiet hours) |
| `birthday_check` | Daily 09:00 | No | Alert on upcoming birthdays (7-day dedup per contact) |
| `weather_monitor` | Every 60 min | No | Alert when rain appears in forecast |
| `morning_prompt` | At `PROFILE_WAKE_UP` | Yes | Briefing (with open loops) + "What are your goals today?" |
| `smart_question` | Daily 13:00 | Yes | Action nudge (22h dedup) or knowledge-gap question |
| `decision_followup` | Daily 10:00 | No | Follow up on past decisions (+14 day repeat) |
| `evening_prompt` | Daily 21:00 | Yes | Day summary + goal review + auto-diary entry |
| `weekly_review` | Sunday 20:00 | Yes | Pattern analysis → insights → learnings saved to memory |
| `cleanup_old_data` | Sunday 03:00 | No | Delete interactions / api_costs older than `data_retention_days` |

## Database Schema

Single SQLite file: `data/mindsecretary.db`

| Table | Key columns | Purpose |
|-------|-------------|---------|
| `memories` | content, embedding (BLOB), category, importance, related_person, source_type, source_ref, confidence, status | Semantic memory with Voyage AI vectors + provenance |
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
├── Dockerfile                     # Python 3.12-slim, non-root user, MINDSECRETARY_ROOT=/app
├── docker-compose.yaml            # Single service with data/config volumes
├── .env.example                   # Template for API keys and profile
├── SECURITY.md                    # Security policy
├── config/
│   ├── profile.yaml               # User profile (YAML fallback, env vars override)
│   └── settings.yaml              # Model, STT, embeddings, proactive toggles, safety limits
├── migrations/                    # Numbered .sql files, applied via PRAGMA user_version
│   └── README.md                  # Migration convention
├── scripts/
│   ├── backup.sh                  # SQLite online backup with 30-day rotation
│   ├── healthcheck.py             # Docker healthcheck: process alive + DB accessible
│   └── reembed.py                 # Rescue: re-embed memories marked embed_failed / zero-vector
├── .github/
│   ├── dependabot.yml             # Weekly pip dependency updates
│   └── workflows/
│       ├── test.yml               # pytest on push/PR (Python 3.10/3.11/3.12)
│       └── docker-publish.yml     # Build & push to ghcr.io on release
└── src/mindsecretary/
    ├── app.py                     # Entry point: wires all components, runs bot
    ├── core/
    │   ├── brain.py               # Orchestrator: context → LLM → tool execution loop + cost breaker
    │   ├── config.py              # Profile + Settings + AppConfig (project-root resolution)
    │   ├── database.py            # SQLite: all tables, CRUD, cost, cleanup, migrations
    │   ├── enums.py               # Status, Priority, Sentiment, Feedback enums
    │   ├── memory.py              # Voyage embeddings + vectorized cosine + embed_failed quarantine
    │   └── prompt_safety.py       # sanitize_for_context — defense against prompt injection
    ├── llm/
    │   ├── client.py              # AnthropicClient with retry + stop_reason warning
    │   ├── router.py              # Model router (single client wrapper)
    │   ├── prompts.py             # System prompts: main, briefing, evening, weekly, diary
    │   └── tools.py               # 16 tool definitions + argument validation + executor
    ├── voice/
    │   └── stt.py                 # Groq Whisper large-v3, 3x retry with exp-backoff
    ├── interfaces/
    │   └── telegram.py            # Telegram bot: handlers, commands, rate limit
    ├── proactive/
    │   ├── scheduler.py           # APScheduler: 9 jobs, quiet hours, notification limit
    │   ├── briefing.py            # Morning (with open loops) / evening / diary via LLM
    │   ├── monitor.py             # Reminder checks (no LLM)
    │   └── smart_questions.py     # Midday knowledge-gap questions / action nudge
    ├── learning/
    │   ├── tracker.py             # Feedback tracking (positive/negative/response time)
    │   ├── reflection.py          # Weekly review: patterns → insights → learnings to memory
    │   └── mood.py                # Russian keyword-based mood analysis + contact frequency
    └── integrations/
        └── weather.py             # Open-Meteo API client (free, no key)
tests/                             # 129 tests (pytest + pytest-asyncio)
├── conftest.py                    # Shared fixtures (temp database)
├── test_brain.py                  # Sanitization
├── test_database.py               # CRUD + timestamp + cleanup + migrations
├── test_habits.py                 # Habit tracking
├── test_integration.py            # Brain.process end-to-end with mocked LLM
├── test_llm_client.py             # Anthropic client retry + stop_reason
├── test_mood.py                   # Mood analysis
├── test_scheduler.py              # Quiet hours + action nudge
├── test_telegram.py               # Telegram handler auth + rate limit
├── test_tools.py                  # Argument sanitization
└── test_tools_datetime.py         # Datetime validation
```

## Operations

```bash
docker compose up -d                             # Start
docker compose logs -f                           # Follow logs
docker compose logs --tail=100 mindsecretary    # Last 100 log lines
docker compose ps                                # Health status
docker compose down                              # Stop
docker compose pull && docker compose up -d      # Update to latest
docker compose exec mindsecretary python scripts/reembed.py --dry-run   # Rescue failed embeddings
./scripts/backup.sh                              # Backup SQLite (safe while running)
```

## Troubleshooting

### Bot starts but doesn't respond in Telegram

1. Check `TELEGRAM_USER_ID` in `.env` matches your actual numeric ID — message [@userinfobot](https://t.me/userinfobot) to double-check.
2. Check logs: `docker compose logs | grep -i "unauthorized"`. If you see "Unauthorized: user N", the `N` is what you sent — copy it into `TELEGRAM_USER_ID`.
3. Make sure you started a chat with your bot first (send any message). Telegram won't deliver anything until the user initiates.

### `.env` permissions warning at startup

```
.env permissions are 0664 — API keys readable beyond owner. Run: chmod 600 .env
```

That's exactly what to do — `chmod 600 .env`. The warning is loud but non-fatal; API keys in `.env` are world-readable under default umask. Only your user needs to read them.

### `Permission denied` writing to `/app/data` (Linux hosts)

The container's `app` user has UID 1000 by default. If your host user has a different UID, bind mounts on Linux reject writes. Fix by pinning the container UID in your `docker-compose.yaml`:

```yaml
services:
  mindsecretary:
    user: "${UID}:${GID}"   # or explicit numbers like "1001:1001"
    # ... rest unchanged
```

Then `UID=$(id -u) GID=$(id -g) docker compose up -d`. macOS + Docker Desktop handles UID mapping automatically, no changes needed.

### "Cost limit hit" messages when you haven't spent much

The default daily limit is `$5.00`. If you intentionally want a bigger ceiling, bump it in `config/settings.yaml`:

```yaml
tuning:
  daily_cost_limit_usd: 20.0
```

Then `docker compose restart`. The limit resets at the start of each local day.

### "Too many retries" or STT failures

Voice → text goes through Groq. If Groq is briefly down, STT retries 3x with 1s / 2s / 4s backoff before giving up. The Anthropic LLM call has the same 3x retry. If you see sustained failures: check `docker compose logs | grep -iE "failed|error"` — type names (like `APIConnectionError`, `RateLimitError`) in the logs point to the right fix.

### "no such column: source_type" or similar schema errors

You're running an old DB image against a newer schema. The bot auto-migrates on startup (both the numbered SQL migrations in `migrations/` and an idempotent `ALTER TABLE` in `memory.py`). If the error persists, your container isn't restarting cleanly — `docker compose down && docker compose pull && docker compose up -d`.

### How do I wipe and start over?

```bash
docker compose down
rm -rf data/
docker compose up -d
```

Your `.env` and `config/` are preserved; only the SQLite file + logs are deleted. Make a backup first (`./scripts/backup.sh`) if there's anything worth keeping.

## Security

- **Auth**: Telegram user ID check on every handler (including callback queries)
- **Rate limit**: Per-minute cap on inbound messages (20/min default) — protects against a compromised Telegram session
- **Cost circuit breaker**: Bot refuses LLM calls once daily spend crosses `daily_cost_limit_usd` (default $5/day)
- **SQL injection**: Parameterized queries everywhere, column name whitelists for dynamic queries, LIKE wildcard escaping
- **Prompt injection**: Instruction-like patterns (EN + RU) stripped from memory/event/goal content before system-prompt injection. Role-lock stanza in MAIN, BRIEFING, EVENING, WEEKLY, DIARY, and GAPS prompts. Every user-origin field passes through `sanitize_for_context`.
- **Input limits**: Voice 25 MB / 10 min, transcript 30K chars, photo 10 MB, text 10K chars, `/forget` query 500 chars, tool-calls capped at 10 per round, LLM rounds capped at 5, processing timeout 90s. STT and LLM both retry 3x with exp-backoff on transient errors
- **Proactive limits**: Quiet hours enforcement, daily notification cap, birthday dedup (7 days), midday action nudge dedup (22h)
- **Docker**: Non-root user, read-only config volume, explicit `MINDSECRETARY_ROOT=/app`
- **Secrets**: `.env` gitignored, startup warning if `.env` is world/group-readable, logs use `type(e).__name__` instead of `str(e)` to avoid leaking request IDs / URLs / coords
- **Data retention**: Weekly cleanup job deletes interactions and api_costs older than `data_retention_days` (default 90)
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
