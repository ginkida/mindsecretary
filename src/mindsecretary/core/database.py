from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from . import tz_now
from .enums import Priority, Status

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path, timezone: str | None = None,
                 migrations_dir: Path | None = None):
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._timezone = timezone
        self._init_tables()
        if migrations_dir is None:
            migrations_dir = Path(__file__).resolve().parents[3] / "migrations"
        self._apply_migrations(migrations_dir)

    def _apply_migrations(self, migrations_dir: Path):
        """Apply pending SQL migrations from `migrations_dir` in lexical order.

        The current migration level is tracked via SQLite's PRAGMA user_version
        (a simple integer stored in the DB header). Each applied file bumps
        it by 1. New installs: `_init_tables` creates the current schema,
        then migrations apply on top.
        """
        if not migrations_dir.exists():
            return
        files = sorted(p for p in migrations_dir.glob("*.sql") if p.is_file())
        if not files:
            return
        current = self.db.execute("PRAGMA user_version").fetchone()[0]
        for idx, path in enumerate(files, start=1):
            if idx <= current:
                continue
            logger.info("Applying migration %s", path.name)
            self.db.executescript(path.read_text(encoding="utf-8"))
            self.db.execute(f"PRAGMA user_version = {idx}")
            self.db.commit()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
                title TEXT NOT NULL,
                start_at TEXT NOT NULL,
                end_at TEXT,
                location TEXT,
                description TEXT,
                related_person TEXT,
                recurring TEXT,
                source TEXT DEFAULT 'voice',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_at);

            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
                text TEXT NOT NULL,
                trigger_at TEXT NOT NULL,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_reminders_trigger ON reminders(trigger_at);
            CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status);

            CREATE TABLE IF NOT EXISTS contacts (
                id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
                name TEXT NOT NULL,
                aliases TEXT,
                relation TEXT,
                birthday TEXT,
                phone TEXT,
                notes TEXT,
                last_contact TEXT,
                contact_frequency INTEGER,
                mention_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_contacts_name ON contacts(name);

            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
                timestamp TEXT DEFAULT (datetime('now')),
                direction TEXT,
                message_type TEXT,
                content TEXT NOT NULL,
                voice_duration_sec REAL,
                feedback TEXT,
                feedback_at TEXT,
                read_at TEXT,
                response_time_sec REAL,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_interactions_ts_type ON interactions(timestamp, message_type, direction);

            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                source TEXT DEFAULT 'default',
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS habits (
                id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
                name TEXT NOT NULL UNIQUE,
                target TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS habit_log (
                habit_id TEXT REFERENCES habits(id),
                date TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 1,
                notes TEXT,
                PRIMARY KEY (habit_id, date)
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
                description TEXT NOT NULL,
                context TEXT,
                outcome TEXT,
                outcome_sentiment TEXT,
                follow_up_at TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);
            CREATE INDEX IF NOT EXISTS idx_decisions_followup ON decisions(follow_up_at);

            CREATE TABLE IF NOT EXISTS diary_entries (
                date TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                mood TEXT,
                people TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_goals (
                id TEXT PRIMARY KEY DEFAULT (hex(randomblob(8))),
                date TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                reflection TEXT,
                completed_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_goals_date ON daily_goals(date);
            CREATE INDEX IF NOT EXISTS idx_goals_status ON daily_goals(status);

            CREATE TABLE IF NOT EXISTS api_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                provider TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_costs_ts ON api_costs(timestamp);
        """)
        self._run_migrations()
        self.db.commit()

    def _run_migrations(self):
        """Idempotent schema migrations (add columns to existing tables)."""
        migrations = [
            "ALTER TABLE contacts ADD COLUMN last_birthday_alert TEXT",
            "ALTER TABLE reminders ADD COLUMN recurrence TEXT",
        ]
        for sql in migrations:
            try:
                self.db.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists

    def _now(self) -> datetime:
        return tz_now(self._timezone)

    # --- Events ---

    def create_event(self, title: str, start_at: str, end_at: str | None = None,
                     location: str | None = None, description: str | None = None,
                     related_person: str | None = None) -> dict:
        cur = self.db.execute(
            "INSERT INTO events (title, start_at, end_at, location, description, related_person) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING *",
            (title, start_at, end_at, location, description, related_person),
        )
        row = cur.fetchone()
        self.db.commit()
        return dict(row)

    def get_events(self, date_from: str, date_to: str | None = None) -> list[dict]:
        if date_to is None:
            date_to = date_from
        rows = self.db.execute(
            "SELECT * FROM events WHERE date(start_at) >= date(?) AND date(start_at) <= date(?) "
            "ORDER BY start_at",
            (date_from, date_to),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Reminders ---

    def create_reminder(self, text: str, trigger_at: str,
                        priority: str = "medium",
                        recurrence: str | None = None) -> dict:
        cur = self.db.execute(
            "INSERT INTO reminders (text, trigger_at, priority, recurrence) "
            "VALUES (?, ?, ?, ?) RETURNING *",
            (text, trigger_at, priority, recurrence),
        )
        row = cur.fetchone()
        self.db.commit()
        return dict(row)

    def get_pending_reminders(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM reminders WHERE status = 'pending' ORDER BY trigger_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_due_reminders(self) -> list[dict]:
        now = self._now().strftime(self._SQL_TS_FMT)
        rows = self.db.execute(
            "SELECT * FROM reminders WHERE status = 'pending' AND trigger_at <= ?",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]

    _RECURRENCE_DELTAS = {
        "daily": timedelta(days=1),
        "weekly": timedelta(weeks=1),
        "monthly": timedelta(days=30),
    }

    def mark_reminder_sent(self, reminder_id: str):
        row = self.db.execute(
            "SELECT text, trigger_at, priority, recurrence FROM reminders WHERE id = ?",
            (reminder_id,),
        ).fetchone()
        self.db.execute(
            "UPDATE reminders SET status = 'sent' WHERE id = ?", (reminder_id,)
        )
        # Auto-create next occurrence for recurring reminders
        if row and row["recurrence"] in self._RECURRENCE_DELTAS:
            delta = self._RECURRENCE_DELTAS[row["recurrence"]]
            try:
                old_trigger = datetime.fromisoformat(
                    row["trigger_at"].replace(" ", "T")
                )
                next_trigger = old_trigger + delta
                self.create_reminder(
                    row["text"], next_trigger.strftime(self._SQL_TS_FMT),
                    row["priority"], row["recurrence"],
                )
            except (ValueError, TypeError):
                pass  # Can't parse trigger_at — skip recurrence
        self.db.commit()

    # --- Contacts ---

    _CONTACT_FIELDS = frozenset({
        "relation", "birthday", "phone", "notes", "aliases",
        "mention_count", "last_contact", "updated_at",
    })

    @staticmethod
    def _escape_like(s: str | None) -> str:
        if not s:
            return ""
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def upsert_contact(self, name: str, relation: str | None = None,
                       birthday: str | None = None, notes: str | None = None,
                       phone: str | None = None) -> dict:
        existing = self.db.execute(
            "SELECT * FROM contacts WHERE lower(name) = lower(?)", (name,)
        ).fetchone()
        # Also check aliases (comma-separated) for fuzzy matching
        if not existing:
            escaped = self._escape_like(name.lower())
            existing = self.db.execute(
                "SELECT * FROM contacts WHERE lower(aliases) LIKE ? ESCAPE '\\'",
                (f"%{escaped}%",),
            ).fetchone()

        if existing:
            existing = dict(existing)
            updates: dict[str, str | int] = {}
            if relation:
                updates["relation"] = relation
            if birthday:
                updates["birthday"] = birthday
            if phone:
                updates["phone"] = phone
            if notes:
                old_notes = existing.get("notes") or ""
                if notes not in old_notes:
                    updates["notes"] = f"{old_notes}\n{notes}".strip()
            updates["mention_count"] = (existing.get("mention_count") or 0) + 1
            updates["last_contact"] = self._now().strftime(self._SQL_TS_FMT)
            updates["updated_at"] = self._now().strftime(self._SQL_TS_FMT)

            if updates:
                # Validate column names against whitelist to prevent SQL injection
                safe_keys = [k for k in updates if k in self._CONTACT_FIELDS]
                safe_updates = {k: updates[k] for k in safe_keys}
                sets = ", ".join(f"{k} = ?" for k in safe_updates)
                vals = list(safe_updates.values()) + [existing["id"]]
                self.db.execute(f"UPDATE contacts SET {sets} WHERE id = ?", vals)
                self.db.commit()

            return {**existing, **updates}

        cur = self.db.execute(
            "INSERT INTO contacts (name, relation, birthday, phone, notes, last_contact) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING *",
            (name, relation, birthday, phone, notes,
             self._now().strftime(self._SQL_TS_FMT)),
        )
        row = cur.fetchone()
        self.db.commit()
        return dict(row)

    def get_contacts(self, query: str) -> list[dict]:
        escaped = self._escape_like(query.lower())
        rows = self.db.execute(
            "SELECT * FROM contacts WHERE lower(name) LIKE ? ESCAPE '\\' "
            "OR lower(relation) LIKE ? ESCAPE '\\' "
            "ORDER BY mention_count DESC",
            (f"%{escaped}%", f"%{escaped}%"),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_upcoming_birthdays(self, days: int = 7,
                               skip_recent_alerts: bool = False) -> list[dict]:
        today = self._now()
        dates = []
        for i in range(days):
            d = today + timedelta(days=i)
            dates.append(d.strftime("%m-%d"))

        placeholders = ",".join("?" * len(dates))
        where_extra = ""
        if skip_recent_alerts:
            where_extra = (
                " AND (last_birthday_alert IS NULL "
                "OR last_birthday_alert < datetime('now', '-7 days'))"
            )
        rows = self.db.execute(
            f"SELECT * FROM contacts WHERE substr(birthday, -5) IN ({placeholders}) "
            f"AND birthday IS NOT NULL{where_extra}",
            dates,
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_birthday_alerted(self, contact_id: str):
        self.db.execute(
            "UPDATE contacts SET last_birthday_alert = datetime('now') WHERE id = ?",
            (contact_id,),
        )
        self.db.commit()

    # --- Interactions ---

    def log_interaction(self, direction: str, message_type: str, content: str,
                        voice_duration_sec: float | None = None,
                        metadata: dict | None = None) -> str:
        cur = self.db.execute(
            "INSERT INTO interactions (direction, message_type, content, "
            "voice_duration_sec, metadata) VALUES (?, ?, ?, ?, ?) RETURNING id",
            (direction, message_type, content, voice_duration_sec,
             json.dumps(metadata) if metadata else None),
        )
        row = cur.fetchone()
        self.db.commit()
        return row["id"]

    # SQL-compatible timestamp format: matches SQLite's datetime('now') which
    # produces space-separated strings like '2026-04-11 14:30:00'. Python's
    # datetime.isoformat() produces T-separated strings, which compare
    # LEXICOGRAPHICALLY WRONG against space-separated storage
    # (' ' < 'T' in ASCII, so Python isoformat is always "less than" any
    # same-minute SQL timestamp). Always use strftime for SQL params.
    _SQL_TS_FMT = "%Y-%m-%d %H:%M:%S"

    def get_interactions(self, since: datetime | None = None,
                         until: datetime | None = None,
                         message_type: str | None = None,
                         limit: int = 100) -> list[dict]:
        where = "WHERE 1=1"
        params: list = []
        if since:
            where += " AND timestamp >= ?"
            params.append(since.strftime(self._SQL_TS_FMT))
        if until:
            where += " AND timestamp <= ?"
            params.append(until.strftime(self._SQL_TS_FMT))
        if message_type:
            where += " AND message_type = ?"
            params.append(message_type)
        rows = self.db.execute(
            f"SELECT * FROM interactions {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_messages(self, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT direction, content, timestamp FROM interactions "
            "WHERE message_type IN ('voice', 'text', 'forward', 'chat') "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def count_notifications_today(self) -> int:
        today = self._now().strftime("%Y-%m-%d")
        row = self.db.execute(
            "SELECT COUNT(*) as cnt FROM interactions "
            "WHERE direction = 'out' AND message_type = 'notification' "
            "AND date(timestamp) = ?",
            (today,),
        ).fetchone()
        return row["cnt"]

    # --- Preferences ---

    def get_preference(self, key: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def set_preference(self, key: str, value: str,
                       confidence: float = 0.5, source: str = "default"):
        self.db.execute(
            "INSERT INTO preferences (key, value, confidence, source) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=?, confidence=?, source=?, "
            "updated_at=datetime('now')",
            (key, value, confidence, source, value, confidence, source),
        )
        self.db.commit()

    # --- Habits ---

    def log_habit(self, habit_name: str, done: bool,
                  date: str | None = None, notes: str | None = None) -> dict:
        date = date or self._now().strftime("%Y-%m-%d")

        habit = self.db.execute(
            "SELECT id FROM habits WHERE lower(name) = lower(?)", (habit_name,)
        ).fetchone()

        if not habit:
            habit = self.db.execute(
                "INSERT INTO habits (name) VALUES (?) RETURNING id", (habit_name,)
            ).fetchone()

        self.db.execute(
            "INSERT INTO habit_log (habit_id, date, done, notes) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(habit_id, date) DO UPDATE SET done=?, notes=?",
            (habit["id"], date, int(done), notes, int(done), notes),
        )
        self.db.commit()
        return {"habit": habit_name, "date": date, "done": done}

    def get_habit_stats(self) -> list[dict]:
        """Get all habits with current streak and 7-day completion rate."""
        habits = self.db.execute(
            "SELECT id, name FROM habits ORDER BY name"
        ).fetchall()
        results = []
        today = self._now().strftime("%Y-%m-%d")
        for h in habits:
            # Current streak: consecutive days with done=1, counting back from today
            logs = self.db.execute(
                "SELECT date, done FROM habit_log WHERE habit_id = ? "
                "AND date <= ? ORDER BY date DESC LIMIT 60",
                (h["id"], today),
            ).fetchall()
            streak = 0
            for log in logs:
                if log["done"]:
                    streak += 1
                else:
                    break
            # 7-day rate
            week_start = (self._now() - timedelta(days=6)).strftime("%Y-%m-%d")
            week_logs = self.db.execute(
                "SELECT COUNT(*) as cnt FROM habit_log "
                "WHERE habit_id = ? AND date >= ? AND done = 1",
                (h["id"], week_start),
            ).fetchone()
            week_done = week_logs["cnt"]
            results.append({
                "name": h["name"],
                "streak": streak,
                "week_done": week_done,
                "week_rate": round(week_done / 7 * 100),
            })
        return results

    # --- Decisions ---

    def create_decision(self, description: str, context: str | None = None,
                        follow_up_days: int = 30) -> dict:
        follow_up_at = (self._now() + timedelta(days=follow_up_days)).strftime(
            self._SQL_TS_FMT
        )
        cur = self.db.execute(
            "INSERT INTO decisions (description, context, follow_up_at) "
            "VALUES (?, ?, ?) RETURNING *",
            (description, context, follow_up_at),
        )
        row = cur.fetchone()
        self.db.commit()
        return dict(row)

    def get_pending_decision_followups(self) -> list[dict]:
        now = self._now().strftime(self._SQL_TS_FMT)
        rows = self.db.execute(
            "SELECT * FROM decisions WHERE status = 'pending' AND follow_up_at <= ?",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_decisions(self, limit: int = 10) -> list[dict]:
        """All currently pending decisions, newest first."""
        rows = self.db.execute(
            "SELECT * FROM decisions WHERE status = 'pending' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def push_decision_followup(self, decision_id: str, days: int = 14):
        """Push the next follow-up date forward. Called after sending a follow-up."""
        new_time = (self._now() + timedelta(days=days)).strftime(self._SQL_TS_FMT)
        self.db.execute(
            "UPDATE decisions SET follow_up_at = ? WHERE id = ?",
            (new_time, decision_id),
        )
        self.db.commit()

    def resolve_decision(self, decision_id: str, outcome: str,
                         sentiment: str = "neutral") -> bool:
        cur = self.db.execute(
            "UPDATE decisions SET outcome = ?, outcome_sentiment = ?, "
            "status = 'resolved', resolved_at = datetime('now') WHERE id = ?",
            (outcome, sentiment, decision_id),
        )
        self.db.commit()
        return cur.rowcount > 0

    def resolve_decision_by_hint(self, description_hint: str, outcome: str,
                                 sentiment: str = "neutral") -> dict | None:
        """Find the most recent pending decision matching the hint and resolve it."""
        escaped = self._escape_like(description_hint)
        row = self.db.execute(
            "SELECT id, description FROM decisions WHERE status = 'pending' "
            "AND description LIKE ? ESCAPE '\\' "
            "ORDER BY created_at DESC LIMIT 1",
            (f"%{escaped}%",),
        ).fetchone()
        if not row:
            return None
        self.resolve_decision(row["id"], outcome, sentiment)
        return {"id": row["id"], "description": row["description"]}

    def get_theme_clusters(self, days: int = 30, limit: int = 5) -> list[dict]:
        """Cluster recent high-importance memories by related_person or category.

        Returns top clusters ordered by total importance. Each cluster:
          {"label": str, "count": int}
        where label is a person name (preferred) or category name.
        """
        since = (self._now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self.db.execute(
            "SELECT "
            "  COALESCE(NULLIF(related_person, ''), category) AS label, "
            "  COUNT(*) AS cnt, "
            "  SUM(importance) AS total_importance "
            "FROM memories "
            "WHERE status = 'active' "
            "  AND importance >= 5 "
            "  AND date(created_at) >= date(?) "
            "GROUP BY label "
            "HAVING cnt >= 2 "
            "ORDER BY total_importance DESC, cnt DESC "
            "LIMIT ?",
            (since, limit),
        ).fetchall()
        return [{"label": r["label"], "count": r["cnt"]} for r in rows]

    def get_past_decisions(self, query: str = "", limit: int = 5) -> list[dict]:
        escaped = self._escape_like(query)
        rows = self.db.execute(
            "SELECT * FROM decisions WHERE status = 'resolved' "
            "AND (description LIKE ? ESCAPE '\\' OR context LIKE ? ESCAPE '\\') "
            "ORDER BY created_at DESC LIMIT ?",
            (f"%{escaped}%", f"%{escaped}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Diary ---

    def save_diary_entry(self, date: str, content: str, mood: str | None = None,
                         people: str | None = None):
        self.db.execute(
            "INSERT INTO diary_entries (date, content, mood, people) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET content=?, mood=?, people=?",
            (date, content, mood, people, content, mood, people),
        )
        self.db.commit()

    def get_diary_entries(self, days: int = 7) -> list[dict]:
        since = (self._now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self.db.execute(
            "SELECT * FROM diary_entries WHERE date >= ? ORDER BY date DESC",
            (since,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Daily Goals ---

    def create_daily_goal(self, title: str, description: str | None = None,
                          priority: str = "medium",
                          date: str | None = None) -> dict:
        date = date or self._now().strftime("%Y-%m-%d")
        if priority not in (Priority.HIGH, Priority.MEDIUM, Priority.LOW):
            priority = Priority.MEDIUM
        cur = self.db.execute(
            "INSERT INTO daily_goals (date, title, description, priority) "
            "VALUES (?, ?, ?, ?) RETURNING *",
            (date, title, description, priority),
        )
        row = cur.fetchone()
        self.db.commit()
        return dict(row)

    def get_daily_goals(self, date: str | None = None) -> list[dict]:
        date = date or self._now().strftime("%Y-%m-%d")
        rows = self.db.execute(
            "SELECT * FROM daily_goals WHERE date = ? ORDER BY "
            "CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, "
            "created_at",
            (date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def complete_daily_goal_by_hint(self, hint: str, status: str = "completed",
                                    reflection: str | None = None) -> dict | None:
        """Find today's pending goal by keyword and mark it."""
        today = self._now().strftime("%Y-%m-%d")
        if status not in (Status.COMPLETED, Status.SKIPPED, Status.PARTIAL):
            status = Status.COMPLETED
        escaped = self._escape_like(hint)
        row = self.db.execute(
            "SELECT id, title FROM daily_goals "
            "WHERE date = ? AND status = 'pending' "
            "AND title LIKE ? ESCAPE '\\' "
            "ORDER BY created_at LIMIT 1",
            (today, f"%{escaped}%"),
        ).fetchone()
        if not row:
            return None
        completed_at = self._now().strftime(self._SQL_TS_FMT) if status == "completed" else None
        self.db.execute(
            "UPDATE daily_goals SET status = ?, reflection = ?, completed_at = ? "
            "WHERE id = ?",
            (status, reflection, completed_at, row["id"]),
        )
        self.db.commit()
        return {"id": row["id"], "title": row["title"], "status": status}

    # --- API Costs ---

    # Prices per 1M tokens (input/output)
    _PRICES = {
        "anthropic": (3.00, 15.00),
        "groq_stt": (0.0, 0.0),  # billed per minute, tracked separately
        "voyage": (0.06, 0.0),
    }

    def log_cost(self, provider: str, input_tokens: int = 0,
                 output_tokens: int = 0):
        inp_price, out_price = self._PRICES.get(provider, (0, 0))
        cost = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000
        self.db.execute(
            "INSERT INTO api_costs (provider, input_tokens, output_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?)",
            (provider, input_tokens, output_tokens, cost),
        )
        self.db.commit()

    def get_today_cost(self) -> float:
        """Total API cost (USD) accumulated today. Used by cost circuit breaker."""
        today = self._now().strftime("%Y-%m-%d")
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM api_costs WHERE date(timestamp) = ?",
            (today,),
        ).fetchone()
        return float(row[0])

    def get_open_loops(self, days_ahead: int = 2, limit_per_section: int = 5) -> dict:
        """Return a snapshot of open items that currently need attention."""
        now = self._now()
        now_sql = now.strftime(self._SQL_TS_FMT)
        today = now.strftime("%Y-%m-%d")
        horizon = (now + timedelta(days=max(1, days_ahead))).strftime("%Y-%m-%d")
        end_of_today = now.replace(hour=23, minute=59, second=59, microsecond=0)
        end_of_today_sql = end_of_today.strftime(self._SQL_TS_FMT)

        overdue_reminders = self.db.execute(
            "SELECT id, text, trigger_at, priority FROM reminders "
            "WHERE status = 'pending' AND trigger_at < ? "
            "ORDER BY trigger_at LIMIT ?",
            (now_sql, limit_per_section),
        ).fetchall()
        due_today_reminders = self.db.execute(
            "SELECT id, text, trigger_at, priority FROM reminders "
            "WHERE status = 'pending' AND trigger_at >= ? AND trigger_at <= ? "
            "ORDER BY trigger_at LIMIT ?",
            (now_sql, end_of_today_sql, limit_per_section),
        ).fetchall()
        upcoming_events = self.db.execute(
            "SELECT id, title, start_at, related_person, location FROM events "
            "WHERE start_at >= ? AND date(start_at) <= date(?) "
            "ORDER BY start_at LIMIT ?",
            (now_sql, horizon, limit_per_section),
        ).fetchall()
        pending_goals = self.db.execute(
            "SELECT id, title, priority, status, reflection FROM daily_goals "
            "WHERE date = ? AND status IN ('pending', 'partial') "
            "ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, created_at "
            "LIMIT ?",
            (today, limit_per_section),
        ).fetchall()
        due_decisions = self.db.execute(
            "SELECT id, description, follow_up_at, created_at FROM decisions "
            "WHERE status = 'pending' AND follow_up_at IS NOT NULL AND follow_up_at <= ? "
            "ORDER BY follow_up_at LIMIT ?",
            (now_sql, limit_per_section),
        ).fetchall()

        counts = {
            "overdue_reminders": self.db.execute(
                "SELECT COUNT(*) FROM reminders WHERE status = 'pending' AND trigger_at < ?",
                (now_sql,),
            ).fetchone()[0],
            "due_today_reminders": self.db.execute(
                "SELECT COUNT(*) FROM reminders "
                "WHERE status = 'pending' AND trigger_at >= ? AND trigger_at <= ?",
                (now_sql, end_of_today_sql),
            ).fetchone()[0],
            "upcoming_events": self.db.execute(
                "SELECT COUNT(*) FROM events WHERE start_at >= ? AND date(start_at) <= date(?)",
                (now_sql, horizon),
            ).fetchone()[0],
            "pending_goals": self.db.execute(
                "SELECT COUNT(*) FROM daily_goals "
                "WHERE date = ? AND status IN ('pending', 'partial')",
                (today,),
            ).fetchone()[0],
            "due_decisions": self.db.execute(
                "SELECT COUNT(*) FROM decisions "
                "WHERE status = 'pending' AND follow_up_at IS NOT NULL AND follow_up_at <= ?",
                (now_sql,),
            ).fetchone()[0],
        }

        return {
            "overdue_reminders": [dict(r) for r in overdue_reminders],
            "due_today_reminders": [dict(r) for r in due_today_reminders],
            "upcoming_events": [dict(r) for r in upcoming_events],
            "pending_goals": [dict(r) for r in pending_goals],
            "due_decisions": [dict(r) for r in due_decisions],
            "counts": counts,
        }

    def cleanup_old_data(self, days: int = 90) -> dict:
        """Delete interactions and api_costs older than `days`.

        Also hard-deletes soft-deleted memories that crossed the retention
        horizon. Returns a dict of rows deleted per table. `days <= 0` is a
        no-op — interpreted as "disabled" to prevent accidental wipe.
        """
        counts: dict[str, int] = {"interactions": 0, "api_costs": 0, "memories": 0}
        if days <= 0:
            return counts
        # Use SQL_TS_FMT (space-separated) so string comparison matches DB
        # format exactly — see note above _SQL_TS_FMT about isoformat's bug.
        cutoff = (self._now() - timedelta(days=days)).strftime(self._SQL_TS_FMT)

        cur = self.db.execute("DELETE FROM interactions WHERE timestamp < ?", (cutoff,))
        counts["interactions"] = cur.rowcount

        cur = self.db.execute("DELETE FROM api_costs WHERE timestamp < ?", (cutoff,))
        counts["api_costs"] = cur.rowcount

        cur = self.db.execute(
            "DELETE FROM memories WHERE status = 'deleted' AND "
            "COALESCE(last_accessed, created_at) < ?",
            (cutoff,),
        )
        counts["memories"] = cur.rowcount

        self.db.commit()
        return counts

    def get_stats(self) -> dict:
        """Get usage stats for /stats command."""
        today = self._now().strftime("%Y-%m-%d")
        month_start = self._now().strftime("%Y-%m-01")

        # Today's cost
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as cost, "
            "COALESCE(SUM(input_tokens), 0) as inp, "
            "COALESCE(SUM(output_tokens), 0) as outp "
            "FROM api_costs WHERE date(timestamp) = ?", (today,)
        ).fetchone()
        today_cost = row[0]
        today_tokens = row[1] + row[2]

        # Month cost
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as cost "
            "FROM api_costs WHERE date(timestamp) >= ?", (month_start,)
        ).fetchone()
        month_cost = row[0]

        # Memory count
        mem_count = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0]

        # Contact count
        contact_count = self.db.execute(
            "SELECT COUNT(*) FROM contacts"
        ).fetchone()[0]

        # Interactions today
        interactions_today = self.db.execute(
            "SELECT COUNT(*) FROM interactions WHERE date(timestamp) = ?", (today,)
        ).fetchone()[0]

        # Per-provider breakdown (today)
        provider_rows = self.db.execute(
            "SELECT provider, COALESCE(SUM(cost_usd), 0) as cost, "
            "COALESCE(SUM(input_tokens + output_tokens), 0) as tokens "
            "FROM api_costs WHERE date(timestamp) = ? GROUP BY provider",
            (today,),
        ).fetchall()
        providers = {r["provider"]: {"cost": r["cost"], "tokens": r["tokens"]}
                     for r in provider_rows}

        # 7-day cost trend
        week_start = (self._now() - timedelta(days=6)).strftime("%Y-%m-%d")
        trend_rows = self.db.execute(
            "SELECT date(timestamp) as day, COALESCE(SUM(cost_usd), 0) as cost "
            "FROM api_costs WHERE date(timestamp) >= ? "
            "GROUP BY day ORDER BY day",
            (week_start,),
        ).fetchall()
        week_trend = [{"date": r["day"], "cost": r["cost"]} for r in trend_rows]

        return {
            "today_cost": today_cost,
            "today_tokens": today_tokens,
            "month_cost": month_cost,
            "memories": mem_count,
            "contacts": contact_count,
            "interactions_today": interactions_today,
            "providers": providers,
            "week_trend": week_trend,
        }

    def close(self):
        self.db.close()
