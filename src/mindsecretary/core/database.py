from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import tz_now
from .enums import Priority, Status

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path, timezone: str | None = None,
                 migrations_dir: Path | None = None):
        # Store the db path so create_backup() can derive the default
        # backup directory next to it. Only used for backup; queries go
        # through self.db (the Connection) directly.
        self._db_path = Path(db_path)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        # SQLite's built-in lower() is ASCII-only — "ПРОЕКТ" stays "ПРОЕКТ",
        # breaking case-insensitive search on Russian (or any non-ASCII)
        # content. Register a Python-backed function so Cyrillic and mixed
        # scripts fold correctly.
        self.db.create_function(
            "pylower", 1, lambda s: s.lower() if isinstance(s, str) else s,
        )
        self._timezone = timezone
        self._init_tables()
        if migrations_dir is None:
            # Use _project_root() so pip-installed packages find migrations/
            # at the real project root (set via MINDSECRETARY_ROOT=/app in
            # Docker), not at site-packages/.. /migrations.
            from .config import _project_root
            migrations_dir = _project_root() / "migrations"
        self._apply_migrations(migrations_dir)
        self._verify_integrity()

    def create_backup(self, keep: int = 30) -> dict:
        """Online SQLite backup + retention prune.

        Mirrors `scripts/backup.sh` so users running both don't get
        diverging behaviour: writes `mindsecretary_YYYYMMDD_HHMMSS.db`
        into `<db_path>.parent / 'backups'`, keeps the latest `keep`
        backups, deletes the rest by mtime.

        SQLite's online backup API copies pages while the DB is in use,
        so this is safe to run from the scheduler without pausing the
        bot. Best-effort: any failure logs and returns the failure shape
        in the result dict — the scheduler's `daily_backup` job ignores
        the result besides logging, so a one-off disk-full glitch
        doesn't break later attempts.

        Returns: {"ok": bool, "path": str | None, "pruned": int,
                  "error": str | None}
        """
        backup_dir = self._db_path.parent / "backups"
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Cannot create backup dir %s: %s",
                           backup_dir, type(e).__name__)
            return {"ok": False, "path": None, "pruned": 0,
                    "error": type(e).__name__}

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = backup_dir / f"mindsecretary_{ts}.db"
        try:
            with sqlite3.connect(str(target)) as dest:
                self.db.backup(dest)
        except sqlite3.Error as e:
            logger.warning("DB backup to %s failed: %s",
                           target, type(e).__name__)
            # Clean up partial file if any so prune doesn't count it
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
            return {"ok": False, "path": None, "pruned": 0,
                    "error": type(e).__name__}

        pruned = self._prune_backups(backup_dir, keep=keep)
        logger.info("DB backup: %s (pruned %d)", target.name, pruned)
        return {"ok": True, "path": str(target), "pruned": pruned,
                "error": None}

    @staticmethod
    def _prune_backups(directory: Path, keep: int = 30) -> int:
        """Keep the latest `keep` backups by mtime, delete the rest.

        Sort by mtime (filename embeds the timestamp, so mtime ordering
        matches creation order in practice — but mtime is the truth in
        case the user manually copied old backups in). Returns count
        of files deleted.
        """
        if not directory.exists() or keep < 0:
            return 0
        files = sorted(
            directory.glob("mindsecretary_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        pruned = 0
        for old in files[keep:]:
            try:
                old.unlink()
                pruned += 1
            except OSError as e:
                logger.warning("Failed to delete old backup %s: %s",
                               old, type(e).__name__)
        return pruned

    def _verify_integrity(self) -> bool:
        """Run `PRAGMA integrity_check` once on startup.

        SQLite reports corruption on a per-page basis lazily — a damaged
        DB might pass init and only show cryptic errors on the first
        query that touches the broken page. Running this once up front
        gives ops a clear signal in the bot's startup log.

        Doesn't raise — corruption isn't necessarily fatal (most queries
        may still work) and crashing on startup hides the diagnostic
        from anyone tailing the log. Returns True if 'ok', False
        otherwise. The check itself can fail on an unreadable file —
        also logged + returned as False.
        """
        try:
            rows = self.db.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as e:
            logger.error(
                "DB integrity_check itself failed (%s) — DB may be unusable",
                type(e).__name__,
            )
            return False
        # SQLite returns a single row with text "ok" on a healthy DB,
        # otherwise one row per detected problem. Either Row objects or
        # plain tuples depending on row_factory state.
        results = [r[0] for r in rows]
        if results == ["ok"]:
            logger.info("DB integrity_check: ok")
            return True
        # Cap detail logging so a wildly broken DB doesn't dump megabytes.
        sample = "; ".join(str(r)[:200] for r in results[:5])
        suffix = f" (+{len(results) - 5} more)" if len(results) > 5 else ""
        logger.warning(
            "DB integrity_check found %d issue(s): %s%s",
            len(results), sample, suffix,
        )
        return False

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

    def local_now_naive(self) -> datetime:
        """Current profile-local time as a naive datetime.

        Use this when comparing against values written as local-TZ naive
        strings (e.g. `contacts.last_contact` via `_now().strftime()`).
        Returns naive `datetime.now()` if no profile TZ is configured.
        """
        now = self._now()
        return now.replace(tzinfo=None) if now.tzinfo else now

    def _local_tz_offset_minutes(self) -> int:
        """Current UTC offset of the effective local clock, in minutes.

        Uses the profile TZ when set, otherwise system local TZ (picked up
        via `.astimezone()`). DST-aware — re-read on every call so the
        value reflects the current instant's offset, not schema-time.
        """
        now = self._now()
        if now.tzinfo is None:
            now = now.astimezone()  # attach system local TZ
        offset = now.utcoffset()
        if offset is None:
            return 0
        return int(offset.total_seconds() // 60)

    def _local_day_utc_bounds(self, day_offset: int = 0) -> tuple[str, str]:
        """Return (start_utc_sql, end_utc_sql) for the local day `day_offset`
        days from today. Half-open interval: `start <= ts < end`.

        The SQL strings match _SQL_TS_FMT so comparisons line up with values
        written by SQLite's `datetime('now')` (which is UTC). Use to replace
        `WHERE date(timestamp) = local_today_string` patterns, which mix a
        UTC-stored column with a local-TZ date string and break at offsets
        > 0 (off by a full local day for rows logged between local midnight
        and the UTC offset hours later).
        """
        now = self._now()
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_local = base + timedelta(days=day_offset)
        end_local = start_local + timedelta(days=1)
        if start_local.tzinfo is None:
            # No profile TZ — attach system local via astimezone() so the
            # UTC conversion below reflects a real wall-clock → UTC shift
            # rather than pretending the naive string is already UTC.
            # On Docker (system TZ = UTC) this is a no-op; on a dev machine
            # in Asia/Almaty it correctly subtracts 5h to line up with
            # `datetime('now')` storage.
            start_local = start_local.astimezone()
            end_local = end_local.astimezone()
        start_utc = start_local.astimezone(timezone.utc).strftime(self._SQL_TS_FMT)
        end_utc = end_local.astimezone(timezone.utc).strftime(self._SQL_TS_FMT)
        return start_utc, end_utc

    def _local_date_sql(self, column: str) -> str:
        """Build a SQL fragment that extracts the profile-local date from a
        UTC-stored timestamp column.

        Uses SQLite's `date(ts, '+N minutes')` offset modifier so it works
        for any TZ, including fractional offsets (India +5:30, Nepal +5:45).
        The offset is inlined at call time — DST-aware.
        """
        offset = self._local_tz_offset_minutes()
        if offset == 0:
            return f"date({column})"
        sign = "+" if offset >= 0 else "-"
        return f"date({column}, '{sign}{abs(offset)} minutes')"

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

    def reschedule_reminder_by_hint(self, hint: str,
                                    new_trigger_at: str) -> dict | None:
        """Move the most-imminent pending reminder matching `hint` to
        `new_trigger_at`. Returns the updated row dict (with the new
        trigger_at), or None if nothing matched.

        For recurring reminders this changes only the upcoming instance's
        trigger_at — the series continues from there. Sent reminders are
        excluded (you'd create a new one, not reschedule a fired one).
        """
        if not hint or not hint.strip():
            return None
        if not new_trigger_at or not new_trigger_at.strip():
            return None
        escaped = self._escape_like(hint.strip().lower())
        row = self.db.execute(
            "SELECT * FROM reminders "
            "WHERE status = 'pending' AND pylower(text) LIKE ? ESCAPE '\\' "
            "ORDER BY trigger_at LIMIT 1",
            (f"%{escaped}%",),
        ).fetchone()
        if not row:
            return None
        self.db.execute(
            "UPDATE reminders SET trigger_at = ? WHERE id = ?",
            (new_trigger_at, row["id"]),
        )
        self.db.commit()
        updated = dict(row)
        updated["trigger_at"] = new_trigger_at
        return updated

    def cancel_reminder_by_hint(self, hint: str) -> dict | None:
        """Cancel the most-imminent pending reminder whose text matches `hint`
        (case-insensitive substring, Cyrillic-aware via pylower).

        Returns the cancelled row dict, or None if nothing matched.
        Recurring reminders: cancellation stops the series — we do NOT
        auto-create a next occurrence (that's mark_reminder_sent's job),
        because user intent on "отмени напоминание" is "stop bothering me".
        """
        if not hint or not hint.strip():
            return None
        escaped = self._escape_like(hint.strip().lower())
        row = self.db.execute(
            "SELECT * FROM reminders "
            "WHERE status = 'pending' AND pylower(text) LIKE ? ESCAPE '\\' "
            "ORDER BY trigger_at LIMIT 1",
            (f"%{escaped}%",),
        ).fetchone()
        if not row:
            return None
        self.db.execute(
            "UPDATE reminders SET status = 'cancelled' WHERE id = ?",
            (row["id"],),
        )
        self.db.commit()
        return dict(row)

    def count_pending_reminders_matching(self, hint: str) -> int:
        """How many pending reminders match `hint` — used by the cancel
        handler to disclose ambiguity ("matched 3, cancelled the soonest")."""
        if not hint or not hint.strip():
            return 0
        escaped = self._escape_like(hint.strip().lower())
        row = self.db.execute(
            "SELECT COUNT(*) FROM reminders "
            "WHERE status = 'pending' AND pylower(text) LIKE ? ESCAPE '\\'",
            (f"%{escaped}%",),
        ).fetchone()
        return int(row[0])

    def mark_reminder_sent(self, reminder_id: str):
        row = self.db.execute(
            "SELECT text, trigger_at, priority, recurrence FROM reminders WHERE id = ?",
            (reminder_id,),
        ).fetchone()
        self.db.execute(
            "UPDATE reminders SET status = 'sent' WHERE id = ?", (reminder_id,)
        )
        # Auto-create next occurrence for recurring reminders.
        #
        # Roll the next trigger forward past `now` if needed: when the bot
        # was down for several periods, naive `old + delta` lands in the
        # past too, fires immediately on the next 5-min check, marks itself
        # sent, creates ANOTHER past trigger — repeating until caught up.
        # User sees N back-to-back fires of the same daily reminder. Spam.
        #
        # Catch-up loop sends ONE fire (the original due row), schedules the
        # next FUTURE occurrence. The bounded delta (1d/7d/30d) makes
        # iteration count tiny even after long downtime — at most ~365 hops
        # for "daily" after a year offline.
        if row and row["recurrence"] in self._RECURRENCE_DELTAS:
            delta = self._RECURRENCE_DELTAS[row["recurrence"]]
            try:
                old_trigger = datetime.fromisoformat(
                    row["trigger_at"].replace(" ", "T")
                )
                # `trigger_at` is profile-local naive (LLM tool writes via
                # _SQL_TS_FMT.strftime on a tz-aware self._now()). Compare
                # against the same clock so rollforward respects the user's
                # day boundary, not UTC.
                now_local = self.local_now_naive()
                next_trigger = old_trigger + delta
                while next_trigger <= now_local:
                    next_trigger += delta
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
        # pylower (Python str.lower) instead of SQLite native lower() — the
        # latter is ASCII-only, so "Иван" stays "Иван" and dedup against an
        # earlier "иван" fails. Same problem for aliases and get_contacts.
        existing = self.db.execute(
            "SELECT * FROM contacts WHERE pylower(name) = ?", (name.lower(),)
        ).fetchone()
        # Also check aliases (comma-separated) for fuzzy matching
        if not existing:
            escaped = self._escape_like(name.lower())
            existing = self.db.execute(
                "SELECT * FROM contacts WHERE pylower(aliases) LIKE ? ESCAPE '\\'",
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
        # pylower for Cyrillic-aware case folding — SQLite's lower() only
        # handles ASCII, which silently breaks search on Russian names.
        escaped = self._escape_like(query.lower())
        rows = self.db.execute(
            "SELECT * FROM contacts WHERE pylower(name) LIKE ? ESCAPE '\\' "
            "OR pylower(relation) LIKE ? ESCAPE '\\' "
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

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        rows = self.db.execute(
            "SELECT direction, content, timestamp, message_type, metadata "
            "FROM interactions "
            "WHERE message_type IN ('voice', 'text', 'forward', 'chat', 'notification') "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search_past_conversations(self, query: str, days: int = 30,
                                  limit: int = 10) -> list[dict]:
        """Keyword LIKE search over interactions.content (user + bot text).

        Exists so the LLM can recall exchanges older than the replayed
        history window. Matches case-insensitively on a single substring —
        embeddings would be more flexible, but raw content isn't embedded
        (only curated memories are) and adding a second index path just for
        this tool isn't worth it. Returns newest-first.
        """
        if not query or not query.strip():
            return []
        since = (self._now() - timedelta(days=max(1, days))).strftime(self._SQL_TS_FMT)
        escaped = self._escape_like(query.strip().lower())
        rows = self.db.execute(
            "SELECT timestamp, direction, message_type, content, metadata "
            "FROM interactions "
            "WHERE timestamp >= ? "
            "  AND message_type IN ('voice', 'text', 'forward', 'chat', 'notification') "
            "  AND pylower(content) LIKE ? ESCAPE '\\' "
            "ORDER BY timestamp DESC LIMIT ?",
            (since, f"%{escaped}%", max(1, min(limit, 50))),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_notifications_today(self) -> int:
        """Count today's outgoing notifications, aligned to the local day.

        Compares against UTC bounds of the local day so the limiter resets
        at local midnight (not UTC midnight — interactions.timestamp is
        stored via SQLite's `datetime('now')` which is UTC).
        """
        start_utc, end_utc = self._local_day_utc_bounds()
        row = self.db.execute(
            "SELECT COUNT(*) as cnt FROM interactions "
            "WHERE direction = 'out' AND message_type = 'notification' "
            "AND timestamp >= ? AND timestamp < ?",
            (start_utc, end_utc),
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
            "SELECT id FROM habits WHERE pylower(name) = ?",
            (habit_name.lower(),),
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
        """Get all habits with current streak and 7-day completion rate.

        Streak walks calendar days back from today: a day is in the streak
        only if it has a `done=1` log. A missing day (no row) OR a `done=0`
        row both break the streak — without the calendar walk, logs of
        Mon/Tue/Wed/Fri (Thu missing) would falsely count as a 4-day streak
        because the inner loop only iterated rows, not days.
        """
        habits = self.db.execute(
            "SELECT id, name FROM habits ORDER BY name"
        ).fetchall()
        results = []
        now = self._now()
        today_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_str = today_dt.strftime("%Y-%m-%d")
        for h in habits:
            logs = self.db.execute(
                "SELECT date, done FROM habit_log WHERE habit_id = ? "
                "AND date <= ? ORDER BY date DESC LIMIT 60",
                (h["id"], today_str),
            ).fetchall()
            done_by_date = {log["date"]: bool(log["done"]) for log in logs}

            streak = 0
            day = today_dt
            # 60-day cap matches the LIMIT 60 above — we never have data
            # past that anyway, so capping the walk avoids an unbounded
            # backward scan if someone has 60 consecutive done days.
            for _ in range(60):
                key = day.strftime("%Y-%m-%d")
                if done_by_date.get(key):
                    streak += 1
                    day = day - timedelta(days=1)
                else:
                    break

            # 7-day rate
            week_start = (now - timedelta(days=6)).strftime("%Y-%m-%d")
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
                "logged_today": today_str in done_by_date,
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

        `memories.created_at` is stored UTC, so compare against the UTC
        lower bound of local-midnight N days ago — not a local date string,
        which would drop memories created in the first offset-hours of the
        target day on positive-offset timezones.
        """
        since_utc, _ = self._local_day_utc_bounds(day_offset=-days)
        rows = self.db.execute(
            "SELECT "
            "  COALESCE(NULLIF(related_person, ''), category) AS label, "
            "  COUNT(*) AS cnt, "
            "  SUM(importance) AS total_importance "
            "FROM memories "
            "WHERE status = 'active' "
            "  AND importance >= 5 "
            "  AND created_at >= ? "
            "GROUP BY label "
            "HAVING cnt >= 2 "
            "ORDER BY total_importance DESC, cnt DESC "
            "LIMIT ?",
            (since_utc, limit),
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
        """Find today's pending goal by keyword and mark it.

        pylower for Cyrillic case-insensitivity — `title` is user-supplied
        free text (often Russian), and SQLite's native LOWER is ASCII-only,
        so an LLM-produced hint with a different case ('ЗАРЯДКА' vs
        stored 'зарядка') would silently miss.
        """
        today = self._now().strftime("%Y-%m-%d")
        if status not in (Status.COMPLETED, Status.SKIPPED, Status.PARTIAL):
            status = Status.COMPLETED
        escaped = self._escape_like(hint.lower())
        row = self.db.execute(
            "SELECT id, title FROM daily_goals "
            "WHERE date = ? AND status = 'pending' "
            "AND pylower(title) LIKE ? ESCAPE '\\' "
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
        """Total API cost (USD) accumulated today (profile local day).

        Used by the cost circuit breaker. Aligns to local midnight via UTC
        bounds so the daily spend cap resets on the user's clock, not UTC.
        """
        start_utc, end_utc = self._local_day_utc_bounds()
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM api_costs "
            "WHERE timestamp >= ? AND timestamp < ?",
            (start_utc, end_utc),
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

    def skip_stale_pending_goals(self) -> int:
        """Mark `pending` daily_goals from before today as `skipped`.

        Goals stay `pending` forever if the user never reports back —
        polluting completion-rate analytics and slowly accumulating
        rows. Conservative auto-marking (skipped, not deleted, with an
        explicit reflection note) keeps the audit trail clear: future
        readers can tell user-skipped from auto-skipped at a glance.
        """
        today = self._now().strftime("%Y-%m-%d")
        cur = self.db.execute(
            "UPDATE daily_goals "
            "SET status = 'skipped', "
            "    reflection = COALESCE(reflection, '[авто] не закрыто к концу дня') "
            "WHERE date < ? AND status = 'pending'",
            (today,),
        )
        self.db.commit()
        return cur.rowcount

    def cleanup_old_data(self, days: int = 90) -> dict:
        """Delete interactions and api_costs older than `days`.

        Also hard-deletes soft-deleted memories that crossed the retention
        horizon. Returns a dict of rows deleted per table. `days <= 0` is a
        no-op — interpreted as "disabled" to prevent accidental wipe.

        Cutoff is computed in UTC because all three target columns
        (`interactions.timestamp`, `api_costs.timestamp`,
        `memories.last_accessed/created_at`) are written via SQLite's
        `datetime('now')` — UTC-naive strings. Using `self._now()`
        (profile-local naive) drifts by the profile's UTC offset, so
        retention silently skews by a few hours per cycle. CLAUDE.md
        flags this whole class of mixed-TZ comparisons.
        """
        counts: dict[str, int] = {
            "interactions": 0, "api_costs": 0,
            "memories": 0, "stale_goals": 0,
        }
        # Stale goal sweep is independent of retention horizon — it always
        # runs, since stale-pending pollutes analytics regardless of how
        # long the rows persist.
        counts["stale_goals"] = self.skip_stale_pending_goals()
        if days <= 0:
            return counts
        # Use SQL_TS_FMT (space-separated) so string comparison matches DB
        # format exactly — see note above _SQL_TS_FMT about isoformat's bug.
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime(self._SQL_TS_FMT)

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
        """Get usage stats for /stats command, aligned to the local day.

        All date-bucket boundaries are computed in the profile's timezone
        so "today" / "this month" / 7-day trend match what the user sees
        on their clock. Timestamps are UTC-stored, so queries use either
        UTC half-open bounds or the offset-modifier `date(ts, 'N minutes')`.
        """
        start_utc, end_utc = self._local_day_utc_bounds()

        # Month start: compute as local-midnight of the first-of-month, in UTC.
        # Mirrors `_local_day_utc_bounds` — naive `_now()` is attached to the
        # system local TZ via astimezone() before converting, so month roll-
        # over aligns to the user's clock instead of UTC midnight on day 1.
        month_start_local = self._now().replace(
            day=1, hour=0, minute=0, second=0, microsecond=0,
        )
        if month_start_local.tzinfo is None:
            month_start_local = month_start_local.astimezone()
        month_start_utc = month_start_local.astimezone(
            timezone.utc,
        ).strftime(self._SQL_TS_FMT)

        # Today's cost
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as cost, "
            "COALESCE(SUM(input_tokens), 0) as inp, "
            "COALESCE(SUM(output_tokens), 0) as outp "
            "FROM api_costs WHERE timestamp >= ? AND timestamp < ?",
            (start_utc, end_utc),
        ).fetchone()
        today_cost = row[0]
        today_tokens = row[1] + row[2]

        # Month cost
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as cost "
            "FROM api_costs WHERE timestamp >= ?", (month_start_utc,)
        ).fetchone()
        month_cost = row[0]

        # Memory count
        mem_count = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0]

        # Memory category breakdown — what kinds of facts the external
        # brain is accumulating. Surfaces the user's actual usage shape
        # ("oh I have 60 work / 50 personal / 30 promise") which the bare
        # total never reveals. Sorted by count desc so the dominant
        # categories are visible without scrolling.
        mem_categories = self.db.execute(
            "SELECT category, COUNT(*) as cnt FROM memories "
            "WHERE status = 'active' GROUP BY category "
            "ORDER BY cnt DESC, category"
        ).fetchall()

        # Contact count
        contact_count = self.db.execute(
            "SELECT COUNT(*) FROM contacts"
        ).fetchone()[0]

        # Interactions today
        interactions_today = self.db.execute(
            "SELECT COUNT(*) FROM interactions WHERE timestamp >= ? AND timestamp < ?",
            (start_utc, end_utc),
        ).fetchone()[0]

        # Per-provider breakdown (today)
        provider_rows = self.db.execute(
            "SELECT provider, COALESCE(SUM(cost_usd), 0) as cost, "
            "COALESCE(SUM(input_tokens + output_tokens), 0) as tokens "
            "FROM api_costs WHERE timestamp >= ? AND timestamp < ? GROUP BY provider",
            (start_utc, end_utc),
        ).fetchall()
        providers = {r["provider"]: {"cost": r["cost"], "tokens": r["tokens"]}
                     for r in provider_rows}

        # 7-day cost trend — group rows by LOCAL day (via offset modifier)
        # so each bucket aligns to local midnight rather than UTC midnight.
        week_start_utc, _ = self._local_day_utc_bounds(day_offset=-6)
        day_expr = self._local_date_sql("timestamp")
        trend_rows = self.db.execute(
            f"SELECT {day_expr} as day, COALESCE(SUM(cost_usd), 0) as cost "
            f"FROM api_costs WHERE timestamp >= ? "
            f"GROUP BY day ORDER BY day",
            (week_start_utc,),
        ).fetchall()
        week_trend = [{"date": r["day"], "cost": r["cost"]} for r in trend_rows]

        # Monthly projection — extrapolate from 7-day avg.
        # We project from the 7-day window (not month-to-date) because
        # the user's API usage skews early-month-heavy or end-month-heavy
        # depending on cycle, and a recent average is more representative
        # of the *current* burn rate. Need at least 3 days of cost data
        # before showing — projecting from 1 day of usage just amplifies
        # noise (a single $0.30 day → $9 projection). UI hides the line
        # below the threshold so users aren't misled by spiky early data.
        if len(week_trend) >= 3:
            avg_daily = sum(d["cost"] for d in week_trend) / len(week_trend)
            month_projection = avg_daily * 30
        else:
            month_projection = None

        return {
            "today_cost": today_cost,
            "today_tokens": today_tokens,
            "month_cost": month_cost,
            "month_projection": month_projection,
            "memories": mem_count,
            "memory_categories": [
                {"category": r["category"], "count": r["cnt"]}
                for r in mem_categories
            ],
            "contacts": contact_count,
            "interactions_today": interactions_today,
            "providers": providers,
            "week_trend": week_trend,
        }

    # --- Ephemeral state (transient "right now" context with TTL) ---

    _EPHEMERAL_TTL_MIN_HOURS = 0.5
    _EPHEMERAL_TTL_MAX_HOURS = 72.0

    def set_ephemeral_state(self, key: str, value: str, ttl_hours: float) -> None:
        """Upsert a current-state row with a TTL (clamped to 0.5-72 hours).

        INSERT OR REPLACE semantics: same key overwrites prior value + TTL.
        Exists to carry short-lived context (location, health, availability)
        that matters for Claude's answers but shouldn't live in long memory.
        """
        ttl = max(self._EPHEMERAL_TTL_MIN_HOURS,
                  min(self._EPHEMERAL_TTL_MAX_HOURS, float(ttl_hours)))
        expires_at = (self._now() + timedelta(hours=ttl)).strftime(self._SQL_TS_FMT)
        self.db.execute(
            "INSERT OR REPLACE INTO ephemeral_state (key, value, expires_at) "
            "VALUES (?, ?, ?)",
            (key, value, expires_at),
        )
        self.db.commit()

    def get_active_ephemeral_state(self) -> list[dict]:
        """Return non-expired state rows, lazy-cleaning expired ones first."""
        now_sql = self._now().strftime(self._SQL_TS_FMT)
        self.db.execute("DELETE FROM ephemeral_state WHERE expires_at < ?", (now_sql,))
        self.db.commit()
        rows = self.db.execute(
            "SELECT key, value, expires_at, created_at FROM ephemeral_state "
            "ORDER BY expires_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_ephemeral_state(self, key: str | None = None) -> int:
        """Clear ephemeral state. Pass key to clear one; no arg clears all."""
        if key:
            cur = self.db.execute("DELETE FROM ephemeral_state WHERE key = ?", (key,))
        else:
            cur = self.db.execute("DELETE FROM ephemeral_state")
        self.db.commit()
        return cur.rowcount

    def close(self):
        self.db.close()
