"""Tests for core/database.py — CRUD operations and timestamp handling."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mindsecretary.core.database import Database

SQL_TS_FMT = "%Y-%m-%d %H:%M:%S"


class TestEvents:
    def test_create_and_get_event(self, tmp_db: Database):
        event = tmp_db.create_event("Dentist", "2026-04-15 10:00:00")
        assert event["title"] == "Dentist"
        assert event["start_at"] == "2026-04-15 10:00:00"

        events = tmp_db.get_events("2026-04-15")
        assert len(events) == 1
        assert events[0]["title"] == "Dentist"

    def test_get_events_empty_range(self, tmp_db: Database):
        tmp_db.create_event("Meeting", "2026-04-15 09:00:00")
        assert tmp_db.get_events("2026-04-16") == []

    def test_get_events_date_range(self, tmp_db: Database):
        tmp_db.create_event("Day1", "2026-04-15 09:00:00")
        tmp_db.create_event("Day2", "2026-04-16 09:00:00")
        tmp_db.create_event("Day3", "2026-04-17 09:00:00")

        events = tmp_db.get_events("2026-04-15", "2026-04-16")
        assert len(events) == 2


class TestReminders:
    def test_create_and_get_pending(self, tmp_db: Database):
        r = tmp_db.create_reminder("Call Mom", "2026-04-15 18:00:00")
        assert r["text"] == "Call Mom"
        assert r["status"] == "pending"

        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1

    def test_mark_reminder_sent(self, tmp_db: Database):
        r = tmp_db.create_reminder("Water plants", "2026-04-15 08:00:00")
        tmp_db.mark_reminder_sent(r["id"])

        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 0

    def test_get_due_reminders_uses_sql_format(self, tmp_db: Database):
        """Due reminders comparison must work with space-separated timestamps."""
        past = (datetime.now() - timedelta(hours=1)).strftime(SQL_TS_FMT)
        future = (datetime.now() + timedelta(hours=1)).strftime(SQL_TS_FMT)

        tmp_db.create_reminder("Past", past)
        tmp_db.create_reminder("Future", future)

        due = tmp_db.get_due_reminders()
        assert len(due) == 1
        assert due[0]["text"] == "Past"


class TestContacts:
    def test_upsert_creates_new_contact(self, tmp_db: Database):
        c = tmp_db.upsert_contact("Alice", relation="friend")
        assert c["name"] == "Alice"
        assert c["relation"] == "friend"

    def test_upsert_updates_existing(self, tmp_db: Database):
        tmp_db.upsert_contact("Bob", relation="colleague")
        c2 = tmp_db.upsert_contact("Bob", notes="likes coffee")

        assert c2["relation"] == "colleague"
        assert "coffee" in (c2.get("notes") or "")
        assert c2["mention_count"] == 1  # 0 from create (DB default) + 1 from update

    def test_upsert_timestamp_format(self, tmp_db: Database):
        """Contacts must use space-separated timestamps for SQL compat."""
        c = tmp_db.upsert_contact("Carol")
        # last_contact should NOT contain 'T'
        assert "T" not in c["last_contact"]
        assert " " in c["last_contact"]

    def test_get_contacts_by_name(self, tmp_db: Database):
        tmp_db.upsert_contact("Alice", relation="friend")
        tmp_db.upsert_contact("Bob", relation="colleague")

        results = tmp_db.get_contacts("alice")
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_get_contacts_by_relation(self, tmp_db: Database):
        tmp_db.upsert_contact("Alice", relation="friend")
        found = tmp_db.get_contacts("friend")
        assert len(found) == 1

    def test_upcoming_birthdays(self, tmp_db: Database):
        today = datetime.now().strftime("%Y-%m-%d")
        tmp_db.upsert_contact("Eve", birthday=today)
        bdays = tmp_db.get_upcoming_birthdays(days=1)
        assert len(bdays) == 1
        assert bdays[0]["name"] == "Eve"


class TestInteractions:
    def test_log_and_get_interactions(self, tmp_db: Database):
        iid = tmp_db.log_interaction("in", "text", "Hello")
        assert isinstance(iid, str)

        interactions = tmp_db.get_interactions(limit=10)
        assert len(interactions) == 1
        assert interactions[0]["content"] == "Hello"

    def test_recent_messages(self, tmp_db: Database):
        tmp_db.log_interaction("in", "text", "First")
        tmp_db.log_interaction("out", "chat", "Reply")

        recent = tmp_db.get_recent_messages(limit=10)
        assert len(recent) == 2
        # Should be in chronological order (not reversed)
        assert recent[0]["content"] == "First"
        assert recent[1]["content"] == "Reply"

    def test_recent_messages_includes_notifications(self, tmp_db: Database):
        """Proactive sends (briefing/reminder/etc.) must show up in the
        chronological log the Brain feeds to Claude — otherwise the bot has
        no record of what it sent autonomously."""
        tmp_db.log_interaction("in", "text", "hi")
        tmp_db.log_interaction(
            "out", "notification", "☀️ Доброе утро",
            metadata={"kind": "morning_briefing"},
        )
        tmp_db.log_interaction(
            "out", "notification", "⏰ Напоминание: позвонить",
            metadata={"kind": "reminder"},
        )

        recent = tmp_db.get_recent_messages(limit=10)
        assert len(recent) == 3
        notifs = [m for m in recent if m["message_type"] == "notification"]
        assert len(notifs) == 2
        # metadata must round-trip as JSON string with the kind preserved
        import json as _json
        kinds = [_json.loads(m["metadata"])["kind"] for m in notifs]
        assert kinds == ["morning_briefing", "reminder"]

    def test_count_notifications_today(self, tmp_db: Database):
        assert tmp_db.count_notifications_today() == 0
        tmp_db.log_interaction("out", "notification", "Reminder!")
        assert tmp_db.count_notifications_today() == 1

    def test_search_past_conversations_matches_keyword(self, tmp_db: Database):
        """Lets the LLM recall exchanges older than the replayed history
        window when the user references something by keyword."""
        tmp_db.log_interaction("in", "text", "давай поедем в Сочи летом")
        tmp_db.log_interaction("out", "chat", "Хорошо, когда именно?")
        tmp_db.log_interaction("in", "text", "какая-то другая тема")
        tmp_db.log_interaction(
            "out", "notification", "⏰ Напоминание про Сочи",
            metadata={"kind": "reminder"},
        )

        rows = tmp_db.search_past_conversations("Сочи", days=30, limit=10)
        assert len(rows) == 2
        contents = {r["content"] for r in rows}
        assert "давай поедем в Сочи летом" in contents
        assert "⏰ Напоминание про Сочи" in contents
        # Must be newest-first
        assert rows[0]["timestamp"] >= rows[1]["timestamp"]

    def test_search_past_conversations_is_case_insensitive(self, tmp_db: Database):
        tmp_db.log_interaction("in", "text", "Обсудили ПРОЕКТ с Колей")
        rows = tmp_db.search_past_conversations("проект", days=30)
        assert len(rows) == 1

    def test_search_past_conversations_empty_query_noop(self, tmp_db: Database):
        tmp_db.log_interaction("in", "text", "anything")
        assert tmp_db.search_past_conversations("", days=30) == []
        assert tmp_db.search_past_conversations("   ", days=30) == []


class TestDecisions:
    def test_create_decision_timestamp_format(self, tmp_db: Database):
        """Decision follow_up_at must use space-separated format."""
        d = tmp_db.create_decision("Buy a car", follow_up_days=30)
        assert "T" not in d["follow_up_at"]
        assert " " in d["follow_up_at"]

    def test_resolve_decision_by_hint(self, tmp_db: Database):
        tmp_db.create_decision("Buy electric bicycle")
        resolved = tmp_db.resolve_decision_by_hint("bicycle", "Bought it!")
        assert resolved is not None
        assert "bicycle" in resolved["description"]

    def test_resolve_nonexistent_returns_none(self, tmp_db: Database):
        assert tmp_db.resolve_decision_by_hint("nonexistent", "nope") is None

    def test_get_pending_followups(self, tmp_db: Database):
        # Create decision with follow-up in the past
        tmp_db.create_decision("Old choice", follow_up_days=0)
        followups = tmp_db.get_pending_decision_followups()
        assert len(followups) >= 1

    def test_push_followup(self, tmp_db: Database):
        d = tmp_db.create_decision("Test decision", follow_up_days=1)
        original_followup = d["follow_up_at"]
        tmp_db.push_decision_followup(d["id"], days=14)

        updated = tmp_db.get_pending_decisions(limit=1)
        assert updated[0]["follow_up_at"] > original_followup


class TestHabits:
    def test_log_new_habit(self, tmp_db: Database):
        result = tmp_db.log_habit("exercise", done=True)
        assert result["habit"] == "exercise"
        assert result["done"] is True

    def test_log_habit_upsert(self, tmp_db: Database):
        tmp_db.log_habit("reading", done=True, date="2026-04-15")
        tmp_db.log_habit("reading", done=False, date="2026-04-15")
        # Should not raise — upsert via ON CONFLICT


class TestDailyGoals:
    def test_create_goal(self, tmp_db: Database):
        g = tmp_db.create_daily_goal("Go to gym", priority="high")
        assert g["title"] == "Go to gym"
        assert g["priority"] == "high"
        assert g["status"] == "pending"

    def test_invalid_priority_defaults_to_medium(self, tmp_db: Database):
        g = tmp_db.create_daily_goal("Read", priority="urgent")
        assert g["priority"] == "medium"

    def test_complete_goal_by_hint(self, tmp_db: Database):
        tmp_db.create_daily_goal("Write report")
        result = tmp_db.complete_daily_goal_by_hint("report", status="completed")
        assert result is not None
        assert result["status"] == "completed"

    def test_complete_nonexistent_goal(self, tmp_db: Database):
        assert tmp_db.complete_daily_goal_by_hint("nothing") is None


class TestDiary:
    def test_save_and_get_diary(self, tmp_db: Database):
        tmp_db.save_diary_entry("2026-04-15", "Good day.", mood="positive")
        entries = tmp_db.get_diary_entries(days=7)
        assert len(entries) == 1
        assert entries[0]["mood"] == "positive"

    def test_diary_upsert(self, tmp_db: Database):
        tmp_db.save_diary_entry("2026-04-15", "Morning.")
        tmp_db.save_diary_entry("2026-04-15", "Updated.")
        entries = tmp_db.get_diary_entries(days=7)
        assert len(entries) == 1
        assert entries[0]["content"] == "Updated."


class TestStats:
    def test_get_stats_empty(self, tmp_db: Database):
        stats = tmp_db.get_stats()
        assert stats["memories"] == 0
        assert stats["contacts"] == 0
        assert stats["today_cost"] == 0

    def test_log_cost(self, tmp_db: Database):
        tmp_db.log_cost("anthropic", input_tokens=1000, output_tokens=500)
        stats = tmp_db.get_stats()
        assert stats["today_cost"] > 0
        assert stats["today_tokens"] == 1500


class TestCostBreaker:
    def test_get_today_cost_empty(self, tmp_db: Database):
        assert tmp_db.get_today_cost() == 0.0

    def test_get_today_cost_sums_all_providers(self, tmp_db: Database):
        tmp_db.log_cost("anthropic", input_tokens=1000, output_tokens=500)
        tmp_db.log_cost("groq", input_tokens=200, output_tokens=0)
        tmp_db.log_cost("voyage", input_tokens=100, output_tokens=0)
        assert tmp_db.get_today_cost() > 0.0


class TestOpenLoops:
    def test_get_open_loops_collects_pending_items(self, tmp_db: Database):
        now = datetime.now()
        tmp_db.create_reminder("Overdue", (now - timedelta(hours=2)).strftime(SQL_TS_FMT))
        tmp_db.create_reminder("Later today", (now + timedelta(hours=2)).strftime(SQL_TS_FMT))
        tmp_db.create_event("Meeting", (now + timedelta(hours=3)).strftime(SQL_TS_FMT))
        tmp_db.create_daily_goal("Finish doc", priority="high")
        tmp_db.create_decision("Choose hosting", follow_up_days=0)

        loops = tmp_db.get_open_loops(days_ahead=2, limit_per_section=5)

        assert loops["counts"]["overdue_reminders"] >= 1
        assert loops["counts"]["due_today_reminders"] >= 1
        assert loops["counts"]["upcoming_events"] >= 1
        assert loops["counts"]["pending_goals"] >= 1
        assert loops["counts"]["due_decisions"] >= 1


class TestCleanup:
    def test_cleanup_removes_old_rows(self, tmp_db: Database, raw_conn):
        old_ts = "2025-01-01 00:00:00"
        recent_ts = datetime.now().strftime(SQL_TS_FMT)
        raw_conn.execute(
            "INSERT INTO interactions (timestamp, direction, content) VALUES (?, 'in', 'old')",
            (old_ts,),
        )
        raw_conn.execute(
            "INSERT INTO interactions (timestamp, direction, content) VALUES (?, 'in', 'recent')",
            (recent_ts,),
        )
        raw_conn.commit()

        counts = tmp_db.cleanup_old_data(days=90)
        assert counts["interactions"] >= 1

        contents = [r["content"] for r in raw_conn.execute(
            "SELECT content FROM interactions"
        ).fetchall()]
        assert "old" not in contents
        assert "recent" in contents

    def test_cleanup_hard_deletes_old_soft_deleted_memories(self, tmp_db: Database, raw_conn):
        raw_conn.execute(
            "INSERT INTO memories (id, content, embedding, category, status, last_accessed) "
            "VALUES ('a1', 'old deleted', x'', 'personal', 'deleted', '2025-01-01 00:00:00')",
        )
        raw_conn.execute(
            "INSERT INTO memories (id, content, embedding, category, status, last_accessed) "
            "VALUES ('a2', 'still active', x'', 'personal', 'active', '2025-01-01 00:00:00')",
        )
        raw_conn.commit()

        tmp_db.cleanup_old_data(days=90)
        ids = {r["id"] for r in raw_conn.execute("SELECT id FROM memories").fetchall()}
        assert "a1" not in ids
        assert "a2" in ids


class TestMigrations:
    def test_empty_migrations_dir_is_noop(self, tmp_path):
        db = Database(tmp_path / "test.db", migrations_dir=tmp_path / "empty_migrations")
        version = db.db.execute("PRAGMA user_version").fetchone()[0]
        assert version == 0

    def test_applies_migration_and_bumps_version(self, tmp_path):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "001_add_test.sql").write_text(
            "CREATE TABLE test_migration (id INTEGER PRIMARY KEY);",
            encoding="utf-8",
        )
        db = Database(tmp_path / "test.db", migrations_dir=migrations)

        assert db.db.execute("PRAGMA user_version").fetchone()[0] == 1
        rows = db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_migration'"
        ).fetchall()
        assert len(rows) == 1

    def test_idempotent_across_restarts(self, tmp_path):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "001_noop.sql").write_text("SELECT 1;", encoding="utf-8")
        db_path = tmp_path / "test.db"

        db1 = Database(db_path, migrations_dir=migrations)
        assert db1.db.execute("PRAGMA user_version").fetchone()[0] == 1
        db1.close()

        db2 = Database(db_path, migrations_dir=migrations)
        assert db2.db.execute("PRAGMA user_version").fetchone()[0] == 1


class TestEphemeralState:
    def test_set_and_get(self, tmp_db: Database):
        tmp_db.set_ephemeral_state("location", "на работе", ttl_hours=8)
        rows = tmp_db.get_active_ephemeral_state()
        assert len(rows) == 1
        assert rows[0]["key"] == "location"
        assert rows[0]["value"] == "на работе"

    def test_upsert_replaces(self, tmp_db: Database):
        tmp_db.set_ephemeral_state("location", "дома", ttl_hours=4)
        tmp_db.set_ephemeral_state("location", "на работе", ttl_hours=8)
        rows = tmp_db.get_active_ephemeral_state()
        assert len(rows) == 1
        assert rows[0]["value"] == "на работе"

    def test_ttl_clamped_too_small(self, tmp_db: Database):
        tmp_db.set_ephemeral_state("energy", "full", ttl_hours=0.01)
        rows = tmp_db.get_active_ephemeral_state()
        # Should have been clamped to 0.5h so it's still active
        assert len(rows) == 1

    def test_ttl_clamped_too_big(self, tmp_db: Database, raw_conn):
        tmp_db.set_ephemeral_state("health", "fine", ttl_hours=999)
        row = raw_conn.execute(
            "SELECT expires_at FROM ephemeral_state WHERE key = 'health'"
        ).fetchone()
        # Clamp at 72h — with any reasonable TZ skew between _now() and
        # datetime('now') (UTC), expires_at must still be within ~76h of
        # caller's wall clock, and NOT ~999h (41 days).
        expires = datetime.strptime(row["expires_at"], SQL_TS_FMT)
        delta_from_now = expires - datetime.now()
        assert delta_from_now <= timedelta(hours=76)
        assert delta_from_now >= timedelta(hours=60)

    def test_expired_rows_cleaned_on_read(self, tmp_db: Database, raw_conn):
        past = (datetime.now() - timedelta(hours=2)).strftime(SQL_TS_FMT)
        raw_conn.execute(
            "INSERT INTO ephemeral_state (key, value, expires_at) VALUES (?, ?, ?)",
            ("location", "expired", past),
        )
        raw_conn.commit()

        rows = tmp_db.get_active_ephemeral_state()
        assert rows == []
        # Row physically deleted
        count = raw_conn.execute(
            "SELECT COUNT(*) FROM ephemeral_state"
        ).fetchone()[0]
        assert count == 0

    def test_clear_all(self, tmp_db: Database):
        tmp_db.set_ephemeral_state("location", "на работе", ttl_hours=8)
        tmp_db.set_ephemeral_state("health", "OK", ttl_hours=24)
        n = tmp_db.clear_ephemeral_state()
        assert n == 2
        assert tmp_db.get_active_ephemeral_state() == []

    def test_clear_by_key(self, tmp_db: Database):
        tmp_db.set_ephemeral_state("location", "на работе", ttl_hours=8)
        tmp_db.set_ephemeral_state("health", "OK", ttl_hours=24)
        n = tmp_db.clear_ephemeral_state("location")
        assert n == 1
        remaining = tmp_db.get_active_ephemeral_state()
        assert len(remaining) == 1
        assert remaining[0]["key"] == "health"

    def test_clear_nonexistent(self, tmp_db: Database):
        n = tmp_db.clear_ephemeral_state("location")
        assert n == 0
