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

    def test_count_notifications_today(self, tmp_db: Database):
        assert tmp_db.count_notifications_today() == 0
        tmp_db.log_interaction("out", "notification", "Reminder!")
        assert tmp_db.count_notifications_today() == 1


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
