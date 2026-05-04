"""Tests for core/database.py — CRUD operations and timestamp handling."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

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

    def test_cancel_event_by_hint_picks_soonest_future(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        # Two future events matching hint, one past — cancel must pick the
        # soonest future one. Past events are excluded entirely.
        soon = (now + timedelta(days=2)).strftime(SQL_TS_FMT)
        later = (now + timedelta(days=10)).strftime(SQL_TS_FMT)
        past = (now - timedelta(days=2)).strftime(SQL_TS_FMT)
        tmp_db.create_event("ужин с Машей", soon)
        tmp_db.create_event("ужин с Машей в кафе", later)
        tmp_db.create_event("прошлый ужин с Машей", past)

        # Hint "Маш" matches all declensions ("Машей", "Маше", "Машу")
        # without forcing the test to know specific noun cases.
        cancelled = tmp_db.cancel_event_by_hint("Маш")
        assert cancelled is not None
        assert cancelled["start_at"] == soon
        # The cancelled one is hard-deleted, the later future one remains
        # plus the past one (untouched).
        rows = tmp_db.db.execute("SELECT title FROM events ORDER BY start_at").fetchall()
        titles = [r["title"] for r in rows]
        assert "ужин с Машей" not in titles  # hard-deleted
        assert "ужин с Машей в кафе" in titles
        assert "прошлый ужин с Машей" in titles  # past untouched

    def test_cancel_event_no_match_returns_none(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("дантист", future)
        assert tmp_db.cancel_event_by_hint("парикмахер") is None

    def test_cancel_event_matches_description(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        # Title doesn't match, but description does — searcher must look at both.
        tmp_db.create_event(
            "встреча", future, description="обсудить контракт с Сбером",
        )
        cancelled = tmp_db.cancel_event_by_hint("Сбер")
        assert cancelled is not None
        assert cancelled["title"] == "встреча"

    def test_count_future_events_matching_excludes_past(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        past = (now - timedelta(days=1)).strftime(SQL_TS_FMT)
        future = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("прошлый Маша", past)
        tmp_db.create_event("будущий Маша 1", future)
        tmp_db.create_event("будущий Маша 2",
                            (now + timedelta(days=2)).strftime(SQL_TS_FMT))
        # Past row matches the hint but isn't counted — symmetric with the
        # cancel/reschedule SELECTs.
        assert tmp_db.count_future_events_matching("Маш") == 2

    def test_count_future_events_empty_hint_zero(self, tmp_db: Database):
        assert tmp_db.count_future_events_matching("") == 0
        assert tmp_db.count_future_events_matching("   ") == 0

    def test_reschedule_event_picks_soonest_future(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        soon = (now + timedelta(days=2)).strftime(SQL_TS_FMT)
        later = (now + timedelta(days=10)).strftime(SQL_TS_FMT)
        tmp_db.create_event("ужин", soon)
        tmp_db.create_event("ужин с командой", later)

        new_time = (now + timedelta(days=5)).strftime(SQL_TS_FMT)
        updated = tmp_db.reschedule_event_by_hint("ужин", new_time)
        assert updated is not None
        # The soonest one moved; the other is untouched.
        assert updated["start_at"] == new_time
        # Verify in DB: the formerly-soonest now has new time, other still later
        rows = tmp_db.db.execute(
            "SELECT title, start_at FROM events ORDER BY start_at"
        ).fetchall()
        # Originally: soon (day+2) and later (day+10). After: new_time (day+5)
        # and later (day+10). Order by start_at: ужин at day+5, ужин с командой at day+10.
        assert rows[0]["title"] == "ужин"
        assert rows[0]["start_at"] == new_time
        assert rows[1]["title"] == "ужин с командой"
        assert rows[1]["start_at"] == later

    def test_reschedule_event_preserves_end_at_when_omitted(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        start = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        end = (now + timedelta(days=1, hours=2)).strftime(SQL_TS_FMT)
        tmp_db.create_event("встреча", start, end_at=end)

        new_start = (now + timedelta(days=2)).strftime(SQL_TS_FMT)
        # No new_end_at — the existing end_at must survive (common case:
        # "перенеси на завтра" without re-specifying duration).
        updated = tmp_db.reschedule_event_by_hint("встреча", new_start)
        assert updated["start_at"] == new_start
        row = tmp_db.db.execute(
            "SELECT end_at FROM events WHERE id = ?", (updated["id"],)
        ).fetchone()
        assert row["end_at"] == end

    def test_reschedule_event_updates_end_at_when_provided(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        start = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        end = (now + timedelta(days=1, hours=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("встреча", start, end_at=end)

        new_start = (now + timedelta(days=2)).strftime(SQL_TS_FMT)
        new_end = (now + timedelta(days=2, hours=3)).strftime(SQL_TS_FMT)
        updated = tmp_db.reschedule_event_by_hint(
            "встреча", new_start, new_end_at=new_end,
        )
        row = tmp_db.db.execute(
            "SELECT start_at, end_at FROM events WHERE id = ?", (updated["id"],)
        ).fetchone()
        assert row["start_at"] == new_start
        assert row["end_at"] == new_end

    def test_reschedule_event_excludes_past(self, tmp_db: Database):
        """Symmetric with cancel — past events can't be rescheduled, since
        they already happened. The hint matching must skip them."""
        now = tmp_db.local_now_naive()
        past = (now - timedelta(days=2)).strftime(SQL_TS_FMT)
        tmp_db.create_event("прошлая встреча", past)

        new_time = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        result = tmp_db.reschedule_event_by_hint("встреча", new_time)
        assert result is None

    def test_reschedule_event_empty_new_start_returns_none(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("встреча", future)
        assert tmp_db.reschedule_event_by_hint("встреча", "") is None
        assert tmp_db.reschedule_event_by_hint("встреча", "   ") is None

    def test_search_events_finds_future_match(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        soon = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("ужин с Машей", soon, location="кафе Пушкин")
        rows = tmp_db.search_events("кафе")
        assert len(rows) == 1
        assert rows[0]["title"] == "ужин с Машей"

    def test_search_events_excludes_past(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        past = (now - timedelta(days=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("прошлый Маша", past)
        # Past event matches the substring but must not surface — same
        # rationale as cancel/reschedule, asking 'когда встреча с Машей?'
        # for an event already past is meaningless.
        assert tmp_db.search_events("Маш") == []

    def test_search_events_orders_by_start_at(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        for offset in (10, 1, 5):
            ts = (now + timedelta(days=offset)).strftime(SQL_TS_FMT)
            tmp_db.create_event(f"встреча {offset}", ts)
        rows = tmp_db.search_events("встреча")
        # ASC: 1, 5, 10
        assert [r["title"] for r in rows] == ["встреча 1", "встреча 5", "встреча 10"]

    def test_search_events_empty_query_returns_empty(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("anything", future)
        assert tmp_db.search_events("") == []
        assert tmp_db.search_events("   ") == []

    def test_search_events_matches_related_person(self, tmp_db: Database):
        """Searcher must look at related_person too — common case is
        'когда встреча с Олегом?' where the title might just be 'обед'
        and Олег is in related_person."""
        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=2)).strftime(SQL_TS_FMT)
        tmp_db.create_event("обед", future, related_person="Олег")
        rows = tmp_db.search_events("Олег")
        assert len(rows) == 1
        assert rows[0]["title"] == "обед"

    def test_search_events_respects_days_ahead(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        in_5 = (now + timedelta(days=5)).strftime(SQL_TS_FMT)
        in_60 = (now + timedelta(days=60)).strftime(SQL_TS_FMT)
        tmp_db.create_event("ближний Маш", in_5)
        tmp_db.create_event("дальний Маш", in_60)
        rows = tmp_db.search_events("Маш", days_ahead=30)
        # Only the within-30-days match surfaces
        assert len(rows) == 1
        assert rows[0]["title"] == "ближний Маш"

    def test_search_events_respects_limit(self, tmp_db: Database):
        now = tmp_db.local_now_naive()
        for i in range(5):
            ts = (now + timedelta(days=i + 1)).strftime(SQL_TS_FMT)
            tmp_db.create_event(f"yoga {i}", ts)
        rows = tmp_db.search_events("yoga", limit=3)
        assert len(rows) == 3

    def test_cancel_event_cyrillic_case_insensitive(self, tmp_db: Database):
        """SQLite's native LOWER() is ASCII-only — must use pylower for
        case-insensitive Cyrillic matching, same as the reminder by-hint."""
        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime(SQL_TS_FMT)
        tmp_db.create_event("Встреча с Машей", future)
        # Use the stem "Маш" so noun declension doesn't break the match
        # — the case-insensitive part is what's under test.
        cancelled = tmp_db.cancel_event_by_hint("МАШ")
        assert cancelled is not None
        assert cancelled["title"] == "Встреча с Машей"


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

    def test_cancel_reminder_by_hint_picks_soonest(self, tmp_db: Database):
        """When hint matches multiple, cancel the most-imminent one — that's
        almost always what the user means by 'отмени напоминание про X'."""
        tmp_db.create_reminder("дантист в среду", "2099-01-15 10:00:00")
        tmp_db.create_reminder("дантист в апреле", "2099-04-10 10:00:00")
        tmp_db.create_reminder("парикмахер", "2099-01-20 10:00:00")

        cancelled = tmp_db.cancel_reminder_by_hint("дантист")
        assert cancelled is not None
        assert cancelled["text"] == "дантист в среду"  # earliest match

        # Other "дантист" still pending; non-matching reminder also pending
        pending = tmp_db.get_pending_reminders()
        texts = [p["text"] for p in pending]
        assert "дантист в апреле" in texts
        assert "парикмахер" in texts
        assert "дантист в среду" not in texts

    def test_cancel_reminder_by_hint_no_match(self, tmp_db: Database):
        tmp_db.create_reminder("X", "2099-01-15 10:00:00")
        result = tmp_db.cancel_reminder_by_hint("nonexistent")
        assert result is None

    def test_cancel_reminder_by_hint_cyrillic_case_insensitive(self, tmp_db: Database):
        """Hint 'СТОМАТОЛОГ' must match stored 'стоматолог' — pylower path."""
        tmp_db.create_reminder("стоматолог запись", "2099-01-15 10:00:00")
        cancelled = tmp_db.cancel_reminder_by_hint("СТОМАТОЛОГ")
        assert cancelled is not None
        assert cancelled["text"] == "стоматолог запись"

    def test_cancel_reminder_excludes_sent_reminders(self, tmp_db: Database):
        r = tmp_db.create_reminder("done thing", "2099-01-15 10:00:00")
        tmp_db.mark_reminder_sent(r["id"])
        # Same text still searchable but sent rows must NOT be cancellable
        result = tmp_db.cancel_reminder_by_hint("done thing")
        assert result is None

    def test_count_pending_reminders_matching(self, tmp_db: Database):
        tmp_db.create_reminder("дантист 1", "2099-01-15 10:00:00")
        tmp_db.create_reminder("дантист 2", "2099-02-15 10:00:00")
        tmp_db.create_reminder("парикмахер", "2099-01-20 10:00:00")
        assert tmp_db.count_pending_reminders_matching("дантист") == 2
        assert tmp_db.count_pending_reminders_matching("nope") == 0
        assert tmp_db.count_pending_reminders_matching("") == 0

    def test_reschedule_reminder_by_hint_picks_soonest(self, tmp_db: Database):
        """When hint matches multiple, reschedule the soonest — same
        intent shape as cancel_reminder_by_hint."""
        tmp_db.create_reminder("дантист в среду", "2099-01-15 10:00:00")
        tmp_db.create_reminder("дантист в апреле", "2099-04-10 10:00:00")
        tmp_db.create_reminder("парикмахер", "2099-01-20 10:00:00")

        updated = tmp_db.reschedule_reminder_by_hint(
            "дантист", "2099-01-20 14:00:00",
        )
        assert updated is not None
        assert updated["text"] == "дантист в среду"
        assert updated["trigger_at"] == "2099-01-20 14:00:00"

        # Verify persisted, and that the second 'дантист' is untouched
        all_pending = tmp_db.get_pending_reminders()
        assert len(all_pending) == 3  # nothing was deleted
        moved = next(r for r in all_pending if r["text"] == "дантист в среду")
        assert moved["trigger_at"] == "2099-01-20 14:00:00"
        same = next(r for r in all_pending if r["text"] == "дантист в апреле")
        assert same["trigger_at"] == "2099-04-10 10:00:00"

    def test_reschedule_reminder_no_match(self, tmp_db: Database):
        tmp_db.create_reminder("X", "2099-01-15 10:00:00")
        result = tmp_db.reschedule_reminder_by_hint(
            "nonexistent", "2099-01-16 10:00:00",
        )
        assert result is None

    def test_reschedule_reminder_cyrillic_case_insensitive(self, tmp_db: Database):
        tmp_db.create_reminder("стоматолог запись", "2099-01-15 10:00:00")
        updated = tmp_db.reschedule_reminder_by_hint(
            "СТОМАТОЛОГ", "2099-01-20 10:00:00",
        )
        assert updated is not None
        assert updated["trigger_at"] == "2099-01-20 10:00:00"

    def test_reschedule_excludes_sent_reminders(self, tmp_db: Database):
        r = tmp_db.create_reminder("done thing", "2099-01-15 10:00:00")
        tmp_db.mark_reminder_sent(r["id"])
        result = tmp_db.reschedule_reminder_by_hint(
            "done thing", "2099-01-20 10:00:00",
        )
        assert result is None  # sent rows are not reschedulable

    def test_reschedule_recurring_keeps_recurrence(self, tmp_db: Database):
        """Reschedule changes only trigger_at — recurrence metadata stays
        intact so the series continues from the new time."""
        tmp_db.create_reminder(
            "weekly thing", "2099-01-15 10:00:00",
            priority="medium", recurrence="weekly",
        )
        tmp_db.reschedule_reminder_by_hint(
            "weekly thing", "2099-01-22 10:00:00",
        )
        pending = tmp_db.get_pending_reminders()
        assert len(pending) == 1
        assert pending[0]["recurrence"] == "weekly"
        assert pending[0]["trigger_at"] == "2099-01-22 10:00:00"

    def test_reschedule_empty_args_return_none(self, tmp_db: Database):
        tmp_db.create_reminder("X", "2099-01-15 10:00:00")
        assert tmp_db.reschedule_reminder_by_hint("", "2099-01-20 10:00:00") is None
        assert tmp_db.reschedule_reminder_by_hint("X", "") is None

    def test_cancel_reminder_recurring_does_not_create_next(self, tmp_db: Database):
        """Cancellation stops the series — NO auto-roll like mark_reminder_sent.
        User intent on 'отмени' is 'stop bothering me', not 'skip this one'."""
        tmp_db.create_reminder(
            "weekly thing", "2099-01-15 10:00:00",
            priority="medium", recurrence="weekly",
        )
        tmp_db.cancel_reminder_by_hint("weekly thing")
        pending = tmp_db.get_pending_reminders()
        assert pending == []  # No next-week instance was auto-created


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

    # --- Cyrillic-aware case folding (pylower) ---
    # SQLite's native lower() is ASCII-only — without pylower these would
    # silently fail for Russian users (~the entire user base of this app).

    def test_upsert_dedup_cyrillic_case_insensitive(self, tmp_db: Database):
        """Saving 'Иван' then 'иван' should update the same row, not create
        a duplicate. Native SQLite lower() leaves Cyrillic untouched —
        regression risk if pylower wiring breaks."""
        tmp_db.upsert_contact("Иван", relation="брат")
        c2 = tmp_db.upsert_contact("иван", notes="любит шахматы")

        assert c2["name"] == "Иван"  # original casing preserved
        assert "шахматы" in (c2.get("notes") or "")
        # Only one contact row exists
        all_contacts = tmp_db.get_contacts("иван")
        assert len(all_contacts) == 1

    def test_get_contacts_cyrillic_case_insensitive(self, tmp_db: Database):
        tmp_db.upsert_contact("Маша", relation="коллега")

        # Lower-case query against capitalized stored name
        assert len(tmp_db.get_contacts("маша")) == 1
        # Upper-case query against capitalized stored name (SQLite lower()
        # would fail this — pylower passes)
        assert len(tmp_db.get_contacts("МАША")) == 1
        # Substring on relation
        assert len(tmp_db.get_contacts("КОЛЛЕГА")) == 1

    def test_log_habit_dedup_cyrillic_case_insensitive(self, tmp_db: Database):
        """Habit name 'Зарядка' followed by 'зарядка' should merge into one
        habit row — same Cyrillic case-fold issue as contacts."""
        tmp_db.log_habit("Зарядка", True)
        tmp_db.log_habit("зарядка", True)
        tmp_db.log_habit("ЗАРЯДКА", False)

        rows = tmp_db.db.execute(
            "SELECT COUNT(*) FROM habits WHERE pylower(name) = 'зарядка'"
        ).fetchone()
        assert rows[0] == 1  # All three log_habit calls hit the same habit row


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

    def test_complete_goal_cyrillic_case_insensitive(self, tmp_db: Database):
        """Title 'зарядка' must match LLM-supplied hint 'ЗАРЯДКА' — pylower
        path. SQLite native lower() leaves Cyrillic untouched, so the
        old ASCII-only LIKE missed common case mismatches."""
        tmp_db.create_daily_goal("сделать зарядку утром")
        result = tmp_db.complete_daily_goal_by_hint("ЗАРЯДКУ", status="completed")
        assert result is not None
        assert result["status"] == "completed"
        assert "зарядку" in result["title"]

    def test_skip_stale_pending_goals_marks_yesterday(self, tmp_db: Database):
        """Pending goals from prior days must auto-flip to skipped — they
        clutter completion-rate analytics if left alone."""
        from datetime import timedelta as _td
        today = tmp_db._now()
        yesterday = (today - _td(days=1)).strftime("%Y-%m-%d")
        last_week = (today - _td(days=7)).strftime("%Y-%m-%d")
        today_str = today.strftime("%Y-%m-%d")

        tmp_db.create_daily_goal("вчерашнее", date=yesterday)
        tmp_db.create_daily_goal("на прошлой неделе", date=last_week)
        tmp_db.create_daily_goal("сегодняшнее", date=today_str)

        skipped = tmp_db.skip_stale_pending_goals()
        assert skipped == 2  # yesterday + last_week, NOT today

        rows = {
            r["title"]: r for r in tmp_db.db.execute(
                "SELECT title, status, reflection FROM daily_goals"
            ).fetchall()
        }
        assert rows["вчерашнее"]["status"] == "skipped"
        assert "[авто]" in (rows["вчерашнее"]["reflection"] or "")
        assert rows["на прошлой неделе"]["status"] == "skipped"
        assert rows["сегодняшнее"]["status"] == "pending"

    def test_skip_stale_pending_goals_does_not_overwrite_completed(self, tmp_db: Database):
        """Already-resolved goals (completed/skipped/partial) must NOT be
        re-marked. The auto-skip only touches genuinely-abandoned 'pending'
        rows from prior days — no rewriting history."""
        from datetime import timedelta as _td
        yesterday = (tmp_db._now() - _td(days=1)).strftime("%Y-%m-%d")
        for status in ("completed", "skipped", "partial"):
            tmp_db.db.execute(
                "INSERT INTO daily_goals (date, title, status, reflection) "
                "VALUES (?, ?, ?, 'user-provided')",
                (yesterday, f"goal-{status}", status),
            )
        tmp_db.db.commit()

        tmp_db.skip_stale_pending_goals()

        rows = tmp_db.db.execute(
            "SELECT title, status, reflection FROM daily_goals"
        ).fetchall()
        for r in rows:
            assert r["status"] != "pending"
            # Original reflection must NOT be replaced by the auto-note
            assert r["reflection"] == "user-provided"

    def test_skip_stale_preserves_existing_reflection(self, tmp_db: Database):
        """If a stale-pending row somehow already has a reflection, COALESCE
        keeps it — auto-note only fills NULL. Edge case but worth locking
        in: user's words always outrank the bot's bookkeeping."""
        from datetime import timedelta as _td
        yesterday = (tmp_db._now() - _td(days=1)).strftime("%Y-%m-%d")
        tmp_db.db.execute(
            "INSERT INTO daily_goals (date, title, status, reflection) "
            "VALUES (?, 'X', 'pending', 'был занят, не успел')",
            (yesterday,),
        )
        tmp_db.db.commit()

        tmp_db.skip_stale_pending_goals()

        row = tmp_db.db.execute(
            "SELECT status, reflection FROM daily_goals WHERE title = 'X'"
        ).fetchone()
        assert row["status"] == "skipped"
        assert row["reflection"] == "был занят, не успел"


class TestDiary:
    def test_save_and_get_diary(self, tmp_db: Database):
        # Use db's own clock so the test isn't tied to a hardcoded date that
        # drifts out of the 7-day window as real time advances.
        today = tmp_db._now().strftime("%Y-%m-%d")
        tmp_db.save_diary_entry(today, "Good day.", mood="positive")
        entries = tmp_db.get_diary_entries(days=7)
        assert len(entries) == 1
        assert entries[0]["mood"] == "positive"

    def test_diary_upsert(self, tmp_db: Database):
        today = tmp_db._now().strftime("%Y-%m-%d")
        tmp_db.save_diary_entry(today, "Morning.")
        tmp_db.save_diary_entry(today, "Updated.")
        entries = tmp_db.get_diary_entries(days=7)
        assert len(entries) == 1
        assert entries[0]["content"] == "Updated."


class TestStats:
    def test_get_stats_empty(self, tmp_db: Database):
        stats = tmp_db.get_stats()
        assert stats["memories"] == 0
        assert stats["contacts"] == 0
        assert stats["today_cost"] == 0
        assert stats["memory_categories"] == []  # empty stats → empty breakdown

    def test_log_cost(self, tmp_db: Database):
        tmp_db.log_cost("anthropic", input_tokens=1000, output_tokens=500)
        stats = tmp_db.get_stats()
        assert stats["today_cost"] > 0
        assert stats["today_tokens"] == 1500

    def test_monthly_projection_below_3day_threshold_returns_none(
        self, tmp_db: Database,
    ):
        """A single day of cost data is too noisy to project — projecting
        $0.30 daily to a month is fine in theory, but real usage rarely
        looks like the first day. Threshold prevents misleading the user
        with spiky early-data extrapolations."""
        tmp_db.log_cost("anthropic", input_tokens=100_000, output_tokens=0)
        stats = tmp_db.get_stats()
        assert stats["month_projection"] is None

    def test_monthly_projection_renders_with_3plus_days(self, tmp_db: Database):
        """At ≥3 days of data, projection becomes meaningful and surfaces.
        Math: avg(week_trend) * 30. Direct insertion bypasses the 1-row-
        per-call accumulation of log_cost so we can hit the threshold."""
        from datetime import datetime, timezone, timedelta
        # Insert 3 distinct days of cost rows directly into api_costs
        # (UTC-naive, matching the storage convention).
        for offset in range(3):
            day = datetime.now(timezone.utc) - timedelta(days=offset)
            tmp_db.db.execute(
                "INSERT INTO api_costs (provider, input_tokens, "
                "output_tokens, cost_usd, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("anthropic", 0, 0, 0.50,
                 day.strftime("%Y-%m-%d %H:%M:%S")),
            )
        tmp_db.db.commit()
        stats = tmp_db.get_stats()
        assert stats["month_projection"] is not None
        # avg = 0.50; projection = 0.50 * 30 = 15.0
        assert 14.0 < stats["month_projection"] < 16.0

    def test_monthly_projection_none_when_no_costs(self, tmp_db: Database):
        """Day-1 installs (no cost rows yet) get None instead of $0.00 —
        the UI hides the line entirely rather than misleading users with
        a prediction based on a single empty day."""
        stats = tmp_db.get_stats()
        assert stats["month_projection"] is None

    def test_memory_category_breakdown_sorted_desc(self, tmp_db: Database):
        """Breakdown surfaces what kinds of facts are accumulating —
        sorted by count desc so the dominant categories show first."""
        # Insert directly to avoid Memory.save's embed dependency
        for cat, count in [("work", 5), ("personal", 3), ("health", 1)]:
            for i in range(count):
                tmp_db.db.execute(
                    "INSERT INTO memories (id, content, embedding, category, status) "
                    "VALUES (?, ?, x'', ?, 'active')",
                    (f"{cat}{i}", f"fact {cat} {i}", cat),
                )
        tmp_db.db.commit()

        stats = tmp_db.get_stats()
        breakdown = stats["memory_categories"]
        # Sorted by count desc: work=5, personal=3, health=1
        assert [c["category"] for c in breakdown] == ["work", "personal", "health"]
        assert [c["count"] for c in breakdown] == [5, 3, 1]
        assert stats["memories"] == 9  # total matches sum

    def test_memory_category_breakdown_excludes_deleted(self, tmp_db: Database):
        """Soft-deleted memories must NOT appear in the breakdown — same
        filter as the bare `memories` count, otherwise totals diverge."""
        tmp_db.db.execute(
            "INSERT INTO memories (id, content, embedding, category, status) "
            "VALUES ('a1', 'kept', x'', 'work', 'active')"
        )
        tmp_db.db.execute(
            "INSERT INTO memories (id, content, embedding, category, status) "
            "VALUES ('a2', 'gone', x'', 'work', 'deleted')"
        )
        tmp_db.db.commit()

        stats = tmp_db.get_stats()
        # Only one active 'work' memory — deleted one shouldn't show up
        assert stats["memory_categories"] == [{"category": "work", "count": 1}]
        assert stats["memories"] == 1


class TestCostBreaker:
    def test_get_today_cost_empty(self, tmp_db: Database):
        assert tmp_db.get_today_cost() == 0.0

    def test_get_today_cost_sums_all_providers(self, tmp_db: Database):
        tmp_db.log_cost("anthropic", input_tokens=1000, output_tokens=500)
        tmp_db.log_cost("groq", input_tokens=200, output_tokens=0)
        tmp_db.log_cost("voyage", input_tokens=100, output_tokens=0)
        assert tmp_db.get_today_cost() > 0.0


class TestOpenLoops:
    def test_get_open_loops_collects_pending_items(self, tmp_db: Database, monkeypatch):
        # Pin the DB clock to noon local so trigger offsets (+/- 2-3h)
        # land squarely in "today" and "yesterday" / "later today"
        # regardless of when the suite actually runs.
        anchor = datetime(2026, 4, 25, 12, 0, 0)
        monkeypatch.setattr(tmp_db, "_now", lambda: anchor)
        tmp_db.create_reminder("Overdue", (anchor - timedelta(hours=2)).strftime(SQL_TS_FMT))
        tmp_db.create_reminder("Later today", (anchor + timedelta(hours=2)).strftime(SQL_TS_FMT))
        tmp_db.create_event("Meeting", (anchor + timedelta(hours=3)).strftime(SQL_TS_FMT))
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

    def test_cleanup_cutoff_aligns_with_utc_storage(self, tmp_path):
        """Regression: cleanup_old_data used `self._now()` (profile-local
        naive) for the cutoff while interactions.timestamp / api_costs are
        UTC-naive. On a profile TZ like Asia/Almaty (UTC+5), a row written
        at "now - 90 days, exactly" UTC would be wrongly evaluated against
        a local-clock cutoff drifted by 5 hours, leading to retention
        skew on every weekly cleanup.

        Test rebuilds the scenario: profile TZ in the future (UTC+5),
        retention 90 days, a row stamped at 89 days 23 hours ago in UTC.
        Correct behavior: row is KEPT (still inside window). Pre-fix
        would compare against a local cutoff 5h ahead → row deleted.
        """
        from datetime import datetime, timedelta, timezone
        db = Database(tmp_path / "test.db", timezone="Asia/Almaty")
        # memories is created by Memory class normally — replicate the
        # conftest tmp_db fixture's workaround so cleanup's three DELETEs
        # all run.
        db.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                category TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                related_person TEXT,
                related_date TEXT,
                source_type TEXT,
                source_ref TEXT,
                confidence REAL DEFAULT 1.0,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT
            )
        """)
        db.db.commit()
        # Row 89d 23h old in UTC — well inside a 90-day retention window.
        almost_90d_old_utc = (
            datetime.now(timezone.utc) - timedelta(days=89, hours=23)
        ).strftime("%Y-%m-%d %H:%M:%S")
        db.db.execute(
            "INSERT INTO interactions (timestamp, direction, content) "
            "VALUES (?, 'in', 'borderline')",
            (almost_90d_old_utc,),
        )
        db.db.commit()

        db.cleanup_old_data(days=90)

        rows = db.db.execute("SELECT content FROM interactions").fetchall()
        contents = [r["content"] for r in rows]
        assert "borderline" in contents, (
            "89d 23h-old UTC row was deleted by 90-day retention — "
            "cleanup cutoff is using profile-local clock instead of UTC, "
            "creating a TZ-offset drift in the retention window"
        )

    def test_cleanup_invokes_skip_stale_pending_goals(self, tmp_db: Database):
        """cleanup_old_data must invoke the goal sweep — without this,
        stale goals never get auto-marked even with retention enabled."""
        from datetime import timedelta as _td
        yesterday = (tmp_db._now() - _td(days=1)).strftime("%Y-%m-%d")
        tmp_db.create_daily_goal("ghost", date=yesterday)

        counts = tmp_db.cleanup_old_data(days=90)

        assert counts["stale_goals"] == 1
        row = tmp_db.db.execute(
            "SELECT status FROM daily_goals WHERE title = 'ghost'"
        ).fetchone()
        assert row["status"] == "skipped"

    def test_cleanup_runs_goal_sweep_even_when_retention_disabled(
        self, tmp_db: Database,
    ):
        """The stale-goal sweep is independent of retention horizon — it
        runs even when days <= 0 (the 'disabled' setting). Otherwise users
        who turn off retention would lose this hygiene entirely."""
        from datetime import timedelta as _td
        yesterday = (tmp_db._now() - _td(days=1)).strftime("%Y-%m-%d")
        tmp_db.create_daily_goal("ghost", date=yesterday)

        counts = tmp_db.cleanup_old_data(days=0)

        assert counts["stale_goals"] == 1
        # Other counts stay zero — retention disabled, no row deletions
        assert counts["interactions"] == 0
        assert counts["api_costs"] == 0
        assert counts["memories"] == 0

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


class TestBackup:
    """Built-in DB backup mirrors scripts/backup.sh: writes
    mindsecretary_TIMESTAMP.db into <db_path>.parent/'backups', keeps
    the latest N, deletes the rest. Failure is best-effort — never
    propagates so a glitchy disk doesn't kill the scheduler."""

    def test_create_backup_writes_file_with_data(self, tmp_path, tmp_db):
        """Online backup includes current data, not just an empty schema.
        Catches "backup runs but on the wrong connection" regressions."""
        # Seed a row so we can verify the backup actually has it
        tmp_db.create_reminder("backup test", "2099-01-01 10:00:00")

        result = tmp_db.create_backup(keep=30)

        assert result["ok"] is True
        assert result["path"] is not None
        backup_path = Path(result["path"])
        assert backup_path.exists()
        # Open the backup as a fresh connection and verify the row landed
        import sqlite3 as _sq
        copy = _sq.connect(str(backup_path))
        copy.row_factory = _sq.Row
        rows = copy.execute(
            "SELECT text FROM reminders WHERE text = 'backup test'"
        ).fetchall()
        copy.close()
        assert len(rows) == 1

    def test_backup_directory_auto_created(self, tmp_path, tmp_db):
        """First-run case: backups/ doesn't exist yet. create_backup
        creates it instead of failing on FileNotFoundError."""
        backup_dir = tmp_db._db_path.parent / "backups"
        if backup_dir.exists():
            for f in backup_dir.iterdir():
                f.unlink()
            backup_dir.rmdir()
        assert not backup_dir.exists()

        result = tmp_db.create_backup()

        assert result["ok"] is True
        assert backup_dir.exists()

    def test_prune_keeps_only_latest_n(self, tmp_path, tmp_db):
        """Retention prunes oldest by mtime. With 5 backup files and
        keep=2, expect 2 files left and pruned=3 in the result."""
        import time
        backup_dir = tmp_db._db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Create 5 dummy backup files with staggered mtimes so prune
        # has a deterministic ordering target.
        files = []
        for i in range(5):
            f = backup_dir / f"mindsecretary_2026010{i + 1}_120000.db"
            f.write_bytes(b"")
            # Backdate older files so mtime ordering matches the suffix
            mtime = time.time() - (5 - i) * 60
            import os
            os.utime(f, (mtime, mtime))
            files.append(f)

        # New backup → 6 files → prune keeps last 2 → deletes 4 (the
        # 5 stale + the new... wait, new is "newest"). Result: keep=2
        # means {newest backup, second-newest = files[4]}, prunes 4.
        result = tmp_db.create_backup(keep=2)

        assert result["ok"] is True
        assert result["pruned"] == 4
        remaining = sorted(backup_dir.glob("mindsecretary_*.db"))
        assert len(remaining) == 2

    def test_create_backup_failure_returns_shape_doesnt_raise(self, tmp_db):
        """If sqlite3.Connection.backup raises (e.g. disk full), we
        catch + log + return a failure-shape dict. Caller (scheduler)
        treats the result as advisory only.

        sqlite3.Connection methods are read-only (can't monkey-patch
        .backup directly), so swap the whole connection for a Mock
        whose backup() raises."""
        from unittest.mock import MagicMock
        import sqlite3 as _sq

        original_conn = tmp_db.db
        mock_conn = MagicMock()
        mock_conn.backup.side_effect = _sq.OperationalError("disk full")
        tmp_db.db = mock_conn
        try:
            result = tmp_db.create_backup()
        finally:
            tmp_db.db = original_conn

        assert result["ok"] is False
        assert result["error"] == "OperationalError"
        assert result["path"] is None
        # Partial backup file shouldn't linger — verify the dir is empty
        # of any new files (older fixture-created ones may still exist
        # from earlier tests, just check there's no fresh failure-stub).
        backup_dir = tmp_db._db_path.parent / "backups"
        if backup_dir.exists():
            new_files = [
                f for f in backup_dir.iterdir()
                if f.stat().st_size == 0  # the partial-stub would be empty
            ]
            # Empty stubs would mean the backup was attempted but failed
            # mid-write. We deleted them, so this list should be empty.
            assert not new_files

    def test_prune_handles_empty_directory(self, tmp_db):
        """No backups yet → prune returns 0, doesn't crash on empty glob."""
        backup_dir = tmp_db._db_path.parent / "backups_empty"
        backup_dir.mkdir(parents=True, exist_ok=True)
        assert tmp_db._prune_backups(backup_dir, keep=10) == 0

    def test_prune_handles_missing_directory(self, tmp_db):
        """Missing dir returns 0 instead of raising — robust to first-run."""
        nonexistent = tmp_db._db_path.parent / "nope"
        assert tmp_db._prune_backups(nonexistent, keep=10) == 0


class TestAnniversaries:
    """`get_anniversaries` surfaces past items on this calendar date —
    powers the "год назад ты решил X" line in morning briefing."""

    def test_empty_db_returns_empty(self, tmp_db):
        assert tmp_db.get_anniversaries() == []

    def test_high_importance_memory_from_past_year_surfaces(self, tmp_db):
        """A memory created exactly N years ago today, importance ≥ 7,
        must appear with the kind/category labels the briefing renders.

        Year-back replace() preserves MM-DD even across leap years
        (Feb 29 → Feb 28 fallback isn't an issue here)."""
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        past = now_utc.replace(year=now_utc.year - 1)
        past_ts = past.strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.db.execute(
            "INSERT INTO memories (id, content, embedding, category, "
            "importance, status, created_at) "
            "VALUES ('a1', 'big move', x'', 'decision', 8, 'active', ?)",
            (past_ts,),
        )
        tmp_db.db.commit()

        items = tmp_db.get_anniversaries()
        assert len(items) >= 1
        m = next((x for x in items if x["content"] == "big move"), None)
        assert m is not None
        assert m["kind"] == "memory"
        assert m["category"] == "decision"
        assert m["age_days"] >= 360  # ~year ago

    def test_low_importance_memory_excluded(self, tmp_db):
        """Threshold is importance ≥ 7. Low-importance noise shouldn't
        clutter the briefing — most stuff isn't anniversary-worthy."""
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        past = now_utc.replace(year=now_utc.year - 1)
        past_ts = past.strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.db.execute(
            "INSERT INTO memories (id, content, embedding, category, "
            "importance, status, created_at) "
            "VALUES ('low', 'random fact', x'', 'personal', 4, 'active', ?)",
            (past_ts,),
        )
        tmp_db.db.commit()

        items = tmp_db.get_anniversaries()
        assert all(x["content"] != "random fact" for x in items)

    def test_recent_items_excluded_by_min_age(self, tmp_db):
        """Items younger than min_age_days don't count — yesterday's
        decision isn't its own anniversary tomorrow."""
        from datetime import datetime, timezone, timedelta
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        recent_ts = recent.strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.db.execute(
            "INSERT INTO memories (id, content, embedding, category, "
            "importance, status, created_at) "
            "VALUES ('r1', 'this week thing', x'', 'work', 9, 'active', ?)",
            (recent_ts,),
        )
        tmp_db.db.commit()

        items = tmp_db.get_anniversaries(min_age_days=30)
        assert all(x["content"] != "this week thing" for x in items)

    def test_resolved_decision_anniversary_includes_outcome(self, tmp_db):
        """Past resolved decisions are the most resonant — surface with
        outcome + sentiment so the briefing can frame as 'решил X — Y'.

        Anchor on SAME MM-DD a year back (not days=400, which lands on a
        different calendar date and silently misses the substr match)."""
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        past = now_utc.replace(year=now_utc.year - 1)
        past_ts = past.strftime("%Y-%m-%d %H:%M:%S")
        # Insert directly: create_decision uses _now() for created_at,
        # so we'd need to manually update. Easier to insert raw.
        tmp_db.db.execute(
            "INSERT INTO decisions (id, description, outcome, "
            "outcome_sentiment, status, created_at) "
            "VALUES ('d1', 'change job', 'great move', 'positive', "
            "'resolved', ?)",
            (past_ts,),
        )
        tmp_db.db.commit()

        items = tmp_db.get_anniversaries()
        d = next((x for x in items if x["kind"] == "decision"), None)
        assert d is not None
        assert d["content"] == "change job"
        assert d["outcome"] == "great move"
        assert d["sentiment"] == "positive"

    def test_pending_decisions_not_surfaced(self, tmp_db):
        """Only resolved decisions become anniversaries — pending ones
        are still open loops, not memories."""
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        past = now_utc.replace(year=now_utc.year - 1)
        past_ts = past.strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.db.execute(
            "INSERT INTO decisions (id, description, status, created_at) "
            "VALUES ('open', 'thinking about move', 'pending', ?)",
            (past_ts,),
        )
        tmp_db.db.commit()
        items = tmp_db.get_anniversaries()
        assert all(x["content"] != "thinking about move" for x in items)

    def test_results_sorted_by_age_desc(self, tmp_db):
        """Older items come first — 'год назад' lands stronger than
        'месяц назад' for the briefing's lead anniversary line.

        Anchor BOTH on same MM-DD as today (year-back / years-back) so
        the substr match catches both."""
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        # 1 year ago + 2 years ago, both same MM-DD as today
        for tag, years in [("recent", 1), ("oldest", 2)]:
            past = now_utc.replace(year=now_utc.year - years)
            past_ts = past.strftime("%Y-%m-%d %H:%M:%S")
            tmp_db.db.execute(
                "INSERT INTO memories (id, content, embedding, category, "
                "importance, status, created_at) "
                "VALUES (?, ?, x'', 'work', 8, 'active', ?)",
                (tag, f"item from {tag}", past_ts),
            )
        tmp_db.db.commit()

        items = tmp_db.get_anniversaries()
        # Both should appear; oldest first (2y > 1y)
        assert len(items) >= 2
        assert items[0]["age_days"] > items[1]["age_days"]
        assert "oldest" in items[0]["content"]
        assert "recent" in items[1]["content"]


class TestSnooze:
    """`set_snooze_until` / `get_snooze_until` / `is_snoozed_now` —
    persistent proactive-job mute via preferences. Stored UTC-naive so
    comparisons work directly against `datetime.now(utc)`."""

    def test_no_pref_means_not_snoozed(self, tmp_db):
        assert tmp_db.is_snoozed_now() is False
        assert tmp_db.get_snooze_until() is None

    def test_set_then_check_active(self, tmp_db):
        from datetime import datetime, timezone, timedelta
        until = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=2)
        tmp_db.set_snooze_until(until)
        assert tmp_db.is_snoozed_now() is True
        got = tmp_db.get_snooze_until()
        assert got is not None
        # Matches what we set within a second (storage is space-separated SQL fmt)
        assert abs((got - until).total_seconds()) < 2

    def test_past_deadline_treated_as_not_snoozed(self, tmp_db):
        """Stale rows from a past snooze auto-clear semantically — no
        cleanup job needed. Otherwise a forgotten /snooze 1h from
        yesterday would silently keep blocking briefings."""
        from datetime import datetime, timezone, timedelta
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        tmp_db.set_snooze_until(past)
        assert tmp_db.is_snoozed_now() is False
        assert tmp_db.get_snooze_until() is None

    def test_set_none_clears_snooze(self, tmp_db):
        from datetime import datetime, timezone, timedelta
        tmp_db.set_snooze_until(
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=2),
        )
        tmp_db.set_snooze_until(None)
        assert tmp_db.is_snoozed_now() is False

    def test_aware_datetime_normalized_to_utc_naive(self, tmp_db):
        """Caller may pass a tz-aware datetime — must store as UTC-naive
        so comparisons stay consistent with the rest of the app."""
        from datetime import datetime, timezone, timedelta
        from zoneinfo import ZoneInfo
        until_aware = datetime.now(ZoneInfo("Asia/Almaty")) + timedelta(hours=2)
        tmp_db.set_snooze_until(until_aware)
        got = tmp_db.get_snooze_until()
        assert got is not None
        assert got.tzinfo is None  # stored as naive
        # And the absolute moment in time matches
        expected_utc_naive = until_aware.astimezone(timezone.utc).replace(tzinfo=None)
        assert abs((got - expected_utc_naive).total_seconds()) < 2

    def test_corrupted_value_returns_none(self, tmp_db):
        """Garbage in the preference (manual edit, schema migration glitch)
        should NOT crash the scheduler — return None and let the bot
        proceed normally."""
        tmp_db.set_preference("proactive_snoozed_until", "not-a-date")
        assert tmp_db.get_snooze_until() is None
        assert tmp_db.is_snoozed_now() is False


class TestRecentUserMessages:
    """`has_recent_user_messages` powers the conversation-aware defer in
    the proactive scheduler. Cutoff is in UTC because interactions.
    timestamp is UTC-naive (datetime('now'))."""

    def test_returns_false_when_no_messages(self, tmp_db):
        assert tmp_db.has_recent_user_messages(minutes=5) is False

    def test_returns_true_for_recent_inbound(self, tmp_db):
        tmp_db.log_interaction("in", "text", "hello")
        assert tmp_db.has_recent_user_messages(minutes=5) is True

    def test_outbound_messages_ignored(self, tmp_db):
        """Bot replies must NOT count as user activity — otherwise every
        proactive send would trigger its own deferral on the next run."""
        tmp_db.log_interaction("out", "chat", "responding")
        tmp_db.log_interaction("out", "notification", "morning briefing")
        assert tmp_db.has_recent_user_messages(minutes=5) is False

    def test_old_messages_below_window_excluded(self, tmp_db):
        """Backdate a message to 10 minutes ago via direct insert so the
        UTC cutoff in the helper drops it — the 5-min window must reject
        anything older."""
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        tmp_db.db.execute(
            "INSERT INTO interactions (direction, message_type, content, timestamp) "
            "VALUES ('in', 'text', 'old', ?)",
            (old,),
        )
        tmp_db.db.commit()
        assert tmp_db.has_recent_user_messages(minutes=5) is False
        # Same row reachable when the window widens
        assert tmp_db.has_recent_user_messages(minutes=15) is True

    def test_zero_minutes_short_circuits_to_false(self, tmp_db):
        """`minutes=0` is treated as "no deferral window" — defensive
        path so a misconfigured threshold doesn't silently disable
        scheduled jobs."""
        tmp_db.log_interaction("in", "text", "now")
        assert tmp_db.has_recent_user_messages(minutes=0) is False


class TestIntegrityCheck:
    """`PRAGMA integrity_check` runs once on startup so ops sees DB
    corruption in the bot's log instead of as a cryptic query failure
    later. Healthy fresh DB → 'ok'; damaged → warning with detail."""

    def test_fresh_db_passes_integrity(self, tmp_db: Database, caplog):
        import logging as _log
        with caplog.at_level(_log.INFO):
            assert tmp_db._verify_integrity() is True
        # Verify the log surfaced — this is the ops signal we wanted
        assert any(
            "integrity_check: ok" in r.message for r in caplog.records
        )

    def test_corrupt_results_logged_and_return_false(self, tmp_db: Database, caplog):
        """Simulate corruption by swapping the DB connection for a mock
        whose execute() returns non-ok rows. The handler must log a
        WARNING with detail (so the user sees it in the daily log) and
        return False (so future code can decide whether to refuse
        risky operations)."""
        from unittest.mock import MagicMock
        import logging as _log

        fake_rows = [("malformed page on table memories",)]
        cursor = MagicMock()
        cursor.fetchall.return_value = fake_rows
        mock_conn = MagicMock()
        mock_conn.execute.return_value = cursor

        original_conn = tmp_db.db
        tmp_db.db = mock_conn
        try:
            with caplog.at_level(_log.WARNING):
                assert tmp_db._verify_integrity() is False
        finally:
            tmp_db.db = original_conn

        warnings = [
            r for r in caplog.records if r.levelno == _log.WARNING
            and "integrity_check" in r.message
        ]
        assert warnings, "expected an integrity_check warning in logs"
        assert "malformed page" in warnings[0].message

    def test_check_failure_returns_false_without_crashing(self, tmp_db: Database, caplog):
        """If the check itself raises (e.g. DB unreadable), we log + return
        False — don't crash the whole startup over a diagnostic step."""
        import sqlite3 as _sq
        from unittest.mock import MagicMock
        import logging as _log

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = _sq.DatabaseError(
            "file is encrypted or not a database",
        )

        original_conn = tmp_db.db
        tmp_db.db = mock_conn
        try:
            with caplog.at_level(_log.ERROR):
                assert tmp_db._verify_integrity() is False
        finally:
            tmp_db.db = original_conn

        errors = [
            r for r in caplog.records if r.levelno == _log.ERROR
            and "integrity_check itself failed" in r.message
        ]
        assert errors, "expected an integrity_check error log"


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
