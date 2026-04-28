"""Tests for llm/tools.py — argument sanitization and tool execution."""
from __future__ import annotations

import pytest

from mindsecretary.llm.tools import (
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    _sanitize_args,
    _truncate,
)


class TestTruncate:
    def test_none_returns_none(self):
        assert _truncate(None) is None

    def test_short_string_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_long_string_truncated(self):
        result = _truncate("x" * 10000, max_len=100)
        assert len(result) == 100


class TestSanitizeArgs:
    def test_basic_string_passthrough(self):
        args = _sanitize_args("create_event", {"title": "Lunch", "start_at": "2026-04-15T12:00"})
        assert args["title"] == "Lunch"

    def test_int_and_bool_passthrough(self):
        args = _sanitize_args("log_habit", {"habit_name": "gym", "done": True})
        assert args["done"] is True

    def test_none_passthrough(self):
        args = _sanitize_args("create_event", {"title": "X", "location": None})
        assert args["location"] is None

    def test_non_primitive_coerced_to_string(self):
        args = _sanitize_args("test", {"data": ["a", "b"]})
        assert isinstance(args["data"], str)

    def test_save_memory_invalid_category_defaults(self):
        args = _sanitize_args("save_memory", {
            "content": "test",
            "category": "INVALID",
            "importance": 5,
        })
        assert args["category"] == "personal"

    def test_save_memory_valid_categories(self):
        for cat in VALID_CATEGORIES:
            args = _sanitize_args("save_memory", {
                "content": "test", "category": cat, "importance": 5,
            })
            assert args["category"] == cat

    def test_save_memory_importance_clamped(self):
        args = _sanitize_args("save_memory", {
            "content": "x", "category": "work", "importance": 99,
        })
        assert args["importance"] == 10

        args = _sanitize_args("save_memory", {
            "content": "x", "category": "work", "importance": -5,
        })
        assert args["importance"] == 1

    def test_event_invalid_priority_defaults(self):
        args = _sanitize_args("create_event", {
            "title": "X", "start_at": "now", "priority": "URGENT",
        })
        assert args["priority"] == "medium"

    def test_event_valid_priorities(self):
        for prio in VALID_PRIORITIES:
            args = _sanitize_args("create_event", {
                "title": "X", "start_at": "now", "priority": prio,
            })
            assert args["priority"] == prio

    def test_long_strings_truncated(self):
        args = _sanitize_args("save_memory", {
            "content": "x" * 10000,
            "category": "work",
            "importance": 5,
        })
        assert len(args["content"]) == 5000

    def test_get_recent_memories_limit_clamped(self):
        args = _sanitize_args("get_recent_memories", {"limit": 50})
        assert args["limit"] == 10

    def test_get_open_loops_days_clamped(self):
        args = _sanitize_args("get_open_loops", {"days_ahead": 99})
        assert args["days_ahead"] == 7

    def test_get_reminders_defaults(self):
        args = _sanitize_args("get_reminders", {})
        # days_ahead omitted = no window filter
        assert args.get("days_ahead") is None
        assert args["limit"] == 20

    def test_get_reminders_days_ahead_clamped(self):
        args_low = _sanitize_args("get_reminders", {"days_ahead": 0})
        assert args_low["days_ahead"] == 1

        args_high = _sanitize_args("get_reminders", {"days_ahead": 9999})
        assert args_high["days_ahead"] == 365

    def test_get_reminders_limit_clamped(self):
        args_low = _sanitize_args("get_reminders", {"limit": 0})
        assert args_low["limit"] == 1

        args_high = _sanitize_args("get_reminders", {"limit": 999})
        assert args_high["limit"] == 50

    def test_ephemeral_state_valid(self):
        args = _sanitize_args("set_ephemeral_state", {
            "key": "location",
            "value": "на работе до 18",
            "ttl_hours": 9,
        })
        assert args["key"] == "location"
        assert args["value"] == "на работе до 18"
        assert args["ttl_hours"] == 9

    def test_ephemeral_state_invalid_key_falls_back(self):
        args = _sanitize_args("set_ephemeral_state", {
            "key": "nonsense",
            "value": "x",
            "ttl_hours": 1,
        })
        assert args["key"] == "activity"

    def test_ephemeral_state_ttl_clamped(self):
        args_low = _sanitize_args("set_ephemeral_state", {
            "key": "health", "value": "OK", "ttl_hours": 0.001,
        })
        assert args_low["ttl_hours"] == 0.5

        args_high = _sanitize_args("set_ephemeral_state", {
            "key": "health", "value": "OK", "ttl_hours": 9999,
        })
        assert args_high["ttl_hours"] == 72.0

    def test_ephemeral_state_value_truncated(self):
        args = _sanitize_args("set_ephemeral_state", {
            "key": "activity",
            "value": "y" * 500,
            "ttl_hours": 1,
        })
        # Tighter cap than global MAX_STR_LEN — state lives in every prompt
        assert len(args["value"]) == 200

    def test_ephemeral_state_empty_value_defaults(self):
        args = _sanitize_args("set_ephemeral_state", {
            "key": "activity", "value": "", "ttl_hours": 1,
        })
        assert args["value"] == "активно"


class TestSearchConversationsSanitize:
    def test_days_clamped_to_365(self):
        args = _sanitize_args("search_conversations", {
            "query": "x", "days": 9999,
        })
        assert args["days"] == 365

    def test_days_clamped_to_1(self):
        args = _sanitize_args("search_conversations", {
            "query": "x", "days": 0,
        })
        assert args["days"] == 1

    def test_limit_clamped_to_30(self):
        args = _sanitize_args("search_conversations", {
            "query": "x", "limit": 500,
        })
        assert args["limit"] == 30

    def test_defaults_applied(self):
        args = _sanitize_args("search_conversations", {"query": "x"})
        assert args["days"] == 30
        assert args["limit"] == 10


class TestSearchConversationsHandler:
    """ToolExecutor handler for search_conversations must render DB rows
    into a clean string Claude can read — 'Ты:/Бот:/[label]' with
    timestamp, so Claude can cite exact quotes from past conversations."""

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # Use a literal token Alpha in all three messages so Russian
        # grammatical cases don't interfere with substring matching.
        tmp_db.log_interaction("in", "text", "обсуждали проект Alpha")
        tmp_db.log_interaction(
            "out", "notification", "⏰ Напоминание про Alpha",
            metadata={"kind": "reminder"},
        )
        tmp_db.log_interaction("out", "chat", "Проект Alpha интересный")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute(
            "search_conversations", {"query": "Alpha", "days": 30, "limit": 10},
        )

        assert "Ты: обсуждали проект Alpha" in result
        assert "Бот: Проект Alpha интересный" in result
        assert "[напоминание]" in result

    @pytest.mark.asyncio
    async def test_empty_result_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute(
            "search_conversations", {"query": "ничегонет"},
        )
        assert "Ничего не нашёл" in result
        assert "ничегонет" in result

    def test_ts_utc_to_local_converts_offset(self):
        """DB stores UTC timestamps; tool output must show them in the
        user's timezone — otherwise Claude quotes UTC times as if they
        were local ones, confusing the user about when things happened."""
        from mindsecretary.llm.tools import ToolExecutor

        # 10:00 UTC → 13:00 Moscow (UTC+3)
        out = ToolExecutor._ts_utc_to_local_str(
            "2026-04-23 10:00:00", "Europe/Moscow",
        )
        assert out == "2026-04-23 13:00"

    def test_ts_utc_to_local_falls_back_gracefully(self):
        from mindsecretary.llm.tools import ToolExecutor
        # No timezone → truncate to minute precision without conversion
        assert ToolExecutor._ts_utc_to_local_str("2026-04-23 10:00:00", None) == "2026-04-23 10:00"
        # Invalid timestamp string → graceful fallback
        assert ToolExecutor._ts_utc_to_local_str("bogus", "UTC") == "bogus"[:16]
        # Empty string
        assert ToolExecutor._ts_utc_to_local_str("", "UTC") == "?"

    @pytest.mark.asyncio
    async def test_truncated_content_gets_ellipsis(self, tmp_db):
        """A 300+ char message gets truncated in search output — signal
        the cut with an ellipsis so Claude knows the quote is partial."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        long_text = "длинное сообщение " * 30  # ~540 chars
        long_text += " UNIQUEMARKER"
        tmp_db.log_interaction("in", "text", long_text)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute(
            "search_conversations", {"query": "длинное"},
        )
        assert "…" in result
        # Trailing marker shouldn't appear — it's past char 300
        assert "UNIQUEMARKER" not in result


class TestGetRemindersHandler:
    """ToolExecutor handler for get_reminders. The LLM has create_reminder
    but used to be blind to its own pending list — user could ask 'что у
    меня в напоминаниях' and the bot would hallucinate."""

    @pytest.mark.asyncio
    async def test_empty_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_reminders", {})
        assert result == "Нет отложенных напоминаний."

    @pytest.mark.asyncio
    async def test_lists_pending_with_priority_and_recurrence(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # trigger_at is profile-local naive; tmp_db has no TZ so just use
        # a fixed string. Both reminders are far in the future so days_ahead
        # filter test won't false-positive on these.
        tmp_db.create_reminder("позвонить маме", "2099-01-15 18:00:00", "high")
        tmp_db.create_reminder("книга в библиотеку", "2099-02-01 12:00:00",
                               "medium", recurrence="weekly")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_reminders", {})

        # Sorted by trigger_at ASC — позвонить first
        first, second = result.split("\n", 1)
        assert "позвонить маме" in first
        assert "[high]" in first
        assert "книга" in second
        assert "(weekly)" in second  # recurrence tag rendered
        # Seconds dropped from output — minute precision is enough for users
        assert "18:00:00" not in first
        assert "18:00" in first

    @pytest.mark.asyncio
    async def test_days_ahead_filters_far_future_keeps_overdue(self, tmp_db):
        """Overdue reminders MUST always show — filter only restricts
        forward window. Otherwise a user asking 'что на эту неделю' would
        miss something they were supposed to do yesterday."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        overdue = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        soon = (now + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
        far = (now + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

        tmp_db.create_reminder("вчерашнее", overdue, "high")
        tmp_db.create_reminder("через 2 дня", soon, "medium")
        tmp_db.create_reminder("через месяц", far, "low")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_reminders", {"days_ahead": 7})

        assert "вчерашнее" in result   # overdue always kept
        assert "через 2 дня" in result  # within 7-day window
        assert "через месяц" not in result  # past window — dropped

    @pytest.mark.asyncio
    async def test_limit_truncates_output(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        for i in range(5):
            ts = (now + timedelta(days=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
            tmp_db.create_reminder(f"reminder {i}", ts, "medium")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_reminders", {"limit": 3})

        # 3 visible + 1 ellipsis line about the rest
        assert result.count("\n- ") == 2  # 3 items = 2 inter-line breaks
        assert "ещё 2" in result

    @pytest.mark.asyncio
    async def test_does_not_show_sent_reminders(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        r = tmp_db.create_reminder("done thing", "2099-01-01 10:00:00", "low")
        tmp_db.mark_reminder_sent(r["id"])
        tmp_db.create_reminder("still pending", "2099-01-02 10:00:00", "low")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_reminders", {})

        assert "still pending" in result
        assert "done thing" not in result
