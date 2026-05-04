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


class TestCancelReminderHandler:
    """ToolExecutor handler for cancel_reminder. Symmetric with
    create_reminder — closes the CRUD loop on reminders."""

    @pytest.mark.asyncio
    async def test_cancels_unique_match(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_reminder("дантист в среду", "2099-01-15 10:00:00", "high")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_reminder", {"text_hint": "дантист"})

        assert "Отменено" in result
        assert "дантист в среду" in result
        # Status flipped to cancelled — no longer in pending
        assert tmp_db.get_pending_reminders() == []

    @pytest.mark.asyncio
    async def test_no_match_returns_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_reminder", {"text_hint": "nope"})
        assert "Не нашёл" in result and "nope" in result

    @pytest.mark.asyncio
    async def test_ambiguity_disclosure(self, tmp_db):
        """If the hint matches multiple, the response must say so — Claude
        should ask the user to disambiguate or explicitly cancel the rest."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_reminder("дантист 1", "2099-01-15 10:00:00")
        tmp_db.create_reminder("дантист 2", "2099-02-15 10:00:00")
        tmp_db.create_reminder("дантист 3", "2099-03-15 10:00:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_reminder", {"text_hint": "дантист"})

        assert "Отменено: дантист 1" in result
        assert "Похожих ещё 2" in result

    @pytest.mark.asyncio
    async def test_empty_hint_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_reminder", {"text_hint": "   "})
        assert "non-empty" in result


class TestExecuteLatencyLogging:
    """ToolExecutor.execute logs elapsed wall-clock ms on both success
    and failure branches. The log line is the only ops signal for slow
    tools without adding tracing infrastructure — drift here is invisible
    until users complain about lag."""

    @pytest.mark.asyncio
    async def test_success_log_includes_ms(self, tmp_db, caplog):
        from unittest.mock import MagicMock
        import logging as _log
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        with caplog.at_level(_log.INFO, logger="mindsecretary.llm.tools"):
            await te.execute("get_reminders", {})
        oks = [
            r for r in caplog.records
            if r.levelno == _log.INFO and "Tool get_reminders OK" in r.message
        ]
        assert oks, "expected an OK log for get_reminders"
        # Format: "Tool X OK (Yms)" — ms is the trailing ops signal.
        assert "ms)" in oks[0].message
        # Sanity: the elapsed should be non-negative (the format is %.0f
        # so 0ms is valid for a fast no-op call).
        import re
        m = re.search(r"\((\d+)ms\)", oks[0].message)
        assert m is not None
        assert int(m.group(1)) >= 0

    @pytest.mark.asyncio
    async def test_failure_log_includes_ms(self, tmp_db, caplog):
        from unittest.mock import MagicMock
        import logging as _log
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        # Force a handler-side exception by making the underlying memory
        # call raise. update_memory hits memory.update_by_hint which we
        # patch to raise.
        from unittest.mock import AsyncMock
        memory.update_by_hint = AsyncMock(side_effect=RuntimeError("boom"))

        te = ToolExecutor(db=tmp_db, memory=memory)
        with caplog.at_level(_log.ERROR, logger="mindsecretary.llm.tools"):
            result = await te.execute(
                "update_memory",
                {"text_hint": "x", "new_content": "y"},
            )
        # Caller still gets a structured error message back
        assert "Error executing update_memory" in result
        # And ops sees timing alongside the failure type
        fails = [
            r for r in caplog.records
            if r.levelno == _log.ERROR and "Tool update_memory failed" in r.message
        ]
        assert fails
        # Format is "Tool X failed (Yms): ErrType" — both ms and type present
        assert "ms)" in fails[0].message
        assert "RuntimeError" in fails[0].message


class TestDeleteMemoryHandler:
    """ToolExecutor handler for delete_memory mirrors update_memory's
    branch dispatch: distinct user-facing string per status so Claude
    knows what to tell the user."""

    @pytest.mark.asyncio
    async def test_ok_returns_id_category_undo_hint(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.delete_by_hint = AsyncMock(return_value={
            "status": "ok",
            "memory": {"id": "abc12345", "content": "шахматы",
                       "category": "personal"},
        })

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("delete_memory", {"text_hint": "шахматы"})

        assert "Удалено" in result
        assert "abc12345" in result
        assert "[personal]" in result
        # Critical: tell the user about /undo so they can recover
        assert "/undo" in result

    @pytest.mark.asyncio
    async def test_not_found(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.delete_by_hint = AsyncMock(return_value={"status": "not_found"})

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("delete_memory", {"text_hint": "nope"})
        assert "Не нашёл" in result and "nope" in result

    @pytest.mark.asyncio
    async def test_ambiguous_refuses(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.delete_by_hint = AsyncMock(return_value={
            "status": "ambiguous",
            "count": 3,
            "samples": [
                {"id": "a1", "content": "факт A про шахматы"},
                {"id": "b2", "content": "факт B про шахматы"},
                {"id": "c3", "content": "факт C про шахматы"},
            ],
        })

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("delete_memory", {"text_hint": "шахматы"})

        # User-facing copy must NOT say "удалено" on the ambiguous path —
        # otherwise the user thinks deletion happened
        assert "удаление не выполнено" in result
        assert "[a1]" in result and "[b2]" in result and "[c3]" in result

    @pytest.mark.asyncio
    async def test_invalid_branch(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.delete_by_hint = AsyncMock(return_value={"status": "invalid"})

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("delete_memory", {"text_hint": ""})
        assert "non-empty" in result


class TestUpdateMemoryHandler:
    """ToolExecutor handler for update_memory. Covers the not_found,
    ambiguous, and embed_failed branches the underlying Memory method
    distinguishes — Claude needs distinct error messages, not a generic
    'failed'."""

    @pytest.mark.asyncio
    async def test_ok_returns_id_category_content(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.update_by_hint = AsyncMock(return_value={
            "status": "ok",
            "memory": {"id": "abc12345", "content": "работает в Сбере",
                       "category": "work"},
        })

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("update_memory", {
            "text_hint": "Yandex",
            "new_content": "работает в Сбере",
        })

        assert "Обновлено" in result
        assert "abc12345" in result
        assert "[work]" in result
        assert "работает в Сбере" in result

    @pytest.mark.asyncio
    async def test_not_found(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.update_by_hint = AsyncMock(return_value={"status": "not_found"})

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("update_memory", {
            "text_hint": "nope", "new_content": "x",
        })
        assert "Не нашёл" in result and "nope" in result

    @pytest.mark.asyncio
    async def test_ambiguous_lists_samples(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.update_by_hint = AsyncMock(return_value={
            "status": "ambiguous",
            "count": 3,
            "samples": [
                {"id": "a1", "content": "Yandex офис"},
                {"id": "b2", "content": "Yandex удалённо"},
                {"id": "c3", "content": "раньше Yandex"},
            ],
        })

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("update_memory", {
            "text_hint": "Yandex", "new_content": "Сбер",
        })

        assert "3 записей" in result
        assert "уточни" in result.lower()
        assert "[a1]" in result and "[b2]" in result and "[c3]" in result

    @pytest.mark.asyncio
    async def test_embed_failed_keeps_user_informed(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.update_by_hint = AsyncMock(return_value={"status": "embed_failed"})

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("update_memory", {
            "text_hint": "x", "new_content": "y",
        })
        # Distinct error so Claude can suggest "попробуй позже" vs hallucinating success
        assert "Voyage" in result and "не обновлена" in result

    @pytest.mark.asyncio
    async def test_invalid_args_branch(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.update_by_hint = AsyncMock(return_value={"status": "invalid"})

        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("update_memory", {
            "text_hint": "", "new_content": "y",
        })
        assert "non-empty" in result


class TestRescheduleReminderHandler:
    @pytest.mark.asyncio
    async def test_reschedules_unique_match(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_reminder("дантист в среду", "2099-01-15 10:00:00", "high")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_reminder", {
            "text_hint": "дантист",
            "new_trigger_at": "2099-01-20 14:00:00",
        })

        assert "Перенесено" in result
        assert "дантист в среду" in result
        assert "2099-01-20 14:00" in result
        # Persisted
        pending = tmp_db.get_pending_reminders()
        assert pending[0]["trigger_at"] == "2099-01-20 14:00:00"

    @pytest.mark.asyncio
    async def test_no_match(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_reminder", {
            "text_hint": "nope", "new_trigger_at": "2099-01-20 14:00:00",
        })
        assert "Не нашёл" in result

    @pytest.mark.asyncio
    async def test_ambiguity_disclosure(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_reminder("дантист 1", "2099-01-15 10:00:00")
        tmp_db.create_reminder("дантист 2", "2099-02-15 10:00:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_reminder", {
            "text_hint": "дантист", "new_trigger_at": "2099-01-20 14:00:00",
        })

        assert "Перенесено: дантист 1" in result
        assert "Похожих ещё 1" in result

    @pytest.mark.asyncio
    async def test_iso_t_separator_normalized(self, tmp_db):
        """LLM may pass YYYY-MM-DDTHH:MM (schema-suggested format) — sanitize
        normalizes to space separator before hitting the DB."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_reminder("дантист", "2099-01-15 10:00:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        await te.execute("reschedule_reminder", {
            "text_hint": "дантист",
            "new_trigger_at": "2099-01-20T14:00",  # T-separator
        })

        pending = tmp_db.get_pending_reminders()
        # Whatever exact format the sanitize layer normalized to, it must
        # contain a space (DB convention) and the new date+time
        assert "T" not in pending[0]["trigger_at"]
        assert "2099-01-20 14:00" in pending[0]["trigger_at"]

    @pytest.mark.asyncio
    async def test_empty_hint_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_reminder", {
            "text_hint": "  ", "new_trigger_at": "2099-01-20 14:00:00",
        })
        assert "non-empty" in result

    @pytest.mark.asyncio
    async def test_empty_new_trigger_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_reminder", {
            "text_hint": "x", "new_trigger_at": "",
        })
        assert "new_trigger_at" in result


class TestGetDecisionsHandler:
    """ToolExecutor handler for get_decisions. The LLM had track_decision
    + resolve_decision but the active list was only visible via
    get_open_loops, which filters to follow-ups DUE — actively-considered
    decisions in their initial window were invisible. Fills that gap."""

    @pytest.mark.asyncio
    async def test_empty_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_decisions", {})
        assert result == "Нет активных решений в процессе."

    @pytest.mark.asyncio
    async def test_lists_pending_with_context_and_date(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_decision(
            "купить велосипед", context="бюджет 50к, ездить на работу",
        )
        tmp_db.create_decision("сменить тариф телефона")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_decisions", {})

        # Both descriptions present
        assert "купить велосипед" in result
        assert "сменить тариф" in result
        # Context surfaces (em-dash separator) — only when present
        assert "бюджет 50к" in result
        # Date stamp shows up (YYYY-MM-DD prefix from created_at)
        from datetime import datetime
        today_prefix = datetime.utcnow().strftime("%Y-%m-%d")
        assert today_prefix in result

    @pytest.mark.asyncio
    async def test_excludes_resolved_decisions(self, tmp_db):
        """Resolved decisions belong to past_decisions, not pending —
        otherwise the user would see closed items as 'still in process'."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_decision("купить велосипед")
        tmp_db.resolve_decision_by_hint("велосипед", "купил")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_decisions", {})

        # No active rows left after the resolve
        assert result == "Нет активных решений в процессе."

    @pytest.mark.asyncio
    async def test_limit_clamped(self, tmp_db):
        """Limit param goes through _sanitize_args clamp [1, 30]."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        for i in range(5):
            tmp_db.create_decision(f"решение {i}")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        # limit=2 → only 2 lines
        result = await te.execute("get_decisions", {"limit": 2})
        assert result.count("\n") == 1  # 2 lines = 1 newline


class TestCancelEventHandler:
    """ToolExecutor handler for cancel_event. Mirror of cancel_reminder —
    closes the missing CRUD on events. User says 'отмени встречу с Машей'
    → bot can finally do it instead of asking the user to clear the calendar
    manually."""

    @pytest.mark.asyncio
    async def test_cancels_unique_match(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("ужин с Машей", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_event", {"text_hint": "Маш"})

        assert "Отменено" in result
        assert "ужин с Машей" in result

    @pytest.mark.asyncio
    async def test_no_match_returns_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_event", {"text_hint": "nothing"})
        assert "Не нашёл" in result and "nothing" in result

    @pytest.mark.asyncio
    async def test_ambiguity_disclosure(self, tmp_db):
        """When hint matches multiple future events, the response says
        'soonest cancelled, N more remain' so Claude can ask the user to
        disambiguate before cancelling further."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        for offset in (1, 5, 10):
            ts = (now + timedelta(days=offset)).strftime("%Y-%m-%d %H:%M:%S")
            tmp_db.create_event(f"встреча с Машей {offset}", ts)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_event", {"text_hint": "Маш"})

        assert "Отменено: встреча с Машей 1" in result
        assert "Похожих ещё 2" in result

    @pytest.mark.asyncio
    async def test_empty_hint_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_event", {"text_hint": "   "})
        assert "non-empty" in result

    @pytest.mark.asyncio
    async def test_does_not_cancel_past_events(self, tmp_db):
        """Critical safety: 'отмени встречу с прошлой недели' is not a
        meaningful operation — past events already happened. Past events
        must stay in the DB untouched even if the hint matches."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        past = (now - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("вчерашний ужин", past)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("cancel_event", {"text_hint": "ужин"})

        assert "Не нашёл" in result
        # Past event still exists, untouched
        rows = tmp_db.db.execute("SELECT title FROM events").fetchall()
        assert any(r["title"] == "вчерашний ужин" for r in rows)


class TestRescheduleEventHandler:
    """ToolExecutor handler for reschedule_event. Mirror of
    reschedule_reminder — single-tool semantics ('перенеси встречу на
    17:00') instead of forcing Claude into cancel + create_event."""

    @pytest.mark.asyncio
    async def test_reschedules_unique_match(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        new_time = (now + timedelta(days=2, hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("звонок с Сашей", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_event", {
            "text_hint": "Саш", "new_start_at": new_time,
        })

        assert "Перенесено" in result
        assert "звонок с Сашей" in result
        # DB row reflects the new start_at
        row = tmp_db.db.execute(
            "SELECT start_at FROM events WHERE title = 'звонок с Сашей'"
        ).fetchone()
        assert row["start_at"] == new_time

    @pytest.mark.asyncio
    async def test_normalizes_iso_T_separator(self, tmp_db):
        """LLM often emits ISO 'YYYY-MM-DDTHH:MM' — sanitizer must convert
        to space-separated 'YYYY-MM-DD HH:MM:SS' so it matches the DB
        convention. Otherwise lexical comparisons against other events
        break (' ' < 'T' in ASCII)."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("дантист", future)

        # Pick an ISO-formatted future time
        new_time_iso = (now + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S")
        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        await te.execute("reschedule_event", {
            "text_hint": "дантист", "new_start_at": new_time_iso,
        })
        row = tmp_db.db.execute(
            "SELECT start_at FROM events WHERE title = 'дантист'"
        ).fetchone()
        # Stored format must be space-separated, not "T"
        assert "T" not in row["start_at"]
        assert row["start_at"][:13] == new_time_iso[:13].replace("T", " ")

    @pytest.mark.asyncio
    async def test_no_match_returns_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_event", {
            "text_hint": "nothing", "new_start_at": "2099-04-15 10:00:00",
        })
        assert "Не нашёл" in result

    @pytest.mark.asyncio
    async def test_empty_hint_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_event", {
            "text_hint": "  ", "new_start_at": "2099-04-15 10:00:00",
        })
        assert "non-empty" in result

    @pytest.mark.asyncio
    async def test_empty_new_start_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_event", {
            "text_hint": "x", "new_start_at": "",
        })
        assert "new_start_at" in result


class TestGetHabitsHandler:
    """ToolExecutor handler for get_habits. The LLM had log_habit but no
    way to read habits back — user asking 'сколько уже бегаю?' got
    hallucinations. Now the LLM can answer from real data."""

    @pytest.mark.asyncio
    async def test_empty_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_habits", {})
        assert result == "Привычек пока нет."

    @pytest.mark.asyncio
    async def test_lists_habits_with_streak_and_last_done(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        today = tmp_db._now()
        for i in range(3):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            tmp_db.log_habit("бег", done=True, date=d)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_habits", {})

        assert "бег" in result
        # streak surfaced — user-facing wording
        assert "streak 3" in result
        # last_done_date rendered
        assert today.strftime("%Y-%m-%d") in result
        # 7-day completion ratio rendered
        assert "3/7" in result

    @pytest.mark.asyncio
    async def test_renders_never_for_skip_only_habit(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        today_str = tmp_db._now().strftime("%Y-%m-%d")
        tmp_db.log_habit("медитация", done=False, date=today_str)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_habits", {})

        assert "медитация" in result
        # When there's never been a done=True, the placeholder is "никогда"
        # so the LLM doesn't hallucinate a date.
        assert "последний раз: никогда" in result
