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

    def test_search_memory_drops_invalid_category(self):
        """Asymmetric with save_memory: search uses None to mean 'all
        categories', so invalid input falls back to that instead of
        defaulting to 'personal' (which would restrict to one category
        when the LLM clearly wanted broader)."""
        args = _sanitize_args("search_memory", {
            "query": "что я знаю",
            "category": "tasks",  # not a real category
        })
        assert args["category"] is None

    def test_search_memory_passes_valid_category(self):
        for cat in VALID_CATEGORIES:
            args = _sanitize_args("search_memory", {
                "query": "x", "category": cat,
            })
            assert args["category"] == cat

    def test_track_decision_clamps_negative_follow_up(self):
        """Schema says 1-365 but defensive clamp protects against drift —
        negative follow_up_days makes follow_up_at land in the past, so
        the next-morning decision-followup tick prompts the user about a
        decision they just created."""
        args = _sanitize_args("track_decision", {
            "description": "x", "follow_up_days": -10,
        })
        assert args["follow_up_days"] == 1

    def test_track_decision_clamps_huge_follow_up(self):
        args = _sanitize_args("track_decision", {
            "description": "x", "follow_up_days": 99999,
        })
        assert args["follow_up_days"] == 365

    def test_track_decision_omitted_follow_up_unchanged(self):
        """When the LLM doesn't pass follow_up_days, the slot stays absent
        and the handler signature default (30) applies."""
        args = _sanitize_args("track_decision", {"description": "x"})
        assert "follow_up_days" not in args

    def test_set_daily_goal_invalid_priority_defaults(self):
        """Pre-fix only create_event/create_reminder validated priority
        in the sanitizer; set_daily_goal slipped through and the handler
        rendered "(приоритет: urgent)" while the DB silently stored
        medium. Now the sanitizer covers all three uniformly."""
        args = _sanitize_args("set_daily_goal", {
            "title": "x", "priority": "urgent",
        })
        assert args["priority"] == "medium"

    def test_set_daily_goal_valid_priorities(self):
        for prio in VALID_PRIORITIES:
            args = _sanitize_args("set_daily_goal", {
                "title": "x", "priority": prio,
            })
            assert args["priority"] == prio

    def test_get_weather_days_clamp_low(self):
        """0 or negative days reaches Open-Meteo as invalid forecast_days
        and triggers an API error. Floor at 1."""
        args = _sanitize_args("get_weather", {"days": 0})
        assert args["days"] == 1
        args = _sanitize_args("get_weather", {"days": -5})
        assert args["days"] == 1

    def test_get_weather_days_clamp_high(self):
        args = _sanitize_args("get_weather", {"days": 99})
        assert args["days"] == 7

    def test_get_weather_days_omitted_unchanged(self):
        """Days not passed → handler default applies via signature."""
        args = _sanitize_args("get_weather", {"date": "2026-04-15"})
        assert "days" not in args

    def test_search_memory_no_category_unchanged(self):
        """When the LLM doesn't pass a category at all, the slot stays
        absent (not coerced to None or any default) so the handler
        signature default applies."""
        args = _sanitize_args("search_memory", {"query": "x"})
        assert "category" not in args

    def test_get_recent_memories_drops_invalid_category(self):
        """Same guard as search_memory: invalid category becomes None
        so the user gets recent memories across all categories instead
        of an empty 'No recent memories' result for a typo."""
        args = _sanitize_args("get_recent_memories", {
            "limit": 5, "category": "tasks",  # not a real category
        })
        assert args["category"] is None

    def test_get_recent_memories_passes_valid_category(self):
        for cat in VALID_CATEGORIES:
            args = _sanitize_args("get_recent_memories", {
                "limit": 3, "category": cat,
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
    async def test_event_alert_and_reflection_have_distinct_labels(self, tmp_db):
        """Pre-fix _SEARCH_KIND_LABELS lacked entries for event_alert and
        event_reflection so search_conversations output rendered them as
        the generic '[уведомление]' marker. Now each kind gets a distinct
        Russian label so Claude (and the user, when output bubbles up)
        can tell pre-event reminder from post-event reflection."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.log_interaction(
            "out", "notification", "🔔 Через ~10 мин: ужин LANDMARK",
            metadata={"kind": "event_alert", "event_id": "x"},
        )
        tmp_db.log_interaction(
            "out", "notification", "🪞 Как прошло «ужин LANDMARK»?",
            metadata={"kind": "event_reflection", "event_id": "x"},
        )
        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute(
            "search_conversations", {"query": "LANDMARK", "days": 30, "limit": 10},
        )
        assert "[событие скоро]" in result
        assert "[как прошло]" in result
        assert "[уведомление]" not in result

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
        # 3 → form_2_4 "записи" (not legacy hardcoded "записей")
        assert "3 записи" in result
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

        # 3 → form_2_4 ("записи"), not the legacy hardcoded "записей"
        assert "3 записи" in result
        assert "3 записей" not in result
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


class TestGetEventsHandler:
    """ToolExecutor handler for get_events. Single-day vs multi-day output
    must differ: multi-day prefixes each line with the date so the user
    can tell which day it falls on. Location and related_person surface
    only when populated. Long ranges get capped to keep the LLM context
    bounded."""

    @pytest.mark.asyncio
    async def test_single_day_omits_date_prefix(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_event("Dentist", "2026-04-15 10:00:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {"date_from": "2026-04-15"})

        # No "2026-04-15" prefix — date is implicit when single-day
        assert "2026-04-15" not in result
        assert "10:00 Dentist" in result

    @pytest.mark.asyncio
    async def test_multi_day_prefixes_with_date(self, tmp_db):
        """User asking 'что на неделе' must see which day each event is on
        — without the date prefix all events look like 'today'."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_event("Day1", "2026-04-15 09:00:00")
        tmp_db.create_event("Day2", "2026-04-16 14:00:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {
            "date_from": "2026-04-15", "date_to": "2026-04-16",
        })

        assert "2026-04-15 09:00 Day1" in result
        assert "2026-04-16 14:00 Day2" in result

    @pytest.mark.asyncio
    async def test_renders_location_and_person(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_event(
            "обед", "2026-04-15 13:00:00",
            location="кафе Пушкин", related_person="Олег",
        )

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {"date_from": "2026-04-15"})

        assert "📍 кафе Пушкин" in result
        assert "👤 Олег" in result

    @pytest.mark.asyncio
    async def test_omits_redundant_person_in_title(self, tmp_db):
        """If related_person is already in the title (common: 'встреча с
        Машей' + person='Маша'), don't duplicate it as a 👤 line."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_event(
            "встреча с Машей", "2026-04-15 14:00:00", related_person="Маша",
        )

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {"date_from": "2026-04-15"})

        assert "👤" not in result

    @pytest.mark.asyncio
    async def test_renders_end_at_when_same_day(self, tmp_db):
        """Same-day end_at renders as 'HH:MM-HH:MM'. Cross-day end_at
        (overnight events) keeps just the start time so the dash doesn't
        hide a day boundary."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_event(
            "стандап", "2026-04-15 09:00:00", end_at="2026-04-15 09:30:00",
        )
        tmp_db.create_event(
            "хакатон", "2026-04-16 10:00:00", end_at="2026-04-17 10:00:00",
        )

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        same_day = await te.execute("get_events", {"date_from": "2026-04-15"})
        assert "09:00-09:30 стандап" in same_day

        cross_day = await te.execute("get_events", {"date_from": "2026-04-16"})
        # No range — cross-day end is dropped because the dash would mislead
        assert "10:00-10:00" not in cross_day
        assert "10:00 хакатон" in cross_day

    @pytest.mark.asyncio
    async def test_caps_output_for_busy_range(self, tmp_db):
        """A month with 50+ events would dump too much into the LLM
        context. Cap at 30 with a hint that the range can be narrowed."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        for i in range(40):
            day = 1 + (i % 28)  # spread across April
            tmp_db.create_event(f"event {i}", f"2026-04-{day:02d} 10:{i%60:02d}:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {
            "date_from": "2026-04-01", "date_to": "2026-04-30",
        })
        # 30 events + 1 truncation hint line
        assert result.count("\n") == 30
        assert "ещё" in result and "сузь" in result


class TestGetDiaryEntriesHandler:
    """ToolExecutor handler for get_diary_entries. Diary is auto-generated
    every evening with mood + summary + people, but the LLM had no way to
    read it. Closes that gap so 'что я писал на прошлой неделе?' becomes
    answerable."""

    @pytest.mark.asyncio
    async def test_empty_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_diary_entries", {})
        assert "Записей в дневнике" in result and "7" in result

    @pytest.mark.asyncio
    async def test_renders_date_mood_people_and_content(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        today = tmp_db._now().strftime("%Y-%m-%d")
        tmp_db.save_diary_entry(
            date=today,
            content="долгий день, был в офисе и встречался с командой",
            mood="нейтральное",
            people="Маша, Олег",
        )

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_diary_entries", {})

        assert today in result
        assert "нейтральное" in result
        assert "Маша" in result and "Олег" in result
        assert "долгий день" in result

    @pytest.mark.asyncio
    async def test_truncates_long_content_per_entry(self, tmp_db):
        """A 30-day dump of full content would dominate the token budget;
        cap is 600 chars per entry with a trailing ellipsis so the LLM
        knows the entry is partial."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        today = tmp_db._now().strftime("%Y-%m-%d")
        long_content = "x" * 2000  # well past the 600-char cap
        tmp_db.save_diary_entry(date=today, content=long_content, mood=None, people=None)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_diary_entries", {})

        # Cap honored, ellipsis appended so LLM knows it's partial
        assert "…" in result
        # No 2000-x string in output
        assert "x" * 2000 not in result

    @pytest.mark.asyncio
    async def test_limit_caps_count_with_overflow_hint(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # Insert 4 entries on different dates
        base = tmp_db._now()
        for i in range(4):
            d = (base - timedelta(days=i)).strftime("%Y-%m-%d")
            tmp_db.save_diary_entry(date=d, content=f"day {i}", mood=None, people=None)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_diary_entries", {"limit": 2})

        # 2 entries shown + overflow hint. 2 → form_2_4 ("записи"), not
        # the legacy hardcoded "записей" (which read wrong for 2-4).
        assert "ещё 2 записи" in result
        assert "ещё 2 записей" not in result
        # Newest first (DB ORDER BY date DESC)
        # The first day shown is base (i=0), second is base-1
        first_day = base.strftime("%Y-%m-%d")
        assert first_day in result.split("\n\n")[0]


class TestGetEventsDateValidation:
    """Pre-fix get_events accepted any string as date_from. SQLite's
    date(?) silently returns NULL on unparseable input, so an LLM call
    like get_events(date_from="tomorrow") matched zero rows and the
    handler reported 'No events for tomorrow' even with real events
    on the calendar. Strict YYYY-MM-DD check surfaces the format hint
    so the LLM retries with a proper date."""

    @pytest.mark.asyncio
    async def test_invalid_date_from_returns_format_hint(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # Real event on the calendar — confirms we're rejecting at the
        # validation step, not just empty-coincidentally.
        tmp_db.create_event("ужин", "2026-04-15 19:00:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {"date_from": "tomorrow"})
        assert "invalid date_from" in result
        assert "YYYY-MM-DD" in result
        # The real event must NOT leak into a "No events" false-negative
        assert "ужин" not in result

    @pytest.mark.asyncio
    async def test_invalid_date_to_returns_format_hint(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {
            "date_from": "2026-04-15", "date_to": "не понятно",
        })
        assert "invalid date_to" in result

    @pytest.mark.asyncio
    async def test_valid_date_passes_through(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_event("работа", "2026-04-15 09:00:00")
        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_events", {"date_from": "2026-04-15"})
        assert "работа" in result
        assert "invalid" not in result

    @pytest.mark.asyncio
    async def test_get_daily_goals_invalid_date_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_daily_goals", {"date": "вчера"})
        assert "invalid date" in result


class TestGetDailyGoalsHandler:
    """ToolExecutor handler for get_daily_goals. Closes the last write-only
    pair (set_daily_goal + complete_daily_goal had no read tool). User
    asking 'что я хотел сегодня?' previously needed get_open_loops which
    is broader — direct tool surfaces today's goals with statuses."""

    @pytest.mark.asyncio
    async def test_empty_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_daily_goals", {})
        assert "Целей на сегодня нет" in result

    @pytest.mark.asyncio
    async def test_lists_goals_with_status_and_priority(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_daily_goal("починить раковину", priority="high")
        tmp_db.create_daily_goal("позвонить маме", priority="low")
        # Mark one as completed so the response shows mixed statuses
        tmp_db.complete_daily_goal_by_hint("раковину", status="completed")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_daily_goals", {})

        assert "починить раковину" in result
        assert "позвонить маме" in result
        # Russian status labels rendered
        assert "выполнена" in result
        assert "не отмечена" in result
        # Priority labels rendered
        assert "высокий" in result
        assert "низкий" in result

    @pytest.mark.asyncio
    async def test_specific_date_argument(self, tmp_db):
        """User asks 'что я не успел вчера?' → LLM passes date for past
        day. Tool must respect the date arg, not silently default to today."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # Insert a goal for a specific past date by direct DB write —
        # create_daily_goal stamps date=today, but we want a fixed date
        # for this test.
        tmp_db.db.execute(
            "INSERT INTO daily_goals (date, title, priority, status) "
            "VALUES (?, ?, ?, ?)",
            ("2026-04-10", "ужин с Машей", "medium", "skipped"),
        )
        tmp_db.db.commit()

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        # Today should be empty
        empty = await te.execute("get_daily_goals", {})
        assert "ужин" not in empty

        # The 2026-04-10 query surfaces the past goal with skipped status
        past = await te.execute("get_daily_goals", {"date": "2026-04-10"})
        assert "ужин с Машей" in past
        assert "пропущена" in past

    @pytest.mark.asyncio
    async def test_renders_reflection(self, tmp_db):
        """If complete_daily_goal stored a reflection, it must surface in
        get_daily_goals so the user sees their own end-of-day notes."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_daily_goal("книга", priority="medium")
        tmp_db.complete_daily_goal_by_hint(
            "книга", status="partial",
            reflection="прочитал только главу 3",
        )

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("get_daily_goals", {})
        assert "прочитал только главу 3" in result


class TestSearchEventsHandler:
    """ToolExecutor handler for search_events. Lets Claude answer 'когда
    встреча с Машей?' without first guessing a date range for get_events."""

    @pytest.mark.asyncio
    async def test_empty_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("search_events", {"query": "nothing"})
        assert "Не нашёл" in result and "nothing" in result

    @pytest.mark.asyncio
    async def test_finds_match_with_full_datetime(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("ужин с Машей", future, location="кафе Пушкин")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("search_events", {"query": "Маш"})

        assert "ужин с Машей" in result
        # Full date+time prefix (vs HH:MM-only in get_events) — search
        # results may be days out, hiding the day would be confusing.
        assert future[:16] in result
        # Location surfaces
        assert "кафе Пушкин" in result

    @pytest.mark.asyncio
    async def test_orders_results_by_soonest_first(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        for offset in (10, 1, 5):
            ts = (now + timedelta(days=offset)).strftime("%Y-%m-%d %H:%M:%S")
            tmp_db.create_event(f"встреча {offset}", ts)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("search_events", {"query": "встреча"})
        lines = result.split("\n")
        # Soonest match (offset=1) on first line, latest (offset=10) last
        assert "встреча 1" in lines[0]
        assert "встреча 10" in lines[-1]

    @pytest.mark.asyncio
    async def test_limit_clamped(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        for i in range(8):
            ts = (now + timedelta(days=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
            tmp_db.create_event(f"yoga {i}", ts)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("search_events", {"query": "yoga", "limit": 3})
        assert result.count("\n") == 2  # 3 lines = 2 newlines


class TestCompleteDailyGoalStatusValidation:
    """Invalid status from the LLM (typo, drift) gets coerced to
    'completed' both in the DB and in the rendered Russian label —
    pre-fix the rendered label echoed the LLM's raw value while the
    DB silently stored 'completed', producing a confusing mismatch."""

    @pytest.mark.asyncio
    async def test_invalid_status_coerced_to_completed_in_render(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_daily_goal("задача")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("complete_daily_goal", {
            "goal_hint": "задача", "status": "done",  # not a valid enum value
        })

        # Rendered as выполнена (from coerced status="completed")
        assert "marked as выполнена" in result
        # LLM's raw "done" must NOT appear in the rendered label
        assert "marked as done" not in result

        # DB row reflects coerced status
        row = tmp_db.db.execute(
            "SELECT status FROM daily_goals WHERE title = 'задача'"
        ).fetchone()
        assert row["status"] == "completed"

    @pytest.mark.asyncio
    async def test_valid_status_preserved(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_daily_goal("задача")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("complete_daily_goal", {
            "goal_hint": "задача", "status": "skipped",
        })
        assert "marked as пропущена" in result


class TestCompleteDailyGoalAmbiguity:
    """complete_daily_goal_by_hint silently marks the oldest match
    when hint is ambiguous. Same UX gap that resolve_decision and
    cancel_event/cancel_reminder used to have."""

    @pytest.mark.asyncio
    async def test_unique_match_no_ambiguity_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_daily_goal("учить английский")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("complete_daily_goal", {
            "goal_hint": "английский", "status": "completed",
        })
        assert "marked as выполнена" in result
        # Single match — no "Похожих ещё" trailer
        assert "Похожих ещё" not in result

    @pytest.mark.asyncio
    async def test_multiple_matches_disclosed(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_daily_goal("учить английский")
        tmp_db.create_daily_goal("пойти на английский")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("complete_daily_goal", {
            "goal_hint": "английский", "status": "completed",
        })
        assert "marked as выполнена" in result
        assert "Похожих ещё 1" in result

    @pytest.mark.asyncio
    async def test_no_match_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("complete_daily_goal", {
            "goal_hint": "ничего", "status": "completed",
        })
        assert "No pending goal found" in result


class TestTrackDecisionResilience:
    """track_decision must save the user's intent even if the secondary
    past-decisions lookup fails. Pre-fix the order was reversed and a
    DB hiccup on the search would propagate up as 'Error executing
    track_decision' while the row was never written."""

    @pytest.mark.asyncio
    async def test_create_runs_even_when_past_search_throws(self, tmp_db, monkeypatch):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # Sabotage get_past_decisions
        original_search = tmp_db.get_past_decisions
        def boom(*args, **kwargs):
            raise RuntimeError("simulated db hiccup")
        monkeypatch.setattr(tmp_db, "get_past_decisions", boom)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("track_decision", {
            "description": "купить велосипед",
        })

        # Result reflects success, not error
        assert "Decision tracked" in result
        assert "купить велосипед" in result
        # And critically the row IS in the DB
        rows = tmp_db.db.execute(
            "SELECT description FROM decisions WHERE status = 'pending'"
        ).fetchall()
        assert any(r["description"] == "купить велосипед" for r in rows)

    @pytest.mark.asyncio
    async def test_past_decisions_appended_when_lookup_succeeds(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # Resolve a similar past decision so it surfaces in the lookup
        tmp_db.create_decision("купить старый велосипед")
        tmp_db.resolve_decision_by_hint("старый", "купил")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("track_decision", {
            "description": "купить новый велосипед",
        })

        assert "Decision tracked" in result
        assert "Similar past decisions" in result
        assert "купить старый велосипед" in result


class TestResolveDecisionAmbiguity:
    """resolve_decision picks the most-recent match silently when hint
    is ambiguous. Mirror cancel_reminder/cancel_event ambiguity disclosure
    so the LLM knows other matches were skipped."""

    @pytest.mark.asyncio
    async def test_unique_match_no_ambiguity_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_decision("купить велосипед")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("resolve_decision", {
            "description_hint": "велосипед", "outcome": "купил",
        })
        assert "Resolved decision" in result
        # Single match — no "Похожих ещё" trailer
        assert "Похожих ещё" not in result

    @pytest.mark.asyncio
    async def test_multiple_matches_disclosed(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        # Three pending decisions all matching "купить"
        tmp_db.create_decision("купить велосипед")
        tmp_db.create_decision("купить машину")
        tmp_db.create_decision("купить новый ноутбук")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("resolve_decision", {
            "description_hint": "купить", "outcome": "решил",
        })
        # One resolved + disclosure that 2 more matched
        assert "Resolved decision" in result
        assert "Похожих ещё 2" in result

    @pytest.mark.asyncio
    async def test_no_match_returns_friendly_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("resolve_decision", {
            "description_hint": "ничего", "outcome": "x",
        })
        assert "No pending decision found" in result


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


class TestGetWeatherHandler:
    """ToolExecutor handler for get_weather. Pre-fix the date param was
    declared in the schema but completely ignored — LLM passing
    date='2026-05-09' got today's weather. Now date drives the request
    range and filters the output to that day."""

    @staticmethod
    def _make_weather_mock(tz: str = "UTC", daily: list[dict] | None = None):
        """Build a WeatherClient stub. Captures get_forecast calls so the
        test can assert on `days` requested."""
        from unittest.mock import AsyncMock, MagicMock
        weather = MagicMock()
        weather.tz = tz
        weather.get_forecast = AsyncMock(return_value={
            "daily": daily or [],
        })
        # format_daily passes through — easier to assert on output content
        weather.format_daily = MagicMock(
            side_effect=lambda f: "\n".join(
                f"{d['date']}: {d.get('cond', 'cond')}"
                for d in f.get("daily", [])
            ) or "Нет данных.",
        )
        return weather

    @pytest.mark.asyncio
    async def test_no_args_returns_today(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        weather = self._make_weather_mock(daily=[
            {"date": "2026-04-15", "cond": "ясно"},
        ])
        te = ToolExecutor(db=tmp_db, memory=MagicMock(), weather=weather)
        result = await te.execute("get_weather", {})
        weather.get_forecast.assert_awaited_once_with(days=1)
        assert "2026-04-15" in result

    @pytest.mark.asyncio
    async def test_date_arg_filters_to_specific_day(self, tmp_db, monkeypatch):
        """Most user value — pre-fix a Saturday query returned Wednesday's
        weather. Now the handler computes delta from today, requests N+1
        days, and filters output to the requested date."""
        from unittest.mock import MagicMock, patch
        from datetime import datetime
        from mindsecretary.llm.tools import ToolExecutor

        # Pin "today" in the weather TZ via tz_now patch. After v0.14.50
        # tz_now is imported at the tools module top, so patch the local
        # binding `mindsecretary.llm.tools.tz_now` rather than the source
        # `mindsecretary.core.tz_now` — `from X import Y` binds at import,
        # patching X.Y later won't affect the already-bound local Y.
        with patch(
            "mindsecretary.llm.tools.datetime",
        ) as mock_dt, patch(
            "mindsecretary.llm.tools.tz_now",
        ) as mock_tz_now:
            mock_dt.strptime = datetime.strptime
            mock_tz_now.return_value = datetime(2026, 4, 15)
            weather = self._make_weather_mock(daily=[
                {"date": "2026-04-15", "cond": "ясно"},
                {"date": "2026-04-16", "cond": "облачно"},
                {"date": "2026-04-17", "cond": "дождь"},
            ])
            te = ToolExecutor(db=tmp_db, memory=MagicMock(), weather=weather)
            result = await te.execute("get_weather", {"date": "2026-04-17"})

        # Requested 3 days (today + 2)
        weather.get_forecast.assert_awaited_once_with(days=3)
        # Output filtered to just the requested date
        assert "2026-04-17" in result
        assert "дождь" in result
        # Other days NOT in output
        assert "2026-04-15" not in result
        assert "2026-04-16" not in result

    @pytest.mark.asyncio
    async def test_past_date_returns_friendly_error(self, tmp_db):
        from unittest.mock import MagicMock, patch
        from datetime import datetime
        from mindsecretary.llm.tools import ToolExecutor

        with patch("mindsecretary.llm.tools.tz_now") as mock_tz_now:
            mock_tz_now.return_value = datetime(2026, 4, 15)
            weather = self._make_weather_mock()
            te = ToolExecutor(db=tmp_db, memory=MagicMock(), weather=weather)
            result = await te.execute("get_weather", {"date": "2026-04-01"})

        assert "в прошлом" in result
        # Critical: no API call wasted on a past date
        weather.get_forecast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_far_future_date_returns_friendly_error(self, tmp_db):
        """Open-Meteo caps at 7-day horizon; a date 30 days out can't be
        forecast. Don't waste an API call — tell the LLM why."""
        from unittest.mock import MagicMock, patch
        from datetime import datetime
        from mindsecretary.llm.tools import ToolExecutor

        with patch("mindsecretary.llm.tools.tz_now") as mock_tz_now:
            mock_tz_now.return_value = datetime(2026, 4, 15)
            weather = self._make_weather_mock()
            te = ToolExecutor(db=tmp_db, memory=MagicMock(), weather=weather)
            result = await te.execute("get_weather", {"date": "2026-05-30"})

        assert "слишком далеко" in result
        weather.get_forecast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_date_falls_back_to_today(self, tmp_db):
        """Garbage date string → handler degrades gracefully to today's
        forecast instead of crashing or refusing."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        weather = self._make_weather_mock(daily=[
            {"date": "2026-04-15", "cond": "ясно"},
        ])
        te = ToolExecutor(db=tmp_db, memory=MagicMock(), weather=weather)
        result = await te.execute("get_weather", {"date": "not-a-date"})

        weather.get_forecast.assert_awaited_once_with(days=1)
        assert "2026-04-15" in result


class TestUpdateEventHandler:
    """ToolExecutor handler for update_event. Closes the event-CRUD loop
    — non-time fields (title, description, location, related_person)
    have no other path. Symmetric with reschedule_event for time fields."""

    @pytest.mark.asyncio
    async def test_updates_title(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("обед", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_event", {
            "text_hint": "обед", "title": "ужин с Сашей",
        })

        assert "Обновлено" in result
        assert "title" in result
        assert "ужин с Сашей" in result
        # DB reflects the change
        row = tmp_db.db.execute("SELECT title FROM events").fetchone()
        assert row["title"] == "ужин с Сашей"

    @pytest.mark.asyncio
    async def test_updates_multiple_fields(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("встреча", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_event", {
            "text_hint": "встреча",
            "location": "кафе Пушкин",
            "related_person": "Олег",
        })

        assert "location" in result and "related_person" in result
        row = tmp_db.db.execute(
            "SELECT location, related_person FROM events"
        ).fetchone()
        assert row["location"] == "кафе Пушкин"
        assert row["related_person"] == "Олег"

    @pytest.mark.asyncio
    async def test_no_fields_returns_error(self, tmp_db):
        """Calling with only text_hint is a no-op masquerading as a fix
        — reject explicitly so the LLM doesn't think it succeeded."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("встреча", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_event", {"text_hint": "встреча"})
        assert "at least one of" in result

    @pytest.mark.asyncio
    async def test_empty_title_distinguishes_from_no_match(self, tmp_db):
        """Pre-fix: title="   " was forwarded to DB which silently rejects
        it (NOT NULL constraint), and the handler then said 'Не нашёл
        событий' — misleading because the event WAS matched, just the
        title rejected. Now the handler surfaces the distinct cause."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("обед", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_event", {
            "text_hint": "обед", "title": "   ",
        })
        # Distinct error path — must NOT say "Не нашёл" since the event
        # WAS found.
        assert "title cannot be empty" in result
        assert "Не нашёл" not in result

        # And the original event's title is unchanged
        row = tmp_db.db.execute("SELECT title FROM events").fetchone()
        assert row["title"] == "обед"

    @pytest.mark.asyncio
    async def test_no_match_returns_message(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_event", {
            "text_hint": "nothing", "location": "x",
        })
        assert "Не нашёл" in result

    @pytest.mark.asyncio
    async def test_empty_hint_rejected(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_event", {
            "text_hint": "  ", "title": "x",
        })
        assert "non-empty" in result

    @pytest.mark.asyncio
    async def test_ambiguity_disclosure(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        for i in range(3):
            ts = (now + timedelta(days=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
            tmp_db.create_event(f"встреча {i}", ts)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_event", {
            "text_hint": "встреча", "location": "новое место",
        })
        assert "Похожих ещё 2" in result


class TestEmptyInputValidation:
    """Defensive empty-input rejection across the remaining create-style
    handlers. Empty fields would persist broken rows that are hard to
    recover from later via the by-hint mutators."""

    @pytest.mark.asyncio
    async def test_update_contact_rejects_empty_name(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("update_contact", {"name": "  "})
        assert "non-empty name" in result
        assert tmp_db.get_contacts("") == []

    @pytest.mark.asyncio
    async def test_log_habit_rejects_empty_name(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("log_habit", {
            "habit_name": "   ", "done": True,
        })
        assert "non-empty habit_name" in result
        # No row created
        assert tmp_db.get_habit_stats() == []

    @pytest.mark.asyncio
    async def test_log_habit_rejects_invalid_date(self, tmp_db):
        """LLM passing a Russian word like "вчера" as date must be
        caught at the boundary — pre-fix the row landed in habit_log
        with date='вчера', and get_habit_stats filtered it out lexically
        ('вчера' > '2026-05-07' so `date <= today` is FALSE) so the
        user saw "habit not logged" while the row sat polluting the DB."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("log_habit", {
            "habit_name": "зарядка", "done": True, "date": "вчера",
        })
        assert "invalid date" in result
        assert "YYYY-MM-DD" in result
        # Habit row in habits table also not created — boundary stops
        # short of any DB write.
        assert tmp_db.get_habit_stats() == []

    @pytest.mark.asyncio
    async def test_set_daily_goal_rejects_empty_title(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("set_daily_goal", {"title": "  "})
        assert "non-empty title" in result
        assert tmp_db.get_daily_goals() == []

    @pytest.mark.asyncio
    async def test_track_decision_rejects_empty_description(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("track_decision", {"description": "   "})
        assert "non-empty description" in result
        assert tmp_db.get_pending_decisions() == []

    @pytest.mark.asyncio
    async def test_save_memory_rejects_empty_content(self, tmp_db):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        memory = MagicMock()
        memory.save = AsyncMock()
        te = ToolExecutor(db=tmp_db, memory=memory)
        result = await te.execute("save_memory", {
            "content": "  ", "category": "personal", "importance": 5,
        })
        assert "non-empty content" in result
        # Voyage embed never called — no wasted API round trip
        memory.save.assert_not_awaited()


class TestCreateValidation:
    """Defensive empty-validation on create_event / create_reminder so
    a stray empty arg from the LLM doesn't store a row that's invisible
    to subsequent reads."""

    @pytest.mark.asyncio
    async def test_create_event_rejects_empty_title(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "   ", "start_at": "2099-01-15 10:00:00",
        })
        assert "non-empty title" in result
        # No row created
        rows = tmp_db.db.execute("SELECT * FROM events").fetchall()
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_create_event_rejects_empty_start_at(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "встреча", "start_at": "  ",
        })
        assert "non-empty start_at" in result

    @pytest.mark.asyncio
    async def test_create_event_rejects_unparseable_start_at(self, tmp_db):
        """LLM may emit a relative timestamp ("tomorrow 14:00") instead
        of ISO. Sanitizer leaves it as-is on parse failure, so without
        this guard the row stores garbage and date(start_at) queries
        miss it forever."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "встреча", "start_at": "tomorrow 14:00",
        })
        assert "invalid start_at" in result
        # Format hint included so the LLM knows what to retry with
        assert "YYYY-MM-DDTHH:MM" in result
        # No row created
        rows = tmp_db.db.execute("SELECT * FROM events").fetchall()
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_create_event_rejects_unparseable_end_at(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "встреча",
            "start_at": "2099-04-15T14:00",
            "end_at": "tomorrow 16:00",  # garbage
        })
        assert "invalid end_at" in result
        rows = tmp_db.db.execute("SELECT * FROM events").fetchall()
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_create_event_accepts_iso_with_seconds(self, tmp_db):
        """Sanity: full ISO with seconds is accepted by fromisoformat."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "встреча",
            "start_at": "2099-04-15T14:00:00",
        })
        assert "Event created" in result

    @pytest.mark.asyncio
    async def test_create_event_rejects_end_before_start(self, tmp_db):
        """Transposed end_at < start_at would render in /events as a
        nonsense range '14:00-13:00'. Reject up front."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "встреча",
            "start_at": "2099-04-15T14:00",
            "end_at": "2099-04-15T13:00",
        })
        assert "must be after" in result
        # No row created
        rows = tmp_db.db.execute("SELECT * FROM events").fetchall()
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_create_event_rejects_end_equal_to_start(self, tmp_db):
        """Zero-duration events also rejected — same render artifact
        ('14:00-14:00') and probably an LLM mistake (forgot to advance
        the end time)."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "встреча",
            "start_at": "2099-04-15T14:00",
            "end_at": "2099-04-15T14:00",
        })
        assert "must be after" in result

    @pytest.mark.asyncio
    async def test_create_event_accepts_valid_end_at(self, tmp_db):
        """Sanity / regression-guard for the end-after-start check —
        a normal valid pair (end > start) must NOT be rejected. Without
        this test a tightening of the comparison (e.g. end >= start
        accidentally becoming end > start + 1h) could regress silently."""
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_event", {
            "title": "встреча",
            "start_at": "2099-04-15T14:00",
            "end_at": "2099-04-15T15:00",
        })
        assert "Event created" in result
        # Row stored with both times intact
        row = tmp_db.db.execute(
            "SELECT start_at, end_at FROM events WHERE title = 'встреча'"
        ).fetchone()
        assert row["start_at"][:16] == "2099-04-15 14:00"
        assert row["end_at"][:16] == "2099-04-15 15:00"

    @pytest.mark.asyncio
    async def test_reschedule_event_accepts_valid_end_at(self, tmp_db):
        """Same regression guard for reschedule_event."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("встреча", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_event", {
            "text_hint": "встреча",
            "new_start_at": "2099-04-15T14:00",
            "new_end_at": "2099-04-15T16:30",
        })
        assert "Перенесено" in result
        row = tmp_db.db.execute(
            "SELECT start_at, end_at FROM events WHERE title = 'встреча'"
        ).fetchone()
        assert row["start_at"][:16] == "2099-04-15 14:00"
        assert row["end_at"][:16] == "2099-04-15 16:30"

    @pytest.mark.asyncio
    async def test_create_reminder_rejects_empty_text(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_reminder", {
            "text": "  ", "trigger_at": "2099-01-15 10:00:00",
        })
        assert "non-empty text" in result
        assert tmp_db.get_pending_reminders() == []

    @pytest.mark.asyncio
    async def test_create_reminder_rejects_empty_trigger_at(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_reminder", {
            "text": "позвонить", "trigger_at": "",
        })
        assert "non-empty trigger_at" in result

    @pytest.mark.asyncio
    async def test_create_reminder_rejects_unparseable_trigger_at(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("create_reminder", {
            "text": "позвонить", "trigger_at": "tomorrow 10am",
        })
        assert "invalid trigger_at" in result
        assert "YYYY-MM-DDTHH:MM" in result
        # No row created
        assert tmp_db.get_pending_reminders() == []

    @pytest.mark.asyncio
    async def test_reschedule_event_rejects_unparseable_new_start(self, tmp_db):
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("встреча", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_event", {
            "text_hint": "встреча", "new_start_at": "next monday",
        })
        assert "invalid new_start_at" in result
        # Original event time unchanged
        row = tmp_db.db.execute("SELECT start_at FROM events").fetchone()
        assert row["start_at"] == future

    @pytest.mark.asyncio
    async def test_reschedule_event_rejects_end_before_start(self, tmp_db):
        """Same end-after-start guard as create_event — reschedule with
        a transposed pair would silently store a broken range."""
        from datetime import timedelta
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        now = tmp_db.local_now_naive()
        future = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        tmp_db.create_event("встреча", future)

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_event", {
            "text_hint": "встреча",
            "new_start_at": "2099-04-15T14:00",
            "new_end_at": "2099-04-15T13:00",
        })
        assert "must be after" in result
        # Original time unchanged
        row = tmp_db.db.execute("SELECT start_at FROM events").fetchone()
        assert row["start_at"] == future

    @pytest.mark.asyncio
    async def test_reschedule_reminder_rejects_unparseable_trigger(self, tmp_db):
        from unittest.mock import MagicMock
        from mindsecretary.llm.tools import ToolExecutor

        tmp_db.create_reminder("позвонить", "2099-01-15 10:00:00")

        te = ToolExecutor(db=tmp_db, memory=MagicMock())
        result = await te.execute("reschedule_reminder", {
            "text_hint": "позвонить", "new_trigger_at": "ASAP",
        })
        assert "invalid new_trigger_at" in result
        # Reminder unchanged
        pending = tmp_db.get_pending_reminders()
        assert pending[0]["trigger_at"] == "2099-01-15 10:00:00"


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
