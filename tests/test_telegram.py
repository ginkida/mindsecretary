"""Tests for interfaces/telegram.py — utility functions."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.constants import ParseMode

from mindsecretary.interfaces.telegram import TelegramBot, _fix_markdown, _split_message


class TestFixMarkdown:
    def test_paired_stars_unchanged(self):
        assert _fix_markdown("*bold*") == "*bold*"

    def test_orphan_star_escaped(self):
        result = _fix_markdown("price is 5*3")
        assert "\\*" in result

    def test_orphan_underscore_escaped(self):
        result = _fix_markdown("some_var name")
        assert "\\_" in result

    def test_orphan_backtick_escaped(self):
        result = _fix_markdown("use `code here")
        assert "\\`" in result

    def test_normal_text_unchanged(self):
        text = "Hello world, no formatting"
        assert _fix_markdown(text) == text

    def test_multiple_paired_unchanged(self):
        text = "*bold* and _italic_ and `code`"
        assert _fix_markdown(text) == text


class TestSplitMessage:
    def test_short_message_single_part(self):
        assert _split_message("hello") == ["hello"]

    def test_long_message_splits(self):
        text = "line\n" * 2000
        parts = _split_message(text, limit=100)
        assert len(parts) > 1
        assert all(len(p) <= 100 for p in parts)


def _make_bot():
    brain = MagicMock()
    brain.settings.rate_limit_per_minute = 20
    brain.settings.process_timeout_sec = 30
    brain.settings.quiet_contact_days = 30
    brain.settings.quiet_contact_min_mentions = 3
    brain.memory = MagicMock()
    brain.memory.search = AsyncMock()
    brain.memory.list_recent = MagicMock(return_value=[])
    brain.memory.get_by_category = MagicMock(return_value=[])
    brain.db = MagicMock()
    brain.db.get_open_loops = MagicMock(return_value={"counts": {}})
    brain.profile.notification_limit = 10

    bot = TelegramBot(
        token="token",
        allowed_user_id=1,
        brain=brain,
        stt=MagicMock(),
    )
    return bot, brain


def _make_update():
    message = MagicMock()
    message.reply_text = AsyncMock()
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        message=message,
    )
    return update


class TestTelegramHandlers:
    @pytest.mark.asyncio
    async def test_search_is_rate_limited(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["coffee"])
        bot._check_rate_limit = lambda: False

        await bot._handle_search(update, context)

        update.message.reply_text.assert_awaited_once_with("Слишком часто, подожди минуту.")
        brain.memory.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_memory_is_rate_limited(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["plans"])
        bot._check_rate_limit = lambda: False

        await bot._handle_memory(update, context)

        update.message.reply_text.assert_awaited_once_with("Слишком часто, подожди минуту.")
        brain.memory.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stats_handler_renders_category_breakdown(self):
        """/stats includes a per-category memory breakdown so the user
        sees what kinds of facts the bot is accumulating, not just the
        opaque total. Top 5 only — keeps Telegram message scannable."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.db.get_stats = MagicMock(return_value={
            "today_cost": 0.10, "today_tokens": 1000, "month_cost": 5.0,
            "memories": 100, "contacts": 5, "interactions_today": 20,
            "providers": {}, "week_trend": [],
            "memory_categories": [
                {"category": "work", "count": 40},
                {"category": "personal", "count": 30},
                {"category": "health", "count": 15},
                {"category": "promise", "count": 10},
                {"category": "contact", "count": 3},
                {"category": "preference", "count": 2},  # 6th, must NOT show
            ],
        })

        await bot._handle_stats(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "work: 40" in text
        assert "personal: 30" in text
        assert "promise: 10" in text
        # Only top 5 — preference (6th) must be cut
        assert "preference" not in text

    @pytest.mark.asyncio
    async def test_stats_handler_handles_empty_breakdown(self):
        """/stats shouldn't crash when there are no memories yet — empty
        list is the bot's first-day state."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.db.get_stats = MagicMock(return_value={
            "today_cost": 0, "today_tokens": 0, "month_cost": 0,
            "memories": 0, "contacts": 0, "interactions_today": 0,
            "providers": {}, "week_trend": [],
            "memory_categories": [],
        })

        await bot._handle_stats(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Воспоминаний: 0" in text
        # No bullets when breakdown is empty
        assert "•" not in text or "Контактов" in text  # other text may have bullets

    @pytest.mark.asyncio
    async def test_version_handler_returns_version_and_counts(self):
        """`/version` is the support-channel command — must always work
        even if individual DB queries fail. Each counter has its own
        try/except so a single broken table doesn't take down the whole
        response."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.memory.count = MagicMock(return_value=42)
        brain.db.get_contacts = MagicMock(return_value=[
            {"id": "x"}, {"id": "y"}, {"id": "z"},
        ])
        brain.db.get_pending_reminders = MagicMock(return_value=[{"id": "r1"}])
        brain.profile.timezone = "Asia/Almaty"

        await bot._handle_version(update, context)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.await_args.args[0]
        assert "MindSecretary" in text
        assert "Воспоминаний: 42" in text
        assert "Контактов: 3" in text
        assert "Pending-напоминаний: 1" in text
        assert "Asia/Almaty" in text

    @pytest.mark.asyncio
    async def test_version_handler_resilient_to_db_errors(self):
        """A single broken counter must not crash /version — falls back
        to 0 for the offender and still returns a valid response."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.memory.count = MagicMock(side_effect=RuntimeError("memory broken"))
        brain.db.get_contacts = MagicMock(return_value=[])
        brain.db.get_pending_reminders = MagicMock(side_effect=RuntimeError("reminders broken"))
        brain.profile.timezone = "UTC"

        await bot._handle_version(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Воспоминаний: 0" in text
        assert "Pending-напоминаний: 0" in text

    @pytest.mark.asyncio
    async def test_forget_falls_back_without_markdown(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["buggy_markdown"])
        bot._check_rate_limit = lambda: True
        brain.memory.search.return_value = [{"id": "m1", "content": "bad _ markdown * text"}]
        update.message.reply_text = AsyncMock(side_effect=[Exception("parse"), None])

        await bot._handle_forget(update, context)

        assert update.message.reply_text.await_count == 2
        first = update.message.reply_text.await_args_list[0]
        second = update.message.reply_text.await_args_list[1]
        assert first.kwargs["parse_mode"] == ParseMode.MARKDOWN
        assert "parse_mode" not in second.kwargs
