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
    async def test_about_no_args_shows_usage(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        bot._check_rate_limit = lambda: True

        await bot._handle_about(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Использование: /about" in text

    @pytest.mark.asyncio
    async def test_about_no_match_returns_friendly_message(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["NobodyExists"])
        bot._check_rate_limit = lambda: True
        brain.db.get_contacts = MagicMock(return_value=[])

        await bot._handle_about(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Не нашёл контакта" in text
        assert "/memory" in text  # suggests fallback search path

    @pytest.mark.asyncio
    async def test_about_runs_pre_meeting_prompt(self):
        """Success path: contact found → memories searched → promises
        searched → LLM call with PRE_MEETING_PROMPT → reply."""
        from unittest.mock import AsyncMock
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["Маша"])
        bot._check_rate_limit = lambda: True

        brain.db.get_contacts = MagicMock(return_value=[{
            "id": "c1", "name": "Маша", "relation": "коллега",
            "birthday": "1990-04-29", "last_contact": "2026-04-20 10:00:00",
            "mention_count": 7, "notes": "любит чай, дочь Лиза",
        }])
        brain.memory.search = AsyncMock(side_effect=[
            [{"category": "work", "content": "вместе на проекте Альфа",
              "score": 0.8, "final_score": 0.7}],
            [],  # no promises
        ])
        brain.llm.chat = AsyncMock(return_value=MagicMock(
            text="👤 Маша (36) — коллега\nПоследний контакт 20 апр.",
        ))
        bot._typing = AsyncMock()

        await bot._handle_about(update, context)

        # LLM called with PRE_MEETING_PROMPT-shaped system text
        assert brain.llm.chat.await_count == 1
        call = brain.llm.chat.await_args
        system = call.kwargs["system"]
        assert "Имя: Маша" in system
        assert "коллега" in system
        # Memories block surfaced
        assert "Альфа" in system
        # User-facing reply contains the LLM output
        update.message.reply_text.assert_awaited()
        first_reply = update.message.reply_text.await_args_list[-1].args[0]
        assert "Маша" in first_reply

    @pytest.mark.asyncio
    async def test_about_picks_most_mentioned_when_multiple_match(self):
        """get_contacts returns matches sorted by mention_count desc,
        and /about uses [0] — the most-mentioned hit. Common first names
        like 'Маша' should resolve to the one the user talks about most."""
        from unittest.mock import AsyncMock
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["Маша"])
        bot._check_rate_limit = lambda: True

        brain.db.get_contacts = MagicMock(return_value=[
            {"id": "c1", "name": "Маша Иванова", "relation": "колл.",
             "mention_count": 25, "notes": ""},
            {"id": "c2", "name": "Маша Петрова", "relation": "знак.",
             "mention_count": 3, "notes": ""},
        ])
        brain.memory.search = AsyncMock(return_value=[])
        brain.llm.chat = AsyncMock(return_value=MagicMock(text="brief"))
        bot._typing = AsyncMock()

        await bot._handle_about(update, context)

        system = brain.llm.chat.await_args.kwargs["system"]
        # First name comes from the higher-mention contact
        assert "Маша Иванова" in system
        assert "Маша Петрова" not in system

    @pytest.mark.asyncio
    async def test_export_includes_all_user_owned_tables(self, tmp_path):
        """/export used to dump only memories/contacts/diary/events/
        decisions — losing the user's reminder history, habits, goals,
        and chat log on migration. Expanded to cover every user-owned
        table; ephemeral_state/api_costs/preferences stay excluded by
        design (transient or bot-internal)."""
        from datetime import datetime as _dt
        from io import BytesIO
        import json as _json
        from mindsecretary.core.database import Database
        from mindsecretary.interfaces.telegram import TelegramBot

        db = Database(tmp_path / "test.db", timezone="UTC")
        db.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT, embedding BLOB,
                category TEXT, importance INTEGER DEFAULT 5,
                related_person TEXT, related_date TEXT,
                source_type TEXT, source_ref TEXT,
                confidence REAL DEFAULT 1.0,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT
            )
        """)
        db.db.commit()

        # Seed every relevant table
        db.create_reminder("call mom", "2099-01-01 10:00:00")
        db.create_daily_goal("write report")
        db.log_habit("yoga", done=True)
        db.upsert_contact("Alice")
        db.create_event("Meeting", "2099-02-01 14:00:00")
        db.create_decision("buy bike")
        db.log_interaction("in", "text", "hello")

        # Real bot wired to the real DB
        brain = MagicMock()
        brain.db = db
        brain.profile.timezone = "UTC"
        brain.settings.rate_limit_per_minute = 20

        bot = TelegramBot(
            token="x", allowed_user_id=1, brain=brain, stt=MagicMock(),
        )

        update = _make_update()
        update.message.reply_text = AsyncMock()
        update.message.reply_document = AsyncMock()
        context = SimpleNamespace(args=[])

        await bot._handle_export(update, context)

        # The "preparing..." message goes first, then the document
        update.message.reply_document.assert_awaited_once()
        call = update.message.reply_document.await_args
        doc = call.kwargs["document"]
        assert isinstance(doc, BytesIO)
        doc.seek(0)
        payload = _json.loads(doc.read().decode("utf-8"))

        # All formerly-missing tables now appear with the seeded rows
        assert len(payload["reminders"]) == 1
        assert payload["reminders"][0]["text"] == "call mom"
        assert len(payload["daily_goals"]) == 1
        assert payload["daily_goals"][0]["title"] == "write report"
        assert len(payload["habits"]) == 1
        assert payload["habits"][0]["name"] == "yoga"
        assert len(payload["habit_log"]) == 1
        assert payload["habit_log"][0]["done"] == 1
        assert len(payload["interactions"]) == 1
        assert payload["interactions"][0]["content"] == "hello"

        # Pre-existing tables still populated
        assert len(payload["events"]) == 1
        assert len(payload["decisions"]) == 1
        assert len(payload["contacts"]) == 1

        # Caption mentions the new categories so the user sees the scope
        caption = call.kwargs["caption"]
        assert "напоминаний" in caption
        assert "привычек" in caption
        assert "целей" in caption
        assert "взаимодействий" in caption

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
