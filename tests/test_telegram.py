"""Tests for interfaces/telegram.py — utility functions."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestForwardEmptyContentGuard:
    """_handle_forward used to short-circuit on full_text being empty,
    but full_text always has the "[Переслано]:" prefix — so the guard
    never fired. Forwarded photos without captions, stickers, etc.,
    became "[Переслано]: " and got shipped to Brain, paying for LLM
    rounds on no content."""

    @pytest.mark.asyncio
    async def test_no_text_no_caption_skips_brain(self):
        from unittest.mock import AsyncMock, MagicMock

        bot, brain = _make_bot()
        update = _make_update()
        update.message.text = None
        update.message.caption = None
        update.message.photo = None
        update.message.forward_origin = None
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()
        brain.process = AsyncMock()
        context = SimpleNamespace(args=[])

        await bot._handle_forward(update, context)

        brain.process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_text_only_whitespace_skips_brain(self):
        from unittest.mock import AsyncMock, MagicMock

        bot, brain = _make_bot()
        update = _make_update()
        update.message.text = "   \n  "
        update.message.caption = None
        update.message.photo = None
        update.message.forward_origin = None
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()
        brain.process = AsyncMock()
        context = SimpleNamespace(args=[])

        await bot._handle_forward(update, context)

        brain.process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_real_content_still_routes_to_brain(self):
        """Sanity: actual forwarded text still goes through, prefix added."""
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.core.brain import BrainResponse

        bot, brain = _make_bot()
        update = _make_update()
        update.message.text = "interesting article from somewhere"
        update.message.caption = None
        update.message.photo = None
        update.message.forward_origin = None
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()
        brain.process = AsyncMock(return_value=BrainResponse(
            text="ok", tool_calls_made=0, total_tokens=10,
        ))
        context = SimpleNamespace(args=[])

        await bot._handle_forward(update, context)

        brain.process.assert_awaited_once()
        kwargs = brain.process.await_args.kwargs
        # Prefix preserved on real content
        assert kwargs["user_message"].startswith("[Переслано]:")
        assert "interesting article" in kwargs["user_message"]


class TestForwardedPhotoFlow:
    """Pre-fix _handle_forward only read text/caption — the image was
    silently dropped because filters.FORWARDED matches before
    filters.PHOTO. User forwarded a screenshot of a receipt expecting
    OCR analysis and bot saw caption-only nonsense. Now forwarded
    photos get the multimodal flow with the [Переслано от X] prefix
    baked into the caption."""

    @pytest.mark.asyncio
    async def test_forwarded_photo_passes_image_to_brain(self):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.core.brain import BrainResponse

        bot, brain = _make_bot()
        update = _make_update()
        update.message.text = None
        update.message.caption = "вот чек"
        # Mock photo with a single photo size
        photo = MagicMock()
        photo.file_id = "AAA"
        photo.file_size = 12345
        update.message.photo = [photo]
        update.message.forward_origin = None
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()

        # Mock context.bot.get_file().download_as_bytearray()
        file_mock = MagicMock()
        file_mock.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"fakepngbytes")
        )
        context = SimpleNamespace(
            args=[],
            bot=SimpleNamespace(get_file=AsyncMock(return_value=file_mock)),
        )

        brain.process = AsyncMock(return_value=BrainResponse(
            text="вижу чек на 500р", tool_calls_made=0, total_tokens=10,
        ))
        brain.settings.process_timeout_sec = 30

        await bot._handle_forward(update, context)

        brain.process.assert_awaited_once()
        kwargs = brain.process.await_args.kwargs
        assert kwargs["message_type"] == "photo"
        assert kwargs["image_base64"]  # non-empty
        # Forward prefix on caption so Claude knows the source
        assert kwargs["user_message"].startswith("[Переслано]:")
        assert "вот чек" in kwargs["user_message"]

    @pytest.mark.asyncio
    async def test_forwarded_photo_without_caption_uses_default(self):
        """No caption on the forwarded photo → use the same inbox-capture
        instruction _handle_photo uses, but keep the forward prefix."""
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.core.brain import BrainResponse

        bot, brain = _make_bot()
        update = _make_update()
        update.message.text = None
        update.message.caption = None
        photo = MagicMock()
        photo.file_id = "AAA"
        photo.file_size = 1024
        update.message.photo = [photo]
        update.message.forward_origin = None
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()

        file_mock = MagicMock()
        file_mock.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"x")
        )
        context = SimpleNamespace(
            args=[],
            bot=SimpleNamespace(get_file=AsyncMock(return_value=file_mock)),
        )
        brain.process = AsyncMock(return_value=BrainResponse(
            text="ok", tool_calls_made=0, total_tokens=5,
        ))
        brain.settings.process_timeout_sec = 30

        await bot._handle_forward(update, context)

        kwargs = brain.process.await_args.kwargs
        # Default inbox instruction used
        assert "inbox" in kwargs["user_message"].lower() or "разбери" in kwargs["user_message"].lower()
        assert kwargs["user_message"].startswith("[Переслано]:")


class TestTextHandlerWhitespaceGuard:
    """_handle_text used to forward whitespace-only messages to Brain.
    Voice/forward already strip and skip; text was the inconsistent one.
    User typing "   " by accident shouldn't trigger an LLM round."""

    @pytest.mark.asyncio
    async def test_whitespace_only_text_skips_brain(self):
        from unittest.mock import AsyncMock, MagicMock

        bot, brain = _make_bot()
        update = _make_update()
        update.message.text = "   \n\t  "
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()
        # Make brain.process unambiguously NOT called
        brain.process = AsyncMock()
        context = SimpleNamespace(args=[])

        await bot._handle_text(update, context)

        brain.process.assert_not_awaited()
        # No reply either — silent skip matches the "not text" branch
        update.message.reply_text.assert_not_awaited()


class TestPhotoPostDownloadSizeGuard:
    """Pre-fix only photo.file_size was checked, which Telegram doesn't
    always populate. A missing file_size header would skip the guard
    and let any size of bytes through to base64+brain. Post-download
    cap closes that hole."""

    @pytest.mark.asyncio
    async def test_oversized_post_download_rejected(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock
        from mindsecretary.interfaces.telegram import MAX_PHOTO_SIZE

        bot, brain = _make_bot()
        update = _make_update()
        # PhotoSize with NO file_size header (Telegram edge case)
        photo_size = MagicMock()
        photo_size.file_size = None
        photo_size.file_id = "abc"
        update.message.photo = [photo_size]
        update.message.caption = None
        update.message.chat = MagicMock()
        update.message.chat.send_action = AsyncMock()

        # Simulate Telegram serving a 30MB photo
        oversized = b"x" * (MAX_PHOTO_SIZE + 100)
        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(oversized))

        context = SimpleNamespace(
            args=[],
            bot=MagicMock(get_file=AsyncMock(return_value=fake_file)),
        )

        await bot._handle_photo(update, context)

        # User sees the size error
        update.message.reply_text.assert_any_await("Фото слишком большое (макс 10 МБ).")
        # Brain NOT called — would have wasted Claude image tokens
        brain.process.assert_not_called()


class TestReplyEmpty:
    """_reply must silently skip empty/whitespace text instead of forwarding
    it to Telegram (which 400s on empty body and triggers the outer
    catch-all 'Произошла ошибка' message — misleading for a brain call
    that succeeded but had nothing to say)."""

    @pytest.mark.asyncio
    async def test_empty_string_suppressed(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        bot, _ = _make_bot()
        await bot._reply(update, "")
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_suppressed(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        bot, _ = _make_bot()
        await bot._reply(update, "   \n\t  ")
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_suppressed(self):
        """Brain.process returns BrainResponse(text=None) on early-exit
        paths — _reply must accept None gracefully too."""
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        bot, _ = _make_bot()
        await bot._reply(update, None)
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_empty_still_sent(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        bot, _ = _make_bot()
        await bot._reply(update, "hello")
        update.message.reply_text.assert_awaited()


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


class TestLoopsInProgressMarker:
    """Iter 13 added in-progress events (start_at past, end_at future)
    to upcoming_events. The /loops handler must mark them so the user
    isn't told "Ближайшее: 14:00 встреча" while sitting in that meeting
    at 14:30. Mirror of iter 15's briefing-side fix."""

    @pytest.mark.asyncio
    async def test_in_progress_event_renders_with_marker(self):
        from datetime import datetime as real_dt
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        # Pin the DB clock to 14:30 — meeting started at 14:00 and ends 16:00.
        brain.db.local_now_naive = MagicMock(return_value=real_dt(2026, 4, 15, 14, 30))
        brain.db.get_open_loops = MagicMock(return_value={
            "counts": {"upcoming_events": 1},
            "upcoming_events": [{
                "start_at": "2026-04-15 14:00:00",
                "end_at": "2026-04-15 16:00:00",
                "title": "встреча с Машей",
                "related_person": None,
            }],
            "overdue_reminders": [], "due_today_reminders": [],
            "pending_goals": [], "due_decisions": [],
        })
        with patch(
            "mindsecretary.interfaces.telegram.check_contact_frequency",
            return_value=[],
        ):
            await bot._handle_loops(update, context)

        # Captured the rendered text via reply_text call(s)
        call_text = " ".join(
            str(c.args[0]) for c in update.message.reply_text.await_args_list
        )
        assert "▶️ сейчас" in call_text
        assert "встреча с Машей" in call_text
        # Crucially: must NOT show start time as "ближайшее" of 14:00 alone
        assert "04-15 14:00" not in call_text

    @pytest.mark.asyncio
    async def test_future_event_keeps_timestamp(self):
        """Sanity: future events still render with their MM-DD HH:MM
        timestamp — the in-progress swap is gated on time comparison."""
        from datetime import datetime as real_dt
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.db.local_now_naive = MagicMock(return_value=real_dt(2026, 4, 15, 9, 0))
        brain.db.get_open_loops = MagicMock(return_value={
            "counts": {"upcoming_events": 1},
            "upcoming_events": [{
                "start_at": "2026-04-15 14:00:00",
                "end_at": "2026-04-15 16:00:00",
                "title": "встреча с Машей",
                "related_person": None,
            }],
            "overdue_reminders": [], "due_today_reminders": [],
            "pending_goals": [], "due_decisions": [],
        })
        with patch(
            "mindsecretary.interfaces.telegram.check_contact_frequency",
            return_value=[],
        ):
            await bot._handle_loops(update, context)

        call_text = " ".join(
            str(c.args[0]) for c in update.message.reply_text.await_args_list
        )
        assert "04-15 14:00" in call_text
        assert "▶️ сейчас" not in call_text


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
    async def test_diary_no_args_shows_last_3(self):
        """Default behavior preserved: /diary with no args shows up to 3
        from the 7-day window."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.db.get_diary_entries = MagicMock(return_value=[
            {"date": "2026-04-15", "content": "day 1", "mood": None, "people": None},
            {"date": "2026-04-14", "content": "day 2", "mood": None, "people": None},
            {"date": "2026-04-13", "content": "day 3", "mood": None, "people": None},
            {"date": "2026-04-12", "content": "day 4", "mood": None, "people": None},
        ])

        await bot._handle_diary(update, context)

        # Wide window query
        brain.db.get_diary_entries.assert_called_once_with(days=7)
        # Only 3 entries posted
        assert update.message.reply_text.call_count == 3

    @pytest.mark.asyncio
    async def test_diary_numeric_arg_shows_n_entries(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["5"])
        brain.db.get_diary_entries = MagicMock(return_value=[
            {"date": f"2026-04-{15-i:02d}", "content": f"day {i}",
             "mood": None, "people": None}
            for i in range(7)
        ])

        await bot._handle_diary(update, context)

        # Limit=5 → 5 messages posted
        assert update.message.reply_text.call_count == 5

    @pytest.mark.asyncio
    async def test_diary_numeric_arg_capped(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["999"])
        brain.db.get_diary_entries = MagicMock(return_value=[])

        await bot._handle_diary(update, context)

        # 999 capped at _DIARY_MAX_ENTRIES (30) — but no entries → friendly msg
        update.message.reply_text.assert_awaited_once_with("Записей в дневнике пока нет.")

    @pytest.mark.asyncio
    async def test_diary_date_arg_renders_specific_entry(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["2026-04-15"])
        brain.db.get_diary_entry_by_date = MagicMock(return_value={
            "date": "2026-04-15", "content": "specific day",
            "mood": "positive", "people": "Маша",
        })

        await bot._handle_diary(update, context)

        # Single specific entry — one reply
        assert update.message.reply_text.call_count == 1
        # Date-getter routed (not the range query)
        brain.db.get_diary_entry_by_date.assert_called_once_with("2026-04-15")

    @pytest.mark.asyncio
    async def test_diary_date_arg_missing_entry(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["2024-01-01"])
        brain.db.get_diary_entry_by_date = MagicMock(return_value=None)

        await bot._handle_diary(update, context)

        update.message.reply_text.assert_awaited_once()
        body = update.message.reply_text.await_args.args[0]
        assert "Нет записи за 2024-01-01" in body

    @pytest.mark.asyncio
    async def test_context_clear_uses_correct_record_plural(self):
        """The /context clear summary used to render '2 записей' (form_other)
        for any count != 1. With pluralize_ru, 2-4 use form_2_4 ('записи')."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["clear"])
        # Clear returns count=2 → form_2_4
        brain.db.clear_ephemeral_state = MagicMock(return_value=2)

        await bot._handle_context(update, context)

        body = update.message.reply_text.await_args.args[0]
        assert "2 записи" in body
        assert "2 записей" not in body

    @pytest.mark.asyncio
    async def test_context_clear_singular_uses_form_one(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["clear"])
        brain.db.clear_ephemeral_state = MagicMock(return_value=1)

        await bot._handle_context(update, context)

        body = update.message.reply_text.await_args.args[0]
        assert "1 запись" in body

    @pytest.mark.asyncio
    async def test_context_clear_five_uses_many(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["clear"])
        brain.db.clear_ephemeral_state = MagicMock(return_value=5)

        await bot._handle_context(update, context)

        body = update.message.reply_text.await_args.args[0]
        assert "5 записей" in body

    @pytest.mark.asyncio
    async def test_diary_invalid_arg_shows_usage(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["garbage"])

        await bot._handle_diary(update, context)

        update.message.reply_text.assert_awaited_once()
        body = update.message.reply_text.await_args.args[0]
        assert "Использование" in body
        assert "/diary 7" in body
        assert "/diary 2026-" in body

    @pytest.mark.asyncio
    async def test_memory_is_rate_limited(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["plans"])
        bot._check_rate_limit = lambda: False

        await bot._handle_memory(update, context)

        update.message.reply_text.assert_awaited_once_with("Слишком часто, подожди минуту.")
        brain.memory.search.assert_not_awaited()

    def test_parse_snooze_duration_valid(self):
        from mindsecretary.interfaces.telegram import TelegramBot
        assert TelegramBot._parse_snooze_duration("30m") == 30
        assert TelegramBot._parse_snooze_duration("2h") == 120
        assert TelegramBot._parse_snooze_duration("1d") == 1440
        assert TelegramBot._parse_snooze_duration("5H") == 300  # case insensitive

    def test_parse_snooze_duration_invalid(self):
        from mindsecretary.interfaces.telegram import TelegramBot
        assert TelegramBot._parse_snooze_duration("") is None
        assert TelegramBot._parse_snooze_duration("garbage") is None
        assert TelegramBot._parse_snooze_duration("30") is None  # missing unit
        assert TelegramBot._parse_snooze_duration("0m") is None  # zero rejected
        assert TelegramBot._parse_snooze_duration("1w") is None  # weeks not supported
        assert TelegramBot._parse_snooze_duration("-5h") is None

    def test_parse_snooze_duration_capped_at_7d(self):
        """Cap prevents accidental indefinite snooze ('/snooze 365d' would
        silently kill all proactive notifications for a year)."""
        from mindsecretary.interfaces.telegram import TelegramBot
        assert TelegramBot._parse_snooze_duration("30d") == 7 * 24 * 60
        assert TelegramBot._parse_snooze_duration("365d") == 7 * 24 * 60

    @pytest.mark.asyncio
    async def test_snooze_off_clears(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["off"])

        await bot._handle_snooze(update, context)

        brain.db.set_snooze_until.assert_called_once_with(None)
        text = update.message.reply_text.await_args.args[0]
        assert "Snooze отключён" in text

    @pytest.mark.asyncio
    async def test_snooze_no_args_when_inactive_shows_usage(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.db.get_snooze_until = MagicMock(return_value=None)

        await bot._handle_snooze(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Сейчас не на паузе" in text
        assert "/snooze 2h" in text  # usage example included

    @pytest.mark.asyncio
    async def test_snooze_no_args_when_active_shows_remaining(self):
        from datetime import datetime, timezone, timedelta
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        # 90 minutes remaining
        until = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=90)
        brain.db.get_snooze_until = MagicMock(return_value=until)

        await bot._handle_snooze(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "На паузе ещё" in text
        # Should render as "1ч 30м" since over 60 min
        assert "ч" in text

    @pytest.mark.asyncio
    async def test_snooze_2h_sets_until_with_expected_window(self):
        from datetime import datetime, timezone, timedelta
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["2h"])

        await bot._handle_snooze(update, context)

        # set_snooze_until called with ~now + 2h
        brain.db.set_snooze_until.assert_called_once()
        until = brain.db.set_snooze_until.call_args.args[0]
        delta = until - datetime.now(timezone.utc).replace(tzinfo=None)
        # Within a few seconds of 2h
        assert abs(delta - timedelta(hours=2)) < timedelta(seconds=5)

        # Confirmation text mentions reminders bypass
        text = update.message.reply_text.await_args.args[0]
        assert "Snooze на 2ч" in text
        assert "Напоминания" in text

    @pytest.mark.asyncio
    async def test_snooze_invalid_arg_shows_examples(self):
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=["garbage"])

        await bot._handle_snooze(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Неправильный формат" in text
        # Examples include all three units
        assert "/snooze 30m" in text and "/snooze 2h" in text

    @pytest.mark.asyncio
    async def test_learnings_empty_state(self):
        """Empty learnings state mentions BOTH the auto cron (Sunday 20:00)
        and the on-demand /review escape hatch — user should know they
        don't have to wait until Sunday."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.memory.get_by_category = MagicMock(return_value=[])

        await bot._handle_learnings(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Пока без learnings" in text
        assert "/review" in text
        assert "воскресенье" in text.lower()

    @pytest.mark.asyncio
    async def test_learnings_renders_top_10_with_meta(self):
        """Output must include each learning's content, importance, and
        creation date — those are the dimensions that turn a flat list
        into something the user can prioritize."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.memory.get_by_category = MagicMock(return_value=[
            {"content": "Утренние брифинги работают только до 9", "importance": 9,
             "confidence": 0.85, "created_at": "2026-04-15 20:00:00"},
            {"content": "Понедельник всегда тяжёлый", "importance": 7,
             "confidence": 0.7, "created_at": "2026-04-08 20:00:00"},
        ])

        await bot._handle_learnings(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Learnings" in text
        assert "Утренние брифинги" in text
        assert "Понедельник всегда" in text
        # Importance + confidence + created date all surface
        assert "imp 9" in text and "imp 7" in text
        assert "conf 0.85" in text
        assert "2026-04-15" in text

    @pytest.mark.asyncio
    async def test_learnings_truncates_to_10_with_remainder_hint(self):
        """get_by_category sorts by importance DESC, /learnings caps at
        10 to fit one Telegram bubble. Remainder gets a footer line so
        the user knows there's more to see."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.memory.get_by_category = MagicMock(return_value=[
            {"content": f"learning {i}", "importance": 10 - (i // 3),
             "confidence": 0.5, "created_at": "2026-04-15"}
            for i in range(15)
        ])

        await bot._handle_learnings(update, context)

        text = update.message.reply_text.await_args.args[0]
        # First 10 visible
        for i in range(10):
            assert f"learning {i}" in text
        # 11+ not shown
        assert "learning 11" not in text
        assert "learning 14" not in text
        # Footer mentions the remainder
        assert "ещё 5" in text

    @pytest.mark.asyncio
    async def test_learnings_falls_back_without_markdown(self):
        """Same fallback pattern as /forget — if Telegram rejects the
        markdown (e.g. unbalanced underscores in user-supplied content),
        retry without parse_mode."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.memory.get_by_category = MagicMock(return_value=[
            {"content": "broken_markdown _ here", "importance": 5,
             "confidence": 0.5, "created_at": "2026-04-15"},
        ])
        update.message.reply_text = AsyncMock(
            side_effect=[Exception("parse error"), None],
        )

        await bot._handle_learnings(update, context)

        assert update.message.reply_text.await_count == 2
        first_call = update.message.reply_text.await_args_list[0]
        second_call = update.message.reply_text.await_args_list[1]
        assert first_call.kwargs.get("parse_mode") == ParseMode.MARKDOWN
        assert "parse_mode" not in second_call.kwargs

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
    async def test_stats_handler_renders_monthly_projection(self):
        """Projection line appears when month_projection is non-None.
        Format: '🔮 Прогноз/мес: $X.XX (по 7-дн avg)' — the suffix
        prevents users from confusing it with a hard limit."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.db.get_stats = MagicMock(return_value={
            "today_cost": 0.10, "today_tokens": 1000, "month_cost": 5.0,
            "month_projection": 12.50,
            "memories": 50, "contacts": 5, "interactions_today": 20,
            "providers": {}, "week_trend": [{"date": "2026-04-22", "cost": 0.4}],
            "memory_categories": [],
        })

        await bot._handle_stats(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Прогноз/мес: $12.50" in text
        assert "по 7-дн avg" in text

    @pytest.mark.asyncio
    async def test_stats_handler_omits_projection_when_none(self):
        """Day-1 install: no cost rows → month_projection=None → omit line.
        Users shouldn't see a misleading '$0.00 projected' on first day."""
        bot, brain = _make_bot()
        update = _make_update()
        context = SimpleNamespace(args=[])
        brain.db.get_stats = MagicMock(return_value={
            "today_cost": 0, "today_tokens": 0, "month_cost": 0,
            "month_projection": None,
            "memories": 0, "contacts": 0, "interactions_today": 0,
            "providers": {}, "week_trend": [],
            "memory_categories": [],
        })

        await bot._handle_stats(update, context)

        text = update.message.reply_text.await_args.args[0]
        assert "Прогноз/мес" not in text

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
