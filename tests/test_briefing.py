"""BriefingGenerator output — verify habit progress lands in the
evening summary system prompt so Claude can talk about streaks
instead of inventing them.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mindsecretary.core.config import Profile
from mindsecretary.core.database import Database
from mindsecretary.proactive.briefing import BriefingGenerator


def _profile(tz: str = "Asia/Almaty") -> Profile:
    return Profile(
        name="Test", city="Almaty", timezone=tz,
        home_coords=[43.25, 76.9], work_coords=[43.25, 76.9],
        wake_up="07:00", work_start="09:00", work_end="18:00",
        sleep="23:00", commute_method="авто", commute_minutes=30,
        style="кратко", language="ru", notification_limit=10,
        quiet_hours=["23:00", "07:00"],
        priorities=[], dislikes=[],
    )


def _make_briefing(tmp_path: Path) -> tuple[BriefingGenerator, Database, MagicMock]:
    db = Database(tmp_path / "test.db", timezone="Asia/Almaty")
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
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(text="итог"))
    bg = BriefingGenerator(
        llm=llm, memory=MagicMock(), db=db, weather=None,
        profile=_profile(),
    )
    return bg, db, llm


class TestEveningHabitsSlot:
    @pytest.mark.asyncio
    async def test_active_streak_renders_in_prompt(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        # 4 consecutive done days — counts as an "active" streak (>=3)
        today = db._now()
        for i in range(4):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            db.log_habit("зарядка", done=True, date=d)

        await bg.generate_evening()

        system_prompt = llm.chat.call_args.kwargs["system"]
        assert "🔥 Серии" in system_prompt
        assert "зарядка" in system_prompt
        assert "4д" in system_prompt

    @pytest.mark.asyncio
    async def test_unlogged_today_surfaces_gentle_nudge(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        today = db._now()
        # Habit "yoga" — log only yesterday, NOT today; should land in
        # "Не отмечено сегодня" list
        db.log_habit("yoga", done=True,
                     date=(today - timedelta(days=1)).strftime("%Y-%m-%d"))

        await bg.generate_evening()

        system_prompt = llm.chat.call_args.kwargs["system"]
        assert "Не отмечено сегодня" in system_prompt
        assert "yoga" in system_prompt

    @pytest.mark.asyncio
    async def test_no_habits_renders_explicit_placeholder(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        # No habits logged anywhere — placeholder routes Claude to skip
        # the section instead of hallucinating progress
        await bg.generate_evening()

        system_prompt = llm.chat.call_args.kwargs["system"]
        assert "Привычки не отслеживаются" in system_prompt

    @pytest.mark.asyncio
    async def test_streak_under_threshold_not_listed(self, tmp_path):
        """Sub-3-day streaks should NOT be flexed as 'active series' — the
        rule is to celebrate sustained behavior, not 1-day blips."""
        bg, db, llm = _make_briefing(tmp_path)
        today_str = db._now().strftime("%Y-%m-%d")
        db.log_habit("чтение", done=True, date=today_str)

        await bg.generate_evening()

        system_prompt = llm.chat.call_args.kwargs["system"]
        assert "🔥 Серии" not in system_prompt  # 1d streak hidden


class TestFormatEventLine:
    """_format_event_line is the single source of truth for how events
    render into briefing prompts. Pre-consolidation, three different
    sites had three different formats — evening dropped time entirely."""

    def test_basic_time_plus_title(self):
        line = BriefingGenerator._format_event_line({
            "start_at": "2026-04-15 09:00:00",
            "title": "стандап",
        })
        assert line == "- 09:00 стандап"

    def test_renders_person_and_location(self):
        line = BriefingGenerator._format_event_line({
            "start_at": "2026-04-15 13:00:00",
            "title": "обед",
            "related_person": "Олег",
            "location": "Кафе Пушкин",
        })
        # Both extras inside the parens, comma-separated
        assert "обед" in line
        assert "с Олег" in line
        assert "где: Кафе Пушкин" in line
        assert line.startswith("- 13:00 ")

    def test_empty_extras_omits_parens(self):
        """No related_person, no location — line ends with the title, no
        empty parens dangling."""
        line = BriefingGenerator._format_event_line({
            "start_at": "2026-04-15 10:00:00",
            "title": "созвон",
        })
        assert "(" not in line
        assert "созвон" in line

    def test_missing_start_at_falls_back_to_question_marks(self):
        """Bad data shouldn't break the briefing — render '??:??' so the
        reader sees something is up rather than an empty time slot."""
        line = BriefingGenerator._format_event_line({
            "start_at": "",
            "title": "broken",
        })
        assert "??:??" in line

    def test_renders_same_day_end_at_as_range(self):
        """Mirror tools._handle_get_events: same-day end_at shown as
        'HH:MM-HH:MM' so Claude can reason about duration during the
        briefing."""
        line = BriefingGenerator._format_event_line({
            "start_at": "2026-04-15 14:00:00",
            "end_at": "2026-04-15 16:00:00",
            "title": "встреча",
        })
        assert "14:00-16:00 встреча" in line

    def test_omits_cross_day_end_at(self):
        """Cross-day end skipped — the dash would hide the day boundary
        and read wrong ('22:00-06:00' looks 8h within one day, not an
        overnight span)."""
        line = BriefingGenerator._format_event_line({
            "start_at": "2026-04-15 22:00:00",
            "end_at": "2026-04-16 06:00:00",
            "title": "ночная смена",
        })
        # No range
        assert "-06:00" not in line
        assert "22:00 ночная смена" in line

    def test_end_at_missing_renders_start_only(self):
        line = BriefingGenerator._format_event_line({
            "start_at": "2026-04-15 09:00:00",
            "end_at": None,
            "title": "стандап",
        })
        assert "09:00 стандап" in line
        assert "-" not in line.replace("- 09", "X")  # the dash from list bullet only


class TestFormatOpenLoops:
    """_format_open_loops feeds the briefing's 'Хвосты и риски' slot.
    Goals + decisions only appear in this section, so the title/desc
    has to surface — count alone leaves the user blind to WHAT'S
    overdue."""

    def test_empty_returns_no_loops_placeholder(self):
        result = BriefingGenerator._format_open_loops({"counts": {}})
        assert result == "Критичных хвостов нет."

    def test_pending_goals_renders_first_title(self):
        result = BriefingGenerator._format_open_loops({
            "counts": {"pending_goals": 2},
            "pending_goals": [
                {"title": "починить раковину", "priority": "high"},
                {"title": "купить продукты", "priority": "medium"},
            ],
        })
        assert "Незакрытые цели на сегодня: 2" in result
        # Highest-priority goal title surfaces, with priority tag
        assert "починить раковину" in result
        assert "[high]" in result
        # Second goal title NOT in output — keep briefing brief
        assert "купить продукты" not in result

    def test_due_decisions_renders_first_description(self):
        result = BriefingGenerator._format_open_loops({
            "counts": {"due_decisions": 3},
            "due_decisions": [
                {"description": "купить велосипед или нет"},
                {"description": "переехать на дачу"},
                {"description": "сменить тариф"},
            ],
        })
        assert "Решения с просроченным follow-up: 3" in result
        # Most-overdue (first in list) description surfaces
        assert "купить велосипед" in result

    def test_overdue_reminders_count_only(self):
        """Reminders are listed in full elsewhere (reminders_text), so
        open_loops summarizes with count to avoid duplication."""
        result = BriefingGenerator._format_open_loops({
            "counts": {"overdue_reminders": 3, "due_today_reminders": 2},
            "overdue_reminders": [
                {"text": "позвонить маме", "trigger_at": "2026-04-10 10:00:00"},
            ],
        })
        # Both lines render
        assert "Просроченные напоминания: 3" in result
        assert "Напоминания до конца дня: 2" in result
        # But reminder text is NOT inlined — that lives in reminders_text
        assert "позвонить маме" not in result

    def test_pending_goals_without_list_falls_back_to_count(self):
        """Defensive: if counts > 0 but list is empty (shouldn't happen
        normally — DB inconsistency or limit_per_section=0), we still
        render the count alone instead of breaking."""
        result = BriefingGenerator._format_open_loops({
            "counts": {"pending_goals": 5},
            "pending_goals": [],
        })
        assert "Незакрытые цели на сегодня: 5" in result
        # No "—" trailer when no list to source the title from
        assert "—" not in result

    def test_in_progress_event_labeled_as_now_running(self):
        """Iter 13 added in-progress events to upcoming_events. Without
        passing `now_local`, the legacy "Ближайшее" framing applied —
        that's misleading when the meeting is actively running. With
        now_local the format swaps to "Сейчас идёт: HH:MM-HH:MM"."""
        from datetime import datetime as real_dt
        anchor = real_dt(2026, 4, 15, 14, 30)
        result = BriefingGenerator._format_open_loops(
            {
                "counts": {"upcoming_events": 1},
                "upcoming_events": [{
                    "start_at": "2026-04-15 14:00:00",
                    "end_at": "2026-04-15 16:00:00",
                    "title": "встреча с Машей",
                }],
            },
            now_local=anchor,
        )
        assert "Сейчас идёт: 14:00-16:00 встреча с Машей" in result
        assert "Ближайшее событие" not in result

    def test_future_event_keeps_blizhayshee_label(self):
        """Sanity: future events still get the legacy "Ближайшее" label
        (not in-progress). Confirms the swap is gated on the time
        comparison, not always."""
        from datetime import datetime as real_dt
        anchor = real_dt(2026, 4, 15, 9, 0)  # 09:00 — meeting at 14:00 is future
        result = BriefingGenerator._format_open_loops(
            {
                "counts": {"upcoming_events": 1},
                "upcoming_events": [{
                    "start_at": "2026-04-15 14:00:00",
                    "end_at": "2026-04-15 16:00:00",
                    "title": "встреча с Машей",
                }],
            },
            now_local=anchor,
        )
        assert "Ближайшее событие: 14:00 встреча с Машей" in result
        assert "Сейчас идёт" not in result

    def test_no_now_local_keeps_legacy_label(self):
        """Backwards compat: callers that don't pass now_local (e.g.
        existing tests) get the original "Ближайшее" framing — tools.py
        consumer also doesn't pass it."""
        result = BriefingGenerator._format_open_loops({
            "counts": {"upcoming_events": 1},
            "upcoming_events": [{
                "start_at": "2026-04-15 14:00:00",
                "title": "встреча",
            }],
        })
        assert "Ближайшее событие: 14:00 встреча" in result


class TestFormatReminderLine:
    """_format_reminder_line surfaces trigger_at so Claude can sequence
    the day correctly. Pre-consolidation the briefing dropped the time,
    rendering 'позвонить маме' with no clue when."""

    def test_includes_trigger_time(self):
        line = BriefingGenerator._format_reminder_line({
            "trigger_at": "2026-04-15 18:00:00",
            "text": "позвонить маме",
        })
        assert "2026-04-15 18:00" in line
        assert "позвонить маме" in line

    def test_renders_recurrence_marker(self):
        """Recurring reminders should be visually distinct from one-offs
        — Claude treats 'каждый понедельник X' differently from 'X на
        15-е число'."""
        line = BriefingGenerator._format_reminder_line({
            "trigger_at": "2026-04-15 09:00:00",
            "text": "стандап",
            "recurrence": "weekly",
        })
        assert "(weekly)" in line

    def test_missing_trigger_falls_back_to_question_marks(self):
        line = BriefingGenerator._format_reminder_line({
            "trigger_at": None,
            "text": "broken row",
        })
        assert "??" in line


class TestMorningRemindersIncludeTime:
    @pytest.mark.asyncio
    async def test_morning_reminders_show_trigger_time(self, tmp_path):
        """User-visible quality: pre-fix Claude saw 'позвонить маме'
        without time, so morning briefings often suggested doing it
        'позже' instead of pointing at 18:00. Real annoying."""
        bg, db, llm = _make_briefing(tmp_path)
        # Morning briefing calls memory.search twice (context + promises)
        # — replace with AsyncMock so the awaitable contract is honored.
        bg.memory.search = AsyncMock(return_value=[])
        db.create_reminder("позвонить маме", "2099-04-15 18:00:00", "high")

        await bg.generate_morning()

        prompt = llm.chat.call_args.kwargs["system"]
        assert "18:00" in prompt
        assert "позвонить маме" in prompt


class TestDiaryPeopleDeclensions:
    """Diary's people_today set used a plain lowercase substring match
    that missed every Russian declension. Fix: stem-based heuristic
    catches Маша/Маше/Машей/Машу. Test exercises the actual
    generate_diary path (not the helper directly) to lock in the
    integration."""

    @pytest.mark.asyncio
    async def test_declined_form_now_detected(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        bg.memory.search = AsyncMock(return_value=[])
        # Contact in nominative
        db.upsert_contact("Маша", relation="друг")
        # User mentions her in instrumental case (with X = Машей)
        db.log_interaction("in", "text", "сегодня встретился с Машей в кафе")
        db.log_interaction("in", "text", "обсудили проект")
        db.log_interaction("in", "text", "ушёл к 18")

        await bg.generate_diary()

        prompt = llm.chat.call_args.kwargs["system"]
        # The diary prompt's "people" slot must now include Маша even
        # though the message used "Машей". Pre-fix the slot read "Никого".
        assert "Маша" in prompt

    @pytest.mark.asyncio
    async def test_no_mention_means_no_addition(self, tmp_path):
        """Sanity: contacts that aren't mentioned anywhere shouldn't
        leak into people_today via accidental stem collision."""
        bg, db, llm = _make_briefing(tmp_path)
        bg.memory.search = AsyncMock(return_value=[])
        db.upsert_contact("Олег", relation="коллега")
        # Three messages, none about Олег
        db.log_interaction("in", "text", "купил хлеб")
        db.log_interaction("in", "text", "погулял с собакой")
        db.log_interaction("in", "text", "посмотрел кино")

        await bg.generate_diary()

        prompt = llm.chat.call_args.kwargs["system"]
        assert "Олег" not in prompt


class TestDiaryInboundGuard:
    """generate_diary used to fire whenever total interactions >= 3, but
    proactive notifications (reminders, briefings, weather alerts) inflate
    the count. On a day with 0 user messages, the bot would write a
    "diary" from its own outputs — nonsense + wasted LLM call."""

    @pytest.mark.asyncio
    async def test_skips_when_only_outbound(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        # Five proactive notifications, zero user messages
        for i in range(5):
            db.log_interaction(
                direction="out", message_type="notification",
                content=f"reminder {i}",
            )

        result = await bg.generate_diary()
        assert result is None
        # Critical: LLM not called — money saved on bot-only days
        llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_when_inbound_present(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        db.log_interaction(direction="in", content="был тяжёлый день",
                           message_type="text")
        db.log_interaction(direction="out", content="понимаю",
                           message_type="chat")
        db.log_interaction(direction="in", content="устал",
                           message_type="text")

        result = await bg.generate_diary()
        # Diary generated; mock LLM returns "итог" which gets saved
        assert result is not None
        llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_relationship_alert_pluralizes_days(self, tmp_path):
        """Pre-fix the rel_text line hardcoded "дней" — Claude saw
        "не общались 31 дней" in DIARY_SYSTEM_PROMPT and could echo
        it. Now uses pluralize_ru."""
        bg, db, llm = _make_briefing(tmp_path)
        # Need at least one inbound message + 3 total interactions to
        # pass the inbound guard.
        db.log_interaction(direction="in", content="вечер", message_type="text")
        db.log_interaction(direction="out", content="ок", message_type="chat")
        db.log_interaction(direction="in", content="устал", message_type="text")

        with patch(
            "mindsecretary.proactive.briefing.check_contact_frequency",
            return_value=[
                {"name": "Маша", "relation": "друг",
                 "days_since": 31, "mention_count": 5},
                {"name": "Олег", "relation": "коллега",
                 "days_since": 33, "mention_count": 4},
            ],
        ):
            await bg.generate_diary()

        prompt = llm.chat.call_args.kwargs["system"]
        # 31 → singular "день", 33 → few-form "дня"
        assert "не общались 31 день" in prompt
        assert "не общались 33 дня" in prompt
        # Old hardcoded form must NOT leak through
        assert "не общались 31 дней" not in prompt
        assert "не общались 33 дней" not in prompt


class TestEveningEventsTimeIncluded:
    @pytest.mark.asyncio
    async def test_today_events_show_time_and_location(self, tmp_path):
        """Pre-fix the evening summary's events_text only emitted titles —
        Claude saw 'today: ужин' with no clue when or where, and the
        wrap-up section read like a checklist of nouns. Post-fix, the
        helper emits time + title + extras consistently."""
        bg, db, llm = _make_briefing(tmp_path)
        # Anchor on the DB clock so 'today' matches what generate_evening
        # picks via tz_now(profile.timezone).
        today = db._now().strftime("%Y-%m-%d")
        db.create_event("ужин с Машей", f"{today} 19:00:00",
                        location="кафе Пушкин")

        await bg.generate_evening()

        prompt = llm.chat.call_args.kwargs["system"]
        # Time visible
        assert "19:00 ужин с Машей" in prompt
        # Location visible
        assert "кафе Пушкин" in prompt


class TestEveningInteractionLabels:
    """Pre-fix `interactions_text` rendered every notification with a
    generic "(notification)" tag. Claude couldn't tell a reminder
    firing apart from a birthday alert or weather warning when
    writing the evening recap. Now metadata.kind drives a more
    specific Russian label."""

    @pytest.mark.asyncio
    async def test_notification_kinds_render_distinct_labels(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        db.log_interaction(
            direction="out", message_type="notification",
            content="⏰ позвонить",
            metadata={"kind": "reminder", "reminder_id": "r"},
        )
        db.log_interaction(
            direction="out", message_type="notification",
            content="🎂 ДР Маши",
            metadata={"kind": "birthday_alert"},
        )
        db.log_interaction(
            direction="out", message_type="notification",
            content="🌧 дождь",
            metadata={"kind": "weather_alert"},
        )
        db.log_interaction(direction="in", message_type="text", content="ok")

        await bg.generate_evening()

        prompt = llm.chat.call_args.kwargs["system"]
        assert "(напоминание)" in prompt
        assert "(ДР)" in prompt
        assert "(погода)" in prompt
        # User text falls through unchanged
        assert "(text)" in prompt
        # Old generic tag must not surface for notifications
        assert "(notification)" not in prompt

    @pytest.mark.asyncio
    async def test_unknown_kind_falls_back_to_message_type(self, tmp_path):
        """If metadata.kind is something we don't recognize, fall back
        to message_type instead of dropping the line entirely."""
        bg, db, llm = _make_briefing(tmp_path)
        db.log_interaction(
            direction="out", message_type="notification",
            content="новое",
            metadata={"kind": "future_kind_we_dont_know_yet"},
        )
        db.log_interaction(direction="in", message_type="text", content="ok")
        await bg.generate_evening()
        prompt = llm.chat.call_args.kwargs["system"]
        # Falls back to "notification" since the kind isn't in the map
        assert "(notification) новое" in prompt

    @pytest.mark.asyncio
    async def test_malformed_metadata_falls_back_safely(self, tmp_path):
        """Garbled metadata JSON shouldn't crash — fall back to
        message_type as the displayed label."""
        bg, db, llm = _make_briefing(tmp_path)
        # Insert a row with non-JSON metadata bypassing log_interaction
        # which auto-jsonifies.
        db.db.execute(
            "INSERT INTO interactions (direction, message_type, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            ("out", "notification", "x", "this-is-not-json"),
        )
        db.db.commit()
        db.log_interaction(direction="in", message_type="text", content="hi")

        await bg.generate_evening()
        prompt = llm.chat.call_args.kwargs["system"]
        # Doesn't crash; falls back to message_type label
        assert "(notification) x" in prompt


class TestEveningRemindersCount:
    """Pre-fix the evening summary's "completed" reminder counter
    looked for message_type == "reminder", but check_reminders logs
    them as message_type="notification" with metadata.kind="reminder".
    The filter never matched, so the prompt got "0 напоминаний
    отправлено" hardcoded into every evening summary — Claude
    rendered a wrap-up that implied a quiet day even when the bot
    had fired 8 reminders."""

    @pytest.mark.asyncio
    async def test_counts_reminder_notifications_via_metadata(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        # Two reminder firings + one unrelated notification (briefing).
        db.log_interaction(
            direction="out", message_type="notification",
            content="⏰ Напоминание: позвонить",
            metadata={"kind": "reminder", "reminder_id": "r1"},
        )
        db.log_interaction(
            direction="out", message_type="notification",
            content="⏰ Напоминание: оплатить",
            metadata={"kind": "reminder", "reminder_id": "r2"},
        )
        db.log_interaction(
            direction="out", message_type="notification",
            content="☀️ Доброе утро",
            metadata={"kind": "morning_briefing"},
        )

        await bg.generate_evening()

        prompt = llm.chat.call_args.kwargs["system"]
        assert "2 напоминаний отправлено" in prompt
        # Sanity: morning briefing didn't get counted as a reminder
        assert "3 напоминаний отправлено" not in prompt

    @pytest.mark.asyncio
    async def test_zero_when_no_reminders_today(self, tmp_path):
        """No reminder firings → 0 sent. Other notification kinds must
        not bump the count."""
        bg, db, llm = _make_briefing(tmp_path)
        db.log_interaction(
            direction="out", message_type="notification",
            content="🎂 Сегодня ДР",
            metadata={"kind": "birthday_alert"},
        )
        await bg.generate_evening()
        prompt = llm.chat.call_args.kwargs["system"]
        assert "0 напоминаний отправлено" in prompt


class TestIsPersonInTitle:
    """3-char-stem heuristic for suppressing redundant '👤 person' lines
    when the title already names them. The naive substring approach
    fails on Russian declensions (Маша not in Машей), so we compare
    on the first 3 chars (the longest invariant prefix across cases).
    """

    def test_full_match(self):
        from mindsecretary.core import is_person_in_title
        assert is_person_in_title("Маша", "встреча с Машей") is True

    def test_declension_via_stem(self):
        """The point of the helper — Маша not literally in Машей, but
        Маш is in машей. Plain `in` would say False here."""
        from mindsecretary.core import is_person_in_title
        # Plain substring would fail
        assert "маша" not in "встреча с машей"
        # Helper succeeds
        assert is_person_in_title("Маша", "встреча с Машей") is True

    def test_unrelated(self):
        from mindsecretary.core import is_person_in_title
        assert is_person_in_title("Маша", "встреча с командой") is False

    def test_case_insensitive_both_sides(self):
        from mindsecretary.core import is_person_in_title
        assert is_person_in_title("МАША", "встреча с машей") is True
        assert is_person_in_title("маша", "ВСТРЕЧА С МАШЕЙ") is True

    def test_empty_returns_false(self):
        from mindsecretary.core import is_person_in_title
        assert is_person_in_title("", "title") is False
        assert is_person_in_title(None, "title") is False
        assert is_person_in_title("Маша", "") is False
        assert is_person_in_title("Маша", None) is False

    def test_short_name_misses_declension(self):
        """Documented limitation: 3-char names like 'Аня' / 'Оля' don't
        share a 3-char stem with their declensions ('Аней' / 'Оле'),
        so the helper returns False and the renderer keeps the redundant
        '👤 line. Visual repetition is the lesser evil vs lowering the
        stem to 2 chars and matching unrelated words ('ан' in 'анализ')."""
        from mindsecretary.core import is_person_in_title
        # Plain Маша/Машей still works (4 chars → stem "Маш" matches)
        assert is_person_in_title("Маша", "встреча с Машей") is True
        # But 3-char Аня/Аней can't share 3 chars
        assert is_person_in_title("Аня", "встреча с Аней") is False

    def test_olek_stem_matches_declensions(self):
        """Олег / Олега / Олегом — all share 'оле' (3-char stem)."""
        from mindsecretary.core import is_person_in_title
        for declined in ("встреча с Олегом", "обед с Олегом", "звонок Олегу"):
            assert is_person_in_title("Олег", declined) is True


class TestNotificationKindLabels:
    """Single source of truth for kind→label was duplicated across
    brain, tools, briefing. Iter 8 caught a drift (event_alert/reflection
    missing from two of the three). Now all three import from
    core.NOTIFICATION_KIND_LABELS — verify everyone sees the same map."""

    def test_brain_aliases_central_map(self):
        from mindsecretary.core import NOTIFICATION_KIND_LABELS
        from mindsecretary.core.brain import Brain
        # brain re-aliases — same dict object would be ideal but at
        # minimum the keys must match.
        assert Brain._NOTIFICATION_LABELS is NOTIFICATION_KIND_LABELS

    def test_tools_aliases_central_map(self):
        from mindsecretary.core import NOTIFICATION_KIND_LABELS
        from mindsecretary.llm.tools import ToolExecutor
        assert ToolExecutor._SEARCH_KIND_LABELS is NOTIFICATION_KIND_LABELS

    def test_briefing_overrides_birthday_label_only(self):
        """Briefing shortens birthday to 'ДР' for the per-line evening
        format. Other entries stay aligned with the central map."""
        from mindsecretary.core import NOTIFICATION_KIND_LABELS
        from mindsecretary.proactive.briefing import _NOTIFICATION_LABELS
        assert _NOTIFICATION_LABELS["birthday_alert"] == "ДР"
        for k, v in NOTIFICATION_KIND_LABELS.items():
            if k == "birthday_alert":
                continue
            assert _NOTIFICATION_LABELS[k] == v

    def test_central_map_covers_all_notification_kinds(self):
        """Every kind written by _send_proactive must have a label.
        Acts as a regression guard for the iter 8 / iter 25 drift class."""
        from mindsecretary.core import NOTIFICATION_KIND_LABELS
        # Every kind that any of the schedulers / monitor.py writes via
        # metadata.kind:
        expected = {
            "morning_briefing", "evening_summary", "diary",
            "weekly_review", "smart_question", "open_loops_nudge",
            "decision_followup", "birthday_alert", "weather_alert",
            "reminder", "event_alert", "event_reflection",
        }
        missing = expected - set(NOTIFICATION_KIND_LABELS)
        assert not missing, f"missing labels for: {missing}"


class TestFmtUtcToLocal:
    """Shared helper used by tools (LLM output) and telegram (user
    output). Pre-extraction each surface either rolled its own UTC
    conversion or sliced raw — leaving Asia/Almaty users seeing
    "yesterday" for memories saved past local midnight."""

    def test_converts_utc_to_local_string(self):
        from mindsecretary.core import fmt_utc_to_local
        # 22:00 UTC = 03:00 next-day Almaty
        out = fmt_utc_to_local("2026-05-07 22:00:00", "Asia/Almaty")
        assert out == "2026-05-08 03:00"

    def test_no_tz_falls_back_to_slice(self):
        from mindsecretary.core import fmt_utc_to_local
        # tz_name=None → return ts[:16] (matches pre-extraction shape)
        out = fmt_utc_to_local("2026-05-07 22:00:00", None)
        assert out == "2026-05-07 22:00"

    def test_empty_input_returns_question_mark(self):
        from mindsecretary.core import fmt_utc_to_local
        assert fmt_utc_to_local("", "Europe/Moscow") == "?"
        assert fmt_utc_to_local(None, "Europe/Moscow") == "?"  # type: ignore

    def test_invalid_timestamp_falls_back_safely(self):
        from mindsecretary.core import fmt_utc_to_local
        # Garbage → return slice of input, no crash
        out = fmt_utc_to_local("garbage", "Europe/Moscow")
        assert out == "garbage"

    def test_invalid_tz_falls_back_safely(self):
        from mindsecretary.core import fmt_utc_to_local
        # Bad TZ → ZoneInfoNotFoundError → return slice
        out = fmt_utc_to_local("2026-05-07 22:00:00", "Nonsense/TZ")
        assert out == "2026-05-07 22:00"


class TestPluralizeRu:
    """Russian plural helper. The 11-14 teens special case is the most
    common bug — without it 'год' becomes 'лет' too aggressively."""

    YEARS = ("год", "года", "лет")
    DAYS = ("день", "дня", "дней")

    def test_singular_one(self):
        from mindsecretary.core import pluralize_ru
        assert pluralize_ru(1, self.YEARS) == "год"
        assert pluralize_ru(1, self.DAYS) == "день"

    def test_few_two_three_four(self):
        from mindsecretary.core import pluralize_ru
        assert pluralize_ru(2, self.YEARS) == "года"
        assert pluralize_ru(3, self.YEARS) == "года"
        assert pluralize_ru(4, self.YEARS) == "года"

    def test_many_five_to_ten(self):
        from mindsecretary.core import pluralize_ru
        for n in range(5, 11):
            assert pluralize_ru(n, self.YEARS) == "лет", f"failed at {n}"

    def test_teens_use_many_form(self):
        """11-14 are the special case that breaks naive rules — they end
        in 1/2/3/4 but use the form_other ('лет', not 'год/года')."""
        from mindsecretary.core import pluralize_ru
        for n in range(11, 15):
            assert pluralize_ru(n, self.YEARS) == "лет", f"failed at {n}"

    def test_twenty_one_back_to_singular(self):
        """21, 31, 41, 101 — past the teens, last-digit rules resume."""
        from mindsecretary.core import pluralize_ru
        for n in (21, 31, 101, 1001):
            assert pluralize_ru(n, self.YEARS) == "год", f"failed at {n}"

    def test_twenty_two_to_twenty_four_use_few(self):
        from mindsecretary.core import pluralize_ru
        assert pluralize_ru(22, self.YEARS) == "года"
        assert pluralize_ru(23, self.YEARS) == "года"
        assert pluralize_ru(24, self.YEARS) == "года"
        # 25-30 → many
        assert pluralize_ru(25, self.YEARS) == "лет"
        assert pluralize_ru(30, self.YEARS) == "лет"

    def test_zero_uses_many(self):
        """0 should pick form_other ('0 лет', not '0 год'). Edge case but
        it shows up in 'не общались 0 дней' if the comparator allows it."""
        from mindsecretary.core import pluralize_ru
        assert pluralize_ru(0, self.YEARS) == "лет"

    def test_negative_n_treated_by_absolute(self):
        """Defensive: negative inputs use the absolute value's form. We
        don't expect negatives in production but the helper shouldn't
        ValueError if they slip through."""
        from mindsecretary.core import pluralize_ru
        assert pluralize_ru(-1, self.YEARS) == "год"
        assert pluralize_ru(-21, self.YEARS) == "год"


class TestFormatAge:
    """Russian plural rendering for the anniversary label."""

    def test_year_singular(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(365) == "1 год назад"

    def test_year_few(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(2 * 365) == "2 года назад"

    def test_year_many(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(5 * 365) == "5 лет назад"

    def test_month_singular(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(31) == "1 месяц назад"

    def test_month_few(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(2 * 30) == "2 месяца назад"

    def test_month_many(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(6 * 30) == "6 месяцев назад"

    def test_under_30_days_uses_days(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(15) == "15 дн. назад"

    def test_year_21_takes_singular_form(self):
        """The previous inline `< 5 → form_2_4 else form_other` rule said
        '21 лет' instead of '21 год'. Russian rule: 21, 31, 41 (ending in
        1, except teens 11-14) use the singular form."""
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(21 * 365) == "21 год назад"

    def test_year_22_takes_few_form(self):
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(22 * 365) == "22 года назад"

    def test_year_11_takes_many_form_via_teens_special(self):
        """11-14 are the teens special case: always use form_other regardless
        of last digit. So 11 years = '11 лет', not '11 год'."""
        from mindsecretary.proactive.briefing import BriefingGenerator
        assert BriefingGenerator._format_age(11 * 365) == "11 лет назад"


class TestMorningAnniversariesSlot:
    """Morning briefing pulls anniversaries from the DB into the prompt
    and renders them. Empty case must NOT add a section header."""

    @pytest.mark.asyncio
    async def test_anniversary_decision_renders_with_outcome(self, tmp_path):
        from unittest.mock import AsyncMock
        from datetime import datetime, timezone
        bg, db, llm = _make_briefing(tmp_path)
        bg.memory.search = AsyncMock(return_value=[])

        # Same MM-DD a year back — needed for substr match in
        # get_anniversaries. days=400 lands on different calendar date.
        now_utc = datetime.now(timezone.utc)
        past = now_utc.replace(year=now_utc.year - 1)
        past_ts = past.strftime("%Y-%m-%d %H:%M:%S")
        db.db.execute(
            "INSERT INTO decisions (id, description, outcome, "
            "outcome_sentiment, status, created_at) "
            "VALUES ('d1', 'сменить работу', 'отлично, наконец-то', "
            "'positive', 'resolved', ?)",
            (past_ts,),
        )
        db.db.commit()

        await bg.generate_morning()

        sys = llm.chat.call_args.kwargs["system"]
        # Anniversary section appears with both the decision and outcome
        assert "сменить работу" in sys
        assert "наконец-то" in sys
        # Format puts the age label before the content
        assert "назад" in sys

    @pytest.mark.asyncio
    async def test_anniversary_empty_renders_placeholder(self, tmp_path):
        from unittest.mock import AsyncMock
        bg, db, llm = _make_briefing(tmp_path)
        bg.memory.search = AsyncMock(return_value=[])

        await bg.generate_morning()
        sys = llm.chat.call_args.kwargs["system"]
        assert "Нет совпадений по дате" in sys


class TestMorningHabitsSlot:
    """Morning briefing surfaces active streaks for motivation framing
    ('don't break it today'). Mirrors v0.13.6 evening habits but the
    framing is forward-looking, not retrospective."""

    @pytest.mark.asyncio
    async def test_active_streak_surfaces_in_morning_prompt(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        # Inject memory mock so generate_morning's two memory.search calls
        # don't blow up on the MagicMock memory we built.
        from unittest.mock import AsyncMock
        bg.memory.search = AsyncMock(return_value=[])

        today = db._now()
        for i in range(5):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            db.log_habit("зарядка", done=True, date=d)

        await bg.generate_morning()

        system_prompt = llm.chat.call_args.kwargs["system"]
        assert "🔥 Серии" in system_prompt
        assert "зарядка" in system_prompt
        assert "5д" in system_prompt

    @pytest.mark.asyncio
    async def test_no_streak_renders_zero_state(self, tmp_path):
        """When there are habits logged but no streak ≥3, show 'all from
        scratch' framing — sub-3 streaks don't deserve a flex but the
        absence of any streak is still a useful signal for tone."""
        bg, db, llm = _make_briefing(tmp_path)
        from unittest.mock import AsyncMock
        bg.memory.search = AsyncMock(return_value=[])

        today_str = db._now().strftime("%Y-%m-%d")
        db.log_habit("чтение", done=True, date=today_str)  # 1d streak

        await bg.generate_morning()

        system_prompt = llm.chat.call_args.kwargs["system"]
        assert "🔥 Серии" not in system_prompt  # 1d hidden
        assert "Серий нет, всё с нуля" in system_prompt

    @pytest.mark.asyncio
    async def test_no_habits_renders_explicit_placeholder(self, tmp_path):
        bg, db, llm = _make_briefing(tmp_path)
        from unittest.mock import AsyncMock
        bg.memory.search = AsyncMock(return_value=[])

        await bg.generate_morning()

        system_prompt = llm.chat.call_args.kwargs["system"]
        assert "Привычки не отслеживаются" in system_prompt
