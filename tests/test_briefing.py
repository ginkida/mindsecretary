"""BriefingGenerator output — verify habit progress lands in the
evening summary system prompt so Claude can talk about streaks
instead of inventing them.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
