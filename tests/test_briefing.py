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
