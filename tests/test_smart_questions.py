"""Cooldown comparisons in SmartQuestions across the v0.12.2 TZ cleanup.

Pre-v0.12.2 `_set_last_asked` wrote `datetime.now().isoformat()` — naive
system UTC in Docker. Post-v0.12.2 it writes a TZ-aware ISO in profile TZ.
The generator must treat elapsed time correctly for either shape without
double-counting the TZ offset.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from mindsecretary.core.config import Profile
from mindsecretary.proactive.smart_questions import SmartQuestions


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


def _make_sq(profile: Profile | None = None) -> SmartQuestions:
    llm = MagicMock()
    memory = MagicMock()
    db = MagicMock()
    db.get_preference.return_value = None
    sq = SmartQuestions(
        llm=llm, memory=memory, db=db,
        profile=profile or _profile(),
        min_interactions=5,
    )
    return sq


class TestCooldownNormalization:
    @pytest.mark.asyncio
    async def test_legacy_naive_pref_still_on_cooldown(self):
        """Pref written pre-v0.12.2 is naive UTC. Two hours ago < 8h cooldown
        → generator must return None, not re-ask."""
        sq = _make_sq()
        two_hours_ago_naive_utc = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
        )
        sq._get_last_asked = MagicMock(return_value=two_hours_ago_naive_utc)
        result = await sq.generate_question()
        assert result is None

    @pytest.mark.asyncio
    async def test_aware_profile_pref_still_on_cooldown(self):
        """Pref written post-v0.12.2 is aware profile-local. Two hours ago
        must also suppress — previously we compared naive local to naive UTC
        and drifted by the offset."""
        sq = _make_sq()
        two_hours_ago_aware = datetime.now(ZoneInfo("Asia/Almaty")) - timedelta(hours=2)
        sq._get_last_asked = MagicMock(return_value=two_hours_ago_aware)
        result = await sq.generate_question()
        assert result is None

    @pytest.mark.asyncio
    async def test_cooldown_expired_proceeds(self):
        """9h elapsed > 8h cooldown → cooldown check passes, interactions
        check is the next gate (mocked to fall short of min_interactions)."""
        sq = _make_sq()
        nine_hours_ago_aware = datetime.now(ZoneInfo("Asia/Almaty")) - timedelta(hours=9)
        sq._get_last_asked = MagicMock(return_value=nine_hours_ago_aware)
        sq.db.get_interactions.return_value = []  # < min_interactions
        # With insufficient interactions the method returns None, but only
        # AFTER getting past the cooldown gate. The marker that we got past
        # the gate is that get_interactions was called at all.
        result = await sq.generate_question()
        assert result is None
        sq.db.get_interactions.assert_called_once()

    def test_set_last_asked_writes_aware_iso(self):
        """New `_set_last_asked` writes an ISO string with a TZ offset,
        readable in the user's clock."""
        sq = _make_sq(_profile("Asia/Almaty"))
        sq._set_last_asked()
        call = sq.db.set_preference.call_args
        value = call.args[1]
        # Aware ISO string contains a "+HH:MM" offset suffix
        assert "+05:00" in value or "+0500" in value


class TestPastQuestionDedup:
    """Pre-v0.13.5 SmartQuestions only had a time-based 8h cooldown — the
    LLM didn't see what it had asked previously, so the same gap-question
    ("Кто такой Петя?") would re-fire every few days until the user
    answered. The fix is to inject recent smart_question outputs into the
    prompt so the LLM avoids repeats and rephrases."""

    def _enough_recent(self, n: int = 6) -> list[dict]:
        """Build `n` `direction='in'` interactions to clear min_interactions."""
        return [
            {"direction": "in", "content": f"msg {i}", "message_type": "text"}
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_past_questions_injected_into_prompt(self):
        from unittest.mock import AsyncMock

        sq = _make_sq()
        sq.db.get_preference.return_value = None
        # Cooldown check passes (no last-asked record); enough recent
        # interactions; smart_question history fetch returns 2 past Qs.
        sq.db.get_interactions.side_effect = [
            self._enough_recent(),  # first call: recent activity
            [
                {"content": "Кто такой Петя — друг или коллега?",
                 "message_type": "smart_question"},
                {"content": "Как прошёл визит к стоматологу?",
                 "message_type": "smart_question"},
            ],  # second call: past smart_questions
        ]
        sq.db.get_contacts.return_value = []
        sq.memory.search = AsyncMock(return_value=[])
        sq.llm.chat = AsyncMock(return_value=MagicMock(text="Что ты ел?"))

        result = await sq.generate_question()

        assert result == "🤔 Что ты ел?"
        system_prompt = sq.llm.chat.call_args.kwargs["system"]
        # Past questions must be quoted into the prompt
        assert "Кто такой Петя" in system_prompt
        assert "стоматологу" in system_prompt
        # And the second-call signature is the message_type filter
        second_call = sq.db.get_interactions.call_args_list[1]
        assert second_call.kwargs.get("message_type") == "smart_question"

    @pytest.mark.asyncio
    async def test_no_past_questions_renders_placeholder(self):
        from unittest.mock import AsyncMock

        sq = _make_sq()
        sq.db.get_preference.return_value = None
        sq.db.get_interactions.side_effect = [self._enough_recent(), []]
        sq.db.get_contacts.return_value = []
        sq.memory.search = AsyncMock(return_value=[])
        sq.llm.chat = AsyncMock(return_value=MagicMock(text="Q"))

        await sq.generate_question()

        system_prompt = sq.llm.chat.call_args.kwargs["system"]
        # Empty history shows placeholder text, not a stray empty bullet
        assert "Пока не задавал" in system_prompt

    @pytest.mark.asyncio
    async def test_history_fetch_failure_degrades_gracefully(self):
        """A DB error fetching past questions must NOT block the generator
        — it should fall through with empty history rather than skipping
        the daily question entirely."""
        from unittest.mock import AsyncMock

        sq = _make_sq()
        sq.db.get_preference.return_value = None

        def get_int(*args, **kwargs):
            if kwargs.get("message_type") == "smart_question":
                raise RuntimeError("DB exploded")
            return self._enough_recent()
        sq.db.get_interactions.side_effect = get_int
        sq.db.get_contacts.return_value = []
        sq.memory.search = AsyncMock(return_value=[])
        sq.llm.chat = AsyncMock(return_value=MagicMock(text="Q"))

        result = await sq.generate_question()

        assert result == "🤔 Q"  # generator survived
        system_prompt = sq.llm.chat.call_args.kwargs["system"]
        assert "Пока не задавал" in system_prompt

    @pytest.mark.asyncio
    async def test_history_fetch_uses_limit_constant(self):
        """The cap on how many past Qs we feed back must match what the
        prompt budget assumes — drift here would silently bloat context."""
        from mindsecretary.proactive.smart_questions import _PREVIOUS_QUESTIONS_LIMIT
        from unittest.mock import AsyncMock

        sq = _make_sq()
        sq.db.get_preference.return_value = None
        sq.db.get_interactions.side_effect = [self._enough_recent(), []]
        sq.db.get_contacts.return_value = []
        sq.memory.search = AsyncMock(return_value=[])
        sq.llm.chat = AsyncMock(return_value=MagicMock(text="Q"))

        await sq.generate_question()

        second_call = sq.db.get_interactions.call_args_list[1]
        assert second_call.kwargs.get("limit") == _PREVIOUS_QUESTIONS_LIMIT
