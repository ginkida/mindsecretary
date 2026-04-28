"""WeeklyReflection — interaction list shape sent to the LLM.

The weekly review used to feed the LLM the OLDEST 100 interactions out
of the DESC-sorted list, silently dropping the most recent ones.
Regression test locks the corrected ordering.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from mindsecretary.core.config import Profile
from mindsecretary.learning.reflection import WeeklyReflection


def _profile() -> Profile:
    return Profile(
        name="Test", city="Almaty", timezone="UTC",
        home_coords=[43.25, 76.9], work_coords=[43.25, 76.9],
        wake_up="07:00", work_start="09:00", work_end="18:00",
        sleep="23:00", commute_method="авто", commute_minutes=30,
        style="кратко", language="ru", notification_limit=10,
        quiet_hours=["23:00", "07:00"],
        priorities=[], dislikes=[],
    )


def _build():
    return WeeklyReflection(
        llm=MagicMock(), memory=MagicMock(), db=MagicMock(),
        profile=_profile(),
    )


def _make_interaction(ts: str, content: str) -> dict:
    return {
        "timestamp": ts, "direction": "in",
        "message_type": "text", "content": content,
    }


class TestFormatInteractions:
    def test_takes_newest_100_when_over_cap(self):
        """200 interactions in DESC order → newest 100 pass the cap.
        Pre-fix bug: lines[-100:] returned the OLDEST 100 (slice on a
        DESC-sorted list)."""
        wr = _build()
        # Simulate get_interactions output: DESC by timestamp.
        # newest_99...newest_0 = last 100 in DESC order; oldest_99...oldest_0
        # = first 100 of the (chronologically) earliest week.
        interactions = []
        for i in range(99, -1, -1):
            interactions.append(_make_interaction(
                f"2026-04-28 {12:02d}:{i:02d}:00", f"newest_{i}"
            ))
        for i in range(99, -1, -1):
            interactions.append(_make_interaction(
                f"2026-04-22 {10:02d}:{i:02d}:00", f"oldest_{i}"
            ))

        out = wr._format_interactions(interactions)
        # All 100 newest must appear
        for i in range(100):
            assert f"newest_{i}" in out
        # Oldest must NOT (cap)
        assert "oldest_0" not in out
        assert "oldest_50" not in out

    def test_chronological_order_oldest_first(self):
        """Once the latest 100 are picked, they should render in
        chronological (old → new) order — that's how a "log" reads."""
        wr = _build()
        # 3 interactions, DESC by timestamp
        interactions = [
            _make_interaction("2026-04-28 18:00:00", "newest"),
            _make_interaction("2026-04-28 12:00:00", "middle"),
            _make_interaction("2026-04-28 06:00:00", "oldest"),
        ]
        out = wr._format_interactions(interactions)
        oldest_pos = out.index("oldest")
        middle_pos = out.index("middle")
        newest_pos = out.index("newest")
        # Oldest first, newest last — chronological flow
        assert oldest_pos < middle_pos < newest_pos

    def test_under_cap_passes_through(self):
        """Below the 100-item cap, all interactions appear."""
        wr = _build()
        interactions = [
            _make_interaction(f"2026-04-28 12:{i:02d}:00", f"msg_{i}")
            for i in range(5, -1, -1)  # 6 items, DESC
        ]
        out = wr._format_interactions(interactions)
        for i in range(6):
            assert f"msg_{i}" in out

    def test_empty_returns_empty_string(self):
        assert _build()._format_interactions([]) == ""
