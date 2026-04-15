"""Tests for datetime validation in tool argument sanitization."""
from __future__ import annotations

import pytest

from mindsecretary.llm.tools import _sanitize_args


class TestDatetimeValidation:
    def test_iso_format_normalized(self):
        args = _sanitize_args("create_event", {
            "title": "Meeting",
            "start_at": "2026-04-15T10:30",
        })
        # Should normalize to space-separated SQL format
        assert args["start_at"] == "2026-04-15 10:30:00"

    def test_space_format_preserved(self):
        args = _sanitize_args("create_event", {
            "title": "Lunch",
            "start_at": "2026-04-15 12:00:00",
        })
        assert args["start_at"] == "2026-04-15 12:00:00"

    def test_invalid_date_left_as_is(self):
        args = _sanitize_args("create_event", {
            "title": "X",
            "start_at": "tomorrow at noon",
        })
        assert args["start_at"] == "tomorrow at noon"

    def test_reminder_trigger_at_normalized(self):
        args = _sanitize_args("create_reminder", {
            "text": "Call",
            "trigger_at": "2026-04-15T18:00",
        })
        assert args["trigger_at"] == "2026-04-15 18:00:00"

    def test_end_at_optional_validated(self):
        args = _sanitize_args("create_event", {
            "title": "Long meeting",
            "start_at": "2026-04-15T10:00",
            "end_at": "2026-04-15T12:00",
        })
        assert args["end_at"] == "2026-04-15 12:00:00"

    def test_none_end_at_untouched(self):
        args = _sanitize_args("create_event", {
            "title": "X",
            "start_at": "2026-04-15T10:00",
            "end_at": None,
        })
        assert args["end_at"] is None

    def test_non_date_tools_unaffected(self):
        args = _sanitize_args("save_memory", {
            "content": "2026-04-15T10:00",
            "category": "work",
            "importance": 5,
        })
        # save_memory content should NOT be parsed as date
        assert args["content"] == "2026-04-15T10:00"


class TestMoodEnglish:
    """Test that English mood signals work."""

    def test_english_positive(self):
        from mindsecretary.learning.mood import analyze_mood
        messages = [{"direction": "in", "content": "That was awesome and amazing!"}]
        result = analyze_mood(messages)
        assert result["score"] > 0
        assert result["stats"]["positive_signals"] >= 2

    def test_english_negative(self):
        from mindsecretary.learning.mood import analyze_mood
        messages = [{"direction": "in", "content": "I'm tired and frustrated"}]
        result = analyze_mood(messages)
        assert result["score"] < 0
        assert result["stats"]["negative_signals"] >= 2

    def test_mixed_language(self):
        from mindsecretary.learning.mood import analyze_mood
        messages = [{"direction": "in", "content": "Отлично, great job, супер!"}]
        result = analyze_mood(messages)
        assert result["stats"]["positive_signals"] >= 3
