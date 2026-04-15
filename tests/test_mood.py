"""Tests for learning/mood.py — mood analysis and contact frequency."""
from __future__ import annotations

import pytest

from mindsecretary.learning.mood import analyze_mood


class TestAnalyzeMood:
    def test_empty_messages(self):
        result = analyze_mood([])
        assert result["score"] == 0.0
        assert result["label"] == "unknown"

    def test_no_user_messages(self):
        messages = [{"direction": "out", "content": "Hello"}]
        result = analyze_mood(messages)
        assert result["label"] == "unknown"

    def test_positive_signals(self):
        messages = [
            {"direction": "in", "content": "Отлично получилось! Круто!"},
        ]
        result = analyze_mood(messages)
        assert result["score"] > 0
        assert result["label"] == "positive"
        assert len(result["signals"]) > 0

    def test_negative_signals(self):
        messages = [
            {"direction": "in", "content": "Устал, всё плохо, бесит"},
        ]
        result = analyze_mood(messages)
        assert result["score"] < 0
        assert result["label"] == "negative"

    def test_neutral_no_signals(self):
        messages = [
            {"direction": "in", "content": "Сегодня обычный день на работе"},
        ]
        result = analyze_mood(messages)
        assert result["label"] == "neutral"

    def test_mixed_signals(self):
        messages = [
            {"direction": "in", "content": "Отлично поработал но устал"},
        ]
        result = analyze_mood(messages)
        # 1 positive + 1 negative → score near 0
        assert -0.5 <= result["score"] <= 0.5

    def test_short_terse_messages_slight_negative(self):
        """Multiple short messages with no signals → slightly negative (terse)."""
        messages = [
            {"direction": "in", "content": "ок"},
            {"direction": "in", "content": "да"},
            {"direction": "in", "content": "нет"},
        ]
        result = analyze_mood(messages)
        assert result["score"] < 0

    def test_score_clamped(self):
        # Even extreme input should stay in [-1, 1]
        messages = [
            {"direction": "in", "content": " ".join(
                ["ужас", "кошмар", "плохо", "бесит", "устал"] * 10
            )},
        ]
        result = analyze_mood(messages)
        assert -1.0 <= result["score"] <= 1.0

    def test_stats_populated(self):
        messages = [
            {"direction": "in", "content": "Отлично сработало"},
        ]
        result = analyze_mood(messages)
        stats = result["stats"]
        assert stats["messages"] == 1
        assert stats["avg_length"] > 0
        assert stats["positive_signals"] >= 1

    def test_none_content_handled(self):
        messages = [{"direction": "in", "content": None}]
        # Should not crash
        result = analyze_mood(messages)
        assert result["label"] == "neutral"
