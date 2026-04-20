"""Tests for llm/tools.py — argument sanitization and tool execution."""
from __future__ import annotations

import pytest

from mindsecretary.llm.tools import (
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    _sanitize_args,
    _truncate,
)


class TestTruncate:
    def test_none_returns_none(self):
        assert _truncate(None) is None

    def test_short_string_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_long_string_truncated(self):
        result = _truncate("x" * 10000, max_len=100)
        assert len(result) == 100


class TestSanitizeArgs:
    def test_basic_string_passthrough(self):
        args = _sanitize_args("create_event", {"title": "Lunch", "start_at": "2026-04-15T12:00"})
        assert args["title"] == "Lunch"

    def test_int_and_bool_passthrough(self):
        args = _sanitize_args("log_habit", {"habit_name": "gym", "done": True})
        assert args["done"] is True

    def test_none_passthrough(self):
        args = _sanitize_args("create_event", {"title": "X", "location": None})
        assert args["location"] is None

    def test_non_primitive_coerced_to_string(self):
        args = _sanitize_args("test", {"data": ["a", "b"]})
        assert isinstance(args["data"], str)

    def test_save_memory_invalid_category_defaults(self):
        args = _sanitize_args("save_memory", {
            "content": "test",
            "category": "INVALID",
            "importance": 5,
        })
        assert args["category"] == "personal"

    def test_save_memory_valid_categories(self):
        for cat in VALID_CATEGORIES:
            args = _sanitize_args("save_memory", {
                "content": "test", "category": cat, "importance": 5,
            })
            assert args["category"] == cat

    def test_save_memory_importance_clamped(self):
        args = _sanitize_args("save_memory", {
            "content": "x", "category": "work", "importance": 99,
        })
        assert args["importance"] == 10

        args = _sanitize_args("save_memory", {
            "content": "x", "category": "work", "importance": -5,
        })
        assert args["importance"] == 1

    def test_event_invalid_priority_defaults(self):
        args = _sanitize_args("create_event", {
            "title": "X", "start_at": "now", "priority": "URGENT",
        })
        assert args["priority"] == "medium"

    def test_event_valid_priorities(self):
        for prio in VALID_PRIORITIES:
            args = _sanitize_args("create_event", {
                "title": "X", "start_at": "now", "priority": prio,
            })
            assert args["priority"] == prio

    def test_long_strings_truncated(self):
        args = _sanitize_args("save_memory", {
            "content": "x" * 10000,
            "category": "work",
            "importance": 5,
        })
        assert len(args["content"]) == 5000

    def test_get_recent_memories_limit_clamped(self):
        args = _sanitize_args("get_recent_memories", {"limit": 50})
        assert args["limit"] == 10

    def test_get_open_loops_days_clamped(self):
        args = _sanitize_args("get_open_loops", {"days_ahead": 99})
        assert args["days_ahead"] == 7
