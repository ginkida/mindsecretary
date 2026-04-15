"""Tests for interfaces/telegram.py — utility functions."""
from __future__ import annotations

from mindsecretary.interfaces.telegram import _fix_markdown, _split_message


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
