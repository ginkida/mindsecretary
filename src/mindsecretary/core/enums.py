"""Enum constants for status, priority, and sentiment fields.

Uses (str, Enum) mixin for Python 3.10 compat (StrEnum is 3.11+).
Values compare equal to plain strings: Status.PENDING == "pending".
"""
from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    """Base for string enums compatible with Python 3.10+."""
    pass


class Status(_StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    PARTIAL = "partial"
    RESOLVED = "resolved"
    SENT = "sent"
    ACTIVE = "active"
    DELETED = "deleted"


class Priority(_StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Sentiment(_StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Feedback(_StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class MoodLabel(_StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"
