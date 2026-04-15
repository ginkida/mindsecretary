from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

DAYS_RU = {
    0: "Понедельник", 1: "Вторник", 2: "Среда", 3: "Четверг",
    4: "Пятница", 5: "Суббота", 6: "Воскресенье",
}


def tz_now(timezone: str | None = None) -> datetime:
    """Return current datetime in the given timezone.

    If timezone is None, falls back to system local time (naive datetime).
    When a timezone is provided, returns an aware datetime — strftime still
    produces the local time string we need for SQLite.
    """
    if timezone:
        return datetime.now(ZoneInfo(timezone))
    return datetime.now()
