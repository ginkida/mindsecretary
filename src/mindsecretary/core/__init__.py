from __future__ import annotations

from datetime import datetime, timezone as _tz
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


def fmt_local_time(ts: str, profile_tz: str, today_local: str | None = None) -> str:
    """Render a UTC-naive SQL timestamp (`YYYY-MM-DD HH:MM:SS`) as local HH:MM.

    If `today_local` is provided and the row falls on that local date, only
    HH:MM is returned; otherwise MM-DD HH:MM so the LLM can still order
    entries across day boundaries. Returns "??:??" on parse failure — this
    is user-facing metadata, never a correctness-critical value.

    Shared helper so `Brain._fmt_local_time`, the briefing, and the weekly
    review don't each roll their own conversion and silently drift apart.
    """
    try:
        utc_naive = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        # ZoneInfo raises ZoneInfoNotFoundError (a KeyError subclass) on
        # unknown TZ names — catch alongside the parse errors so a corrupt
        # profile.timezone doesn't blow up the briefing.
        local = utc_naive.replace(tzinfo=_tz.utc).astimezone(ZoneInfo(profile_tz))
    except (ValueError, TypeError, KeyError):
        return "??:??"
    if today_local and local.strftime("%Y-%m-%d") == today_local:
        return local.strftime("%H:%M")
    return local.strftime("%m-%d %H:%M")
