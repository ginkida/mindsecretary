from __future__ import annotations

from datetime import datetime, timezone as _tz
from zoneinfo import ZoneInfo

DAYS_RU = {
    0: "Понедельник", 1: "Вторник", 2: "Среда", 3: "Четверг",
    4: "Пятница", 5: "Суббота", 6: "Воскресенье",
}

# Single source of truth for notification kind → short Russian label.
# Used by:
#   - brain._build_history_turns: assistant turn prefix in history replay
#   - tools._handle_search_conversations: row label in tool output
#   - briefing._interaction_label: evening summary interactions_text
# Pre-consolidation each callsite kept its own copy with "keep in sync"
# comments — drift-prone (iter 8 caught a missing event_alert/reflection
# pair). Adding a new kind = add ONE entry here.
NOTIFICATION_KIND_LABELS: dict[str, str] = {
    "morning_briefing": "брифинг",
    "evening_summary": "вечер",
    "diary": "дневник",
    "weekly_review": "неделя",
    "smart_question": "вопрос",
    "open_loops_nudge": "контроль",
    "decision_followup": "решение",
    "birthday_alert": "день рождения",
    "weather_alert": "погода",
    "reminder": "напоминание",
    "event_alert": "событие скоро",
    "event_reflection": "как прошло",
}


def is_person_in_title(person: str | None, title: str | None) -> bool:
    """Heuristic: does `person`'s name stem already appear in `title`?

    Russian declensions defeat a plain substring match — title="ужин с
    Машей" doesn't contain "Маша" literally. We normalize both sides to
    lower-case and compare the first 3 characters of the person against
    the title. That's the longest invariant prefix across common
    declensions (Маша/Маше/Машей all share "Маш"; Олег/Олега/Олегом
    share "Оле").

    Used by event renderers to suppress a redundant "👤 Маша" line when
    the title already names the person — without it the alert reads
    "встреча с Машей / 👤 Маша" which looks like a render bug.

    Returns False on missing inputs (defensive — caller skips the
    suppression). Falls back to True only when the stem actually
    matches; ambiguous cases lean toward keeping the line, since
    rendering an extra hint is cheaper than hiding a real one.
    """
    if not person or not title:
        return False
    stem = person.lower()[:3]
    if not stem:
        return False
    return stem in title.lower()


def pluralize_ru(n: int, forms: tuple[str, str, str]) -> str:
    """Russian plural form for `n` given (form_1, form_2_4, form_other).

    Rule:
        - N ending in 1 (except teens 11-14) → form_1: "год", "месяц"
        - N ending in 2/3/4 (except teens) → form_2_4: "года", "месяца"
        - everything else → form_other: "лет", "месяцев"

    Examples:
        pluralize_ru(1, ("год", "года", "лет"))   → "год"
        pluralize_ru(2, ("год", "года", "лет"))   → "года"
        pluralize_ru(5, ("год", "года", "лет"))   → "лет"
        pluralize_ru(11, ("год", "года", "лет"))  → "лет"  (teens special)
        pluralize_ru(21, ("год", "года", "лет"))  → "год"  (>= 20, ends in 1)
        pluralize_ru(22, ("год", "года", "лет"))  → "года"

    The previous inline rule (`< 5 → form_2_4 else form_other`) broke at
    21 years — said "21 лет" instead of "21 год". Generalized helper so
    every duration render path inherits the correction.
    """
    last_two = abs(n) % 100
    last = abs(n) % 10
    if 11 <= last_two <= 14:
        return forms[2]
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


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
