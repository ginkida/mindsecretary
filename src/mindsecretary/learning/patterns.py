"""Weekly pattern analyzer — stat-based observations for the reflection prompt.

Complements the LLM's qualitative review with hard facts it can reference:
which day of the week was busiest, whether mood is trending up or down,
which habits broke after a streak, how this week's spend compares to last,
and so on.

Detectors are deliberately simple and deterministic. Thresholds below the
`MIN_SIGNAL_COUNT` floor return `None` so we don't surface noise on weeks
with thin data. The `Pattern` dict carries a `label`, a short `detail`
line, and a `strength` score (0-1) so the prompt can emphasize the ones
the LLM should lead with.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..core import DAYS_RU, pluralize_ru
from ..core.config import Profile
from ..core.database import Database
from .mood import analyze_mood, check_contact_frequency

logger = logging.getLogger(__name__)

MIN_SIGNAL_COUNT = 5  # below this row count, most detectors abstain


@dataclass
class Pattern:
    label: str
    detail: str
    strength: float  # 0.0 - 1.0

    def to_line(self) -> str:
        return f"- {self.label}: {self.detail}"


def _utc_bounds_for_local_week(db: Database, weeks_back: int = 0) -> tuple[str, str]:
    """Return (start_utc, end_utc) SQL strings spanning a 7-day local window
    ending at the current local instant. `weeks_back=1` shifts to the prior
    week for week-over-week comparisons.
    """
    start_utc_s, _ = db._local_day_utc_bounds(day_offset=-7 - 7 * weeks_back)
    end_utc_s, _ = db._local_day_utc_bounds(day_offset=-7 * weeks_back)
    return start_utc_s, end_utc_s


def _peak_weekday(db: Database, profile: Profile) -> Pattern | None:
    """Day of the week with the most inbound messages. Returns None on
    weeks where the peak has fewer than 3 messages or dominance is weak."""
    offset = db._local_tz_offset_minutes()
    sign = "+" if offset >= 0 else "-"
    start_utc, end_utc = _utc_bounds_for_local_week(db)
    # strftime('%w', ..., '+N minutes') returns 0-6 with Sunday=0 per SQLite.
    rows = db.db.execute(
        f"SELECT strftime('%w', timestamp, '{sign}{abs(offset)} minutes') as wday, "
        f"COUNT(*) as cnt "
        f"FROM interactions "
        f"WHERE direction = 'in' AND timestamp >= ? AND timestamp < ? "
        f"GROUP BY wday ORDER BY cnt DESC",
        (start_utc, end_utc),
    ).fetchall()
    if not rows:
        return None
    top = rows[0]
    total = sum(r["cnt"] for r in rows)
    if top["cnt"] < 3 or total < MIN_SIGNAL_COUNT:
        return None
    share = top["cnt"] / total
    if share < 0.3:  # too evenly spread to call a "peak"
        return None
    # SQLite's %w: Sunday=0. DAYS_RU keys are ISO weekday (Monday=0), so map.
    sqlite_to_iso = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    wday_idx = int(top["wday"])
    day_name = DAYS_RU.get(sqlite_to_iso[wday_idx], "?")
    return Pattern(
        label="Пиковый день недели",
        detail=f"{day_name} — {top['cnt']} сообщений ({int(share * 100)}% недели)",
        strength=min(1.0, share),
    )


def _late_night_share(db: Database, profile: Profile) -> Pattern | None:
    """Share of messages logged between 22:00 and 06:00 local. Surfaces
    late-working / sleep-deprivation patterns. Silent if week had <5 msgs."""
    offset = db._local_tz_offset_minutes()
    sign = "+" if offset >= 0 else "-"
    start_utc, end_utc = _utc_bounds_for_local_week(db)
    rows = db.db.execute(
        f"SELECT strftime('%H', timestamp, '{sign}{abs(offset)} minutes') as hour "
        f"FROM interactions "
        f"WHERE direction = 'in' AND timestamp >= ? AND timestamp < ?",
        (start_utc, end_utc),
    ).fetchall()
    if len(rows) < MIN_SIGNAL_COUNT:
        return None
    late_count = sum(1 for r in rows if r["hour"] and (int(r["hour"]) >= 22 or int(r["hour"]) < 6))
    share = late_count / len(rows)
    if share < 0.15:
        return None
    return Pattern(
        label="Поздняя активность",
        detail=f"{int(share * 100)}% сообщений после 22:00 или до 06:00 ({late_count}/{len(rows)})",
        strength=min(1.0, share * 2),
    )


def _mood_direction(db: Database, profile: Profile) -> Pattern | None:
    """Compare first half vs second half of the week's mood scores.

    We don't care about absolute mood — the LLM can do that qualitatively.
    We surface the delta so the review can say "настроение росло/падало"
    rather than guessing.
    """
    start_utc_s, end_utc_s = _utc_bounds_for_local_week(db)
    start_dt = datetime.strptime(start_utc_s, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_utc_s, "%Y-%m-%d %H:%M:%S")
    midpoint = start_dt + (end_dt - start_dt) / 2
    first_half = db.get_interactions(since=start_dt, until=midpoint, limit=500)
    second_half = db.get_interactions(since=midpoint, until=end_dt, limit=500)
    first_user = [m for m in first_half if m.get("direction") == "in"]
    second_user = [m for m in second_half if m.get("direction") == "in"]
    if len(first_user) < 3 or len(second_user) < 3:
        return None
    first_mood = analyze_mood(first_user)["score"]
    second_mood = analyze_mood(second_user)["score"]
    delta = second_mood - first_mood
    if abs(delta) < 0.2:
        return None
    if delta > 0:
        detail = f"настроение улучшилось к концу недели ({first_mood:+.2f} → {second_mood:+.2f})"
    else:
        detail = f"настроение ухудшилось к концу недели ({first_mood:+.2f} → {second_mood:+.2f})"
    return Pattern(
        label="Динамика настроения",
        detail=detail,
        strength=min(1.0, abs(delta) / 1.5),
    )


def _habit_breaks(db: Database, profile: Profile) -> list[Pattern]:
    """Habits tracked this week that were missed more often than kept.

    `get_habit_stats` can't distinguish "not tracked" from "tracked and
    missed" because it exposes only `week_done` (done=1 count) out of 7.
    Query `habit_log` directly to get total log entries per habit this
    local week; only habits with ≥3 log entries and ≤40% done rate
    count as "at risk".
    """
    # habit_log.date is a local-date string (written via _now().strftime),
    # so compare against local dates.
    week_start = (db.local_now_naive() - timedelta(days=6)).strftime("%Y-%m-%d")
    rows = db.db.execute(
        "SELECT h.name, "
        "  COUNT(hl.date) as total_logs, "
        "  COALESCE(SUM(hl.done), 0) as done "
        "FROM habits h LEFT JOIN habit_log hl "
        "  ON h.id = hl.habit_id AND hl.date >= ? "
        "GROUP BY h.id",
        (week_start,),
    ).fetchall()
    patterns: list[Pattern] = []
    for r in rows:
        total = r["total_logs"]
        done = r["done"]
        if total < 3:
            continue  # not actively tracked this week
        rate = round(done / total * 100)
        if rate >= 40:
            continue  # kept more often than missed
        # Two day-counts on this line — pluralize against `total` since
        # that's what "дней" agrees with grammatically ("3 дня" vs
        # "5 дней"). `done` is "X out of Y" so reads cleanly without
        # a separate suffix.
        word = pluralize_ru(total, ("день", "дня", "дней"))
        patterns.append(Pattern(
            label="Привычка под риском",
            detail=f"{r['name']}: {done}/{total} {word} ({rate}%)",
            strength=1.0 - rate / 100,
        ))
    return sorted(patterns, key=lambda p: p.strength, reverse=True)[:2]


def _cost_wow(db: Database, profile: Profile) -> Pattern | None:
    """Week-over-week API cost delta. Surfaces only when the swing is
    meaningful (> 50% and absolute difference > $0.20)."""
    this_start, this_end = _utc_bounds_for_local_week(db, weeks_back=0)
    last_start, last_end = _utc_bounds_for_local_week(db, weeks_back=1)

    def _sum(start: str, end: str) -> float:
        row = db.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM api_costs "
            "WHERE timestamp >= ? AND timestamp < ?",
            (start, end),
        ).fetchone()
        return float(row[0])

    this_week = _sum(this_start, this_end)
    last_week = _sum(last_start, last_end)
    if last_week < 0.05:  # no baseline to compare against
        return None
    delta = this_week - last_week
    if abs(delta) < 0.20:
        return None
    pct = abs(delta) / last_week
    if pct < 0.5:
        return None
    direction = "вырос" if delta > 0 else "упал"
    return Pattern(
        label="Расход на API",
        detail=f"{direction} на {int(pct * 100)}% нед-к-нед: ${this_week:.2f} vs ${last_week:.2f}",
        strength=min(1.0, pct),
    )


def _goal_completion(db: Database, profile: Profile) -> Pattern | None:
    """Daily goal completion rate across this local week.

    Shows low completion as a gentle nudge, or high completion as positive
    reinforcement. Silent when too few goals were set to draw a conclusion.
    """
    now_local = db.local_now_naive()
    rows: list[dict[str, Any]] = []
    for i in range(7):
        day = (now_local - timedelta(days=i)).strftime("%Y-%m-%d")
        rows.extend(db.get_daily_goals(date=day))
    if len(rows) < 3:
        return None
    completed = sum(1 for r in rows if r.get("status") == "completed")
    rate = completed / len(rows)
    if 0.3 <= rate <= 0.7:
        return None  # unremarkable
    direction = "высокая" if rate > 0.7 else "низкая"
    return Pattern(
        label="Выполнение целей",
        detail=f"{direction} — {completed}/{len(rows)} закрыто ({int(rate * 100)}%)",
        strength=abs(rate - 0.5) * 2,
    )


def _quiet_contacts(db: Database, profile: Profile) -> Pattern | None:
    """One-line summary of contacts the user hasn't reached out to, leveraging
    the existing frequency checker. Surfaces only the most overdue one so
    the review doesn't turn into a shame list."""
    alerts = check_contact_frequency(db)
    if not alerts:
        return None
    first = alerts[0]
    days = first.get("days_since", 0)
    if days < 14:
        return None
    word = pluralize_ru(days, ("день", "дня", "дней"))
    return Pattern(
        label="Тихий контакт",
        detail=f"{first['name']}: {days} {word} без контакта",
        strength=min(1.0, days / 60),
    )


class PatternAnalyzer:
    """Runs all detectors and returns the significant patterns for a week."""

    def __init__(self, db: Database, profile: Profile):
        self.db = db
        self.profile = profile

    def analyze_week(self) -> list[Pattern]:
        """Return non-empty patterns sorted by strength (strongest first).

        Each detector can return 0, 1, or a small list of patterns; we
        flatten, filter, and sort. Caller decides how many to surface.
        """
        patterns: list[Pattern] = []
        detectors = [
            _peak_weekday,
            _late_night_share,
            _mood_direction,
            _cost_wow,
            _goal_completion,
            _quiet_contacts,
        ]
        for fn in detectors:
            try:
                result = fn(self.db, self.profile)
            except Exception as e:
                logger.warning("Pattern detector %s failed: %s", fn.__name__, type(e).__name__)
                continue
            if result is None:
                continue
            patterns.append(result)

        try:
            patterns.extend(_habit_breaks(self.db, self.profile))
        except Exception as e:
            logger.warning("Habit-break detector failed: %s", type(e).__name__)

        return sorted(patterns, key=lambda p: p.strength, reverse=True)

    def format_for_prompt(self, limit: int = 5) -> str:
        """Render top `limit` patterns as a bulleted list for the LLM."""
        found = self.analyze_week()[:limit]
        if not found:
            return "Недостаточно данных для автоматических наблюдений."
        return "\n".join(p.to_line() for p in found)
