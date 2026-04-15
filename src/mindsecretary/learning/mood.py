from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from ..core.database import Database
from ..core.enums import MoodLabel

logger = logging.getLogger(__name__)

# Simple Russian sentiment signals
POSITIVE_SIGNALS = {
    "отлично", "круто", "здорово", "класс", "супер", "ура", "кайф", "рад",
    "получилось", "успел", "доволен", "нравится", "люблю", "счастлив", "хорошо",
    "прекрасно", "замечательно", "удачно", "наконец-то",
}
NEGATIVE_SIGNALS = {
    "устал", "плохо", "ужас", "кошмар", "злюсь", "бесит", "надоело", "тяжело",
    "болит", "стресс", "проблема", "сломал", "забыл", "опоздал", "провал",
    "жалею", "грустно", "тоска", "скучно", "раздражает", "невыносимо",
}
NEUTRAL_THRESHOLD = 0.1


def analyze_mood(messages: list[dict]) -> dict:
    """Analyze mood from a list of user messages (direction='in').

    Returns: {"score": -1.0..1.0, "label": str, "signals": list, "stats": dict}
    """
    if not messages:
        return {"score": 0.0, "label": MoodLabel.UNKNOWN, "signals": [], "stats": {}}

    user_msgs = [m for m in messages if m.get("direction") == "in"]
    if not user_msgs:
        return {"score": 0.0, "label": MoodLabel.UNKNOWN, "signals": [], "stats": {}}

    total_len = 0
    total_words = 0
    positive_count = 0
    negative_count = 0
    signals_found = []

    for m in user_msgs:
        content = (m.get("content") or "").lower()
        words = content.split()
        total_len += len(content)
        total_words += len(words)

        for w in words:
            clean = re.sub(r"[^\w]", "", w)
            if clean in POSITIVE_SIGNALS:
                positive_count += 1
                signals_found.append(f"+{clean}")
            elif clean in NEGATIVE_SIGNALS:
                negative_count += 1
                signals_found.append(f"-{clean}")

    msg_count = len(user_msgs)
    avg_len = total_len / msg_count if msg_count else 0

    # Score: -1.0 (very negative) to 1.0 (very positive)
    total_signals = positive_count + negative_count
    if total_signals > 0:
        score = (positive_count - negative_count) / total_signals
    else:
        score = 0.0

    # Short messages with no signals = possibly tired/terse
    if avg_len < 20 and total_signals == 0 and msg_count >= 3:
        score -= 0.2

    score = max(-1.0, min(1.0, score))

    if score > 0.3:
        label = MoodLabel.POSITIVE
    elif score < -0.3:
        label = MoodLabel.NEGATIVE
    else:
        label = MoodLabel.NEUTRAL

    return {
        "score": round(score, 2),
        "label": label,
        "signals": signals_found[:10],
        "stats": {
            "messages": msg_count,
            "avg_length": round(avg_len),
            "positive_signals": positive_count,
            "negative_signals": negative_count,
        },
    }


def get_mood_trend(db: Database, days: int = 7) -> list[dict]:
    """Get daily mood for the last N days."""
    results = []
    now = datetime.now()
    for i in range(days):
        day = now - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day.replace(hour=23, minute=59, second=59, microsecond=0)
        # get_interactions handles the correct SQL timestamp format internally
        day_msgs = db.get_interactions(
            since=day_start, until=day_end, limit=200,
        )
        if day_msgs:
            mood = analyze_mood(day_msgs)
            results.append({
                "date": day.strftime("%Y-%m-%d"),
                "score": mood["score"],
                "label": mood["label"],
                "messages": mood["stats"]["messages"],
            })
    return list(reversed(results))


def check_contact_frequency(db: Database) -> list[dict]:
    """Find contacts whose mention frequency dropped significantly."""
    contacts = db.get_contacts("")
    alerts = []

    for c in contacts:
        if not c.get("last_contact"):
            continue
        try:
            last = datetime.fromisoformat(c["last_contact"])
        except (ValueError, TypeError):
            continue

        days_since = (datetime.now() - last).days
        freq = c.get("contact_frequency")
        mention_count = c.get("mention_count", 0)

        # If we have enough data and contact has been quiet
        if mention_count >= 3 and days_since > 30:
            alerts.append({
                "name": c["name"],
                "relation": c.get("relation", ""),
                "days_since": days_since,
                "mention_count": mention_count,
            })
        elif freq and days_since > freq * 2:
            alerts.append({
                "name": c["name"],
                "relation": c.get("relation", ""),
                "days_since": days_since,
                "expected_frequency": freq,
            })

    # Sort by days_since descending
    alerts.sort(key=lambda x: x["days_since"], reverse=True)
    return alerts[:5]
