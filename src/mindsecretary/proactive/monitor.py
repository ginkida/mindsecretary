from __future__ import annotations

import logging
from datetime import datetime

from ..core.database import Database

logger = logging.getLogger(__name__)


async def check_reminders(db: Database, send_fn) -> int:
    """Check and send due reminders. Returns count sent.

    User reminders bypass quiet hours and notification limits — they're
    explicit user intent, not proactive notifications. Each fire is
    logged as a notification interaction so the Brain's recent-messages
    context shows a chronological record of what the bot sent.
    """
    due = db.get_due_reminders()
    sent = 0
    for r in due:
        text = f"⏰ Напоминание: {r['text']}"
        if r["priority"] == "high":
            text = f"🔴 {text}"
        try:
            await send_fn(text)
            # Log to interactions BEFORE marking the reminder sent. If the
            # process crashes between the two writes, a not-logged reminder
            # in 'sent' state would vanish from history entirely. Extra log
            # on a pending reminder is recoverable; missing one isn't.
            db.log_interaction(
                direction="out",
                message_type="notification",
                content=text[:500],
                metadata={"kind": "reminder", "reminder_id": r["id"]},
            )
            db.mark_reminder_sent(r["id"])
            sent += 1
        except Exception as e:
            logger.error("Failed to send reminder %s: %s", r["id"], type(e).__name__)
    return sent


def _format_event_alert(event: dict, now_local: datetime) -> str:
    """Render the pre-event alert text. Includes minutes-until, time, title,
    location and related person when available. Tolerant of missing fields
    — only the title and start_at are guaranteed by the events schema."""
    title = (event.get("title") or "событие").strip()
    start_raw = event.get("start_at") or ""
    # Time portion — start_at is "YYYY-MM-DD HH:MM:SS" local naive, so the
    # last 5 chars before the seconds give us "HH:MM" cleanly.
    time_str = start_raw[11:16] if len(start_raw) >= 16 else "?"
    try:
        start_dt = datetime.fromisoformat(start_raw.replace(" ", "T"))
        delta = start_dt - now_local
        minutes_until = max(1, int(round(delta.total_seconds() / 60)))
    except (ValueError, TypeError):
        minutes_until = None

    parts = [f"🔔 Через ~{minutes_until} мин: {title} в {time_str}"] if minutes_until else [
        f"🔔 Скоро: {title} в {time_str}"
    ]

    location = (event.get("location") or "").strip()
    if location:
        parts.append(f"📍 {location}")
    person = (event.get("related_person") or "").strip()
    # Redundancy check: skip the person line if a 3-char stem of the name
    # already appears in the title. Russian declensions ("Маша" → "Машей"
    # in instrumental case) defeat a literal substring match, but a short
    # stem catches the common "встреча с Машей" + related_person="Маша"
    # case without false-positives on unrelated short stems.
    person_stem = person.lower()[:3]
    if person and (not person_stem or person_stem not in title.lower()):
        parts.append(f"👤 {person}")
    return "\n".join(parts)


async def check_event_alerts(db: Database, send_fn, lead_minutes: int) -> int:
    """Pre-event alert: fire once per event when it's within `lead_minutes`
    of starting. Returns count alerted.

    Bypasses quiet hours / notification limits — same rationale as
    reminders: an imminent calendar event is high-signal user intent,
    not a discretionary proactive ping. Mark BEFORE send so a transient
    Telegram failure doesn't leave the same event eligible to fire again
    on the next 5-min tick (which would spam the user once Telegram
    recovered)."""
    if lead_minutes <= 0:
        return 0
    pending = db.get_events_to_alert(lead_minutes=lead_minutes)
    sent = 0
    now_local = db.local_now_naive()
    for ev in pending:
        text = _format_event_alert(ev, now_local)
        # Mark first, then send — at-most-once delivery, intentional. See
        # docstring above.
        try:
            db.mark_event_alerted(ev["id"])
            await send_fn(text)
            db.log_interaction(
                direction="out",
                message_type="notification",
                content=text[:500],
                metadata={"kind": "event_alert", "event_id": ev["id"]},
            )
            sent += 1
        except Exception as e:
            logger.error(
                "Failed to send event alert for %s: %s",
                ev.get("id"), type(e).__name__,
            )
    return sent
