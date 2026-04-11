from __future__ import annotations

import logging

from ..core.database import Database

logger = logging.getLogger(__name__)


async def check_reminders(db: Database, send_fn) -> int:
    """Check and send due reminders. Returns count sent.

    User reminders bypass quiet hours and notification limits — they're
    explicit user intent, not proactive notifications.
    """
    due = db.get_due_reminders()
    sent = 0
    for r in due:
        text = f"⏰ Напоминание: {r['text']}"
        if r["priority"] == "high":
            text = f"🔴 {text}"
        try:
            await send_fn(text)
            db.mark_reminder_sent(r["id"])
            sent += 1
        except Exception as e:
            logger.error("Failed to send reminder %s: %s", r["id"], type(e).__name__)
    return sent
