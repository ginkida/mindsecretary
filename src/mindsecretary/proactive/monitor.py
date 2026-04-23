from __future__ import annotations

import logging

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
