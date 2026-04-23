"""Tests for proactive/monitor.py — reminder delivery + interaction log."""
from __future__ import annotations

import json

import pytest

from mindsecretary.proactive.monitor import check_reminders


@pytest.mark.asyncio
async def test_check_reminders_logs_interaction_on_fire(tmp_db):
    """When a reminder fires it must be logged as a notification
    interaction. Otherwise the Brain's chronological context misses
    'at 11:00 I reminded you about X' and Claude re-asks or repeats."""
    # Seed a due reminder (trigger_at in the past)
    tmp_db.create_reminder("позвонить маме", "2000-01-01 10:00:00")

    sent_messages: list[str] = []

    async def fake_send(text: str):
        sent_messages.append(text)

    count = await check_reminders(tmp_db, fake_send)

    assert count == 1
    assert len(sent_messages) == 1
    assert "позвонить маме" in sent_messages[0]

    # Interaction log must contain exactly one notification with kind=reminder
    rows = tmp_db.get_interactions(message_type="notification", limit=10)
    assert len(rows) == 1
    assert rows[0]["direction"] == "out"
    assert "позвонить маме" in rows[0]["content"]
    meta = json.loads(rows[0]["metadata"])
    assert meta["kind"] == "reminder"
    assert meta["reminder_id"]  # non-empty — links back to source


@pytest.mark.asyncio
async def test_check_reminders_skips_log_when_send_fails(tmp_db):
    """If Telegram send fails the reminder must NOT be marked sent and
    NOT appear in the interaction log — otherwise next tick drops it
    silently and the user never gets the reminder."""
    tmp_db.create_reminder("тест", "2000-01-01 10:00:00")

    async def failing_send(text: str):
        raise RuntimeError("telegram down")

    count = await check_reminders(tmp_db, failing_send)

    assert count == 0
    rows = tmp_db.get_interactions(message_type="notification", limit=10)
    assert rows == []
    # Reminder should still be pending for retry
    pending = tmp_db.get_pending_reminders()
    assert len(pending) == 1
