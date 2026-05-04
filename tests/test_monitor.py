"""Tests for proactive/monitor.py — reminder delivery + interaction log."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from mindsecretary.proactive.monitor import (
    _format_event_alert,
    check_event_alerts,
    check_reminders,
)


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


# --- Pre-event alerts ---


@pytest.mark.asyncio
async def test_event_alert_fires_within_lead_window(tmp_db):
    """Event 10 min from now with lead=15 → must alert. Event marked
    alerted_at, interaction logged with kind=event_alert."""
    now = tmp_db.local_now_naive()
    in_10 = (now + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
    tmp_db.create_event("ужин с Машей", in_10, location="кафе")

    sent: list[str] = []

    async def fake_send(text: str):
        sent.append(text)

    count = await check_event_alerts(tmp_db, fake_send, lead_minutes=15)

    assert count == 1
    assert "ужин с Машей" in sent[0]
    assert "кафе" in sent[0]
    # alerted_at populated → won't re-fire
    rows = tmp_db.db.execute(
        "SELECT alerted_at FROM events WHERE title = 'ужин с Машей'"
    ).fetchone()
    assert rows["alerted_at"] is not None
    # Interaction logged with the event_alert kind so /search & history
    # know what was sent (matches reminder logging convention).
    inter = tmp_db.get_interactions(message_type="notification", limit=10)
    assert len(inter) == 1
    meta = json.loads(inter[0]["metadata"])
    assert meta["kind"] == "event_alert"
    assert meta["event_id"]


@pytest.mark.asyncio
async def test_event_alert_skips_outside_lead_window(tmp_db):
    """Event 60 min from now with lead=15 → must NOT alert yet.
    Otherwise the user gets pinged way too early and the alert is
    useless context-noise."""
    now = tmp_db.local_now_naive()
    in_60 = (now + timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S")
    tmp_db.create_event("долгая встреча", in_60)

    sent: list[str] = []
    async def fake_send(text: str):
        sent.append(text)

    count = await check_event_alerts(tmp_db, fake_send, lead_minutes=15)
    assert count == 0
    assert sent == []


@pytest.mark.asyncio
async def test_event_alert_skips_past_events(tmp_db):
    """Past events must never alert — the alert is meaningless after
    start_at and would surprise the user with stale info."""
    now = tmp_db.local_now_naive()
    past = (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    tmp_db.create_event("прошедшая встреча", past)

    sent: list[str] = []
    async def fake_send(text: str):
        sent.append(text)

    count = await check_event_alerts(tmp_db, fake_send, lead_minutes=120)
    assert count == 0
    assert sent == []


@pytest.mark.asyncio
async def test_event_alert_dedup_no_double_fire(tmp_db):
    """Once alerted, the event must not re-fire on subsequent ticks even
    if it's still inside the lead window. Without dedup the user would
    get the same alert every check_minutes for the lead duration."""
    now = tmp_db.local_now_naive()
    in_5 = (now + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    tmp_db.create_event("звонок", in_5)

    sent: list[str] = []
    async def fake_send(text: str):
        sent.append(text)

    first = await check_event_alerts(tmp_db, fake_send, lead_minutes=15)
    second = await check_event_alerts(tmp_db, fake_send, lead_minutes=15)
    assert first == 1
    assert second == 0
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_event_alert_marks_before_send_no_spam_on_failure(tmp_db):
    """At-most-once delivery: if send fails, the event must STILL be
    marked alerted so a transient Telegram outage doesn't cause spam
    once it recovers. Slightly different from reminders (at-least-once)
    because event alerts are auto-generated and the event itself remains
    in the calendar — a missed alert is far less bad than spam."""
    now = tmp_db.local_now_naive()
    in_10 = (now + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
    tmp_db.create_event("важная встреча", in_10)

    async def failing_send(text: str):
        raise RuntimeError("telegram down")

    count = await check_event_alerts(tmp_db, failing_send, lead_minutes=15)
    assert count == 0
    # alerted_at MUST be set even though send raised — otherwise next
    # tick re-tries forever and spams the user once Telegram recovers.
    row = tmp_db.db.execute(
        "SELECT alerted_at FROM events WHERE title = 'важная встреча'"
    ).fetchone()
    assert row["alerted_at"] is not None


@pytest.mark.asyncio
async def test_event_alert_resets_after_reschedule(tmp_db):
    """Reschedule must clear alerted_at — otherwise a moved event silently
    skips its new lead window. User-visible bug: 'я перенёс встречу на
    послезавтра, но бот не напомнил'."""
    now = tmp_db.local_now_naive()
    in_10 = (now + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
    tmp_db.create_event("ужин с Олегом", in_10)

    sent: list[str] = []
    async def fake_send(text: str):
        sent.append(text)

    # First fire alerts and marks alerted_at
    await check_event_alerts(tmp_db, fake_send, lead_minutes=15)
    assert len(sent) == 1

    # Reschedule to 8 min from now → must clear alerted_at
    new_time = (now + timedelta(minutes=8)).strftime("%Y-%m-%d %H:%M:%S")
    tmp_db.reschedule_event_by_hint("Олег", new_time)
    row = tmp_db.db.execute(
        "SELECT alerted_at FROM events WHERE title = 'ужин с Олегом'"
    ).fetchone()
    assert row["alerted_at"] is None

    # Next check must re-alert at the new time
    second = await check_event_alerts(tmp_db, fake_send, lead_minutes=15)
    assert second == 1
    assert len(sent) == 2


def test_format_event_alert_basic():
    """Title + minutes-until + HH:MM time always present. Other fields
    (location, person) only render when populated, so a bare event
    doesn't produce empty trailing icons."""
    from datetime import datetime
    now = datetime(2026, 4, 15, 9, 0, 0)
    event = {
        "title": "ужин с Машей",
        "start_at": "2026-04-15 09:15:00",
        "location": "кафе Пушкин",
        "related_person": "Маша",
    }
    text = _format_event_alert(event, now)
    assert "ужин с Машей" in text
    assert "09:15" in text
    assert "Через ~15 мин" in text
    assert "кафе Пушкин" in text


def test_format_event_alert_omits_empty_fields():
    """No location, no related_person → only title + time line. Avoid
    rendering '📍' with empty content (looks broken in Telegram)."""
    from datetime import datetime
    now = datetime(2026, 4, 15, 9, 0, 0)
    event = {
        "title": "стандап",
        "start_at": "2026-04-15 09:10:00",
        "location": None,
        "related_person": None,
    }
    text = _format_event_alert(event, now)
    assert "📍" not in text
    assert "👤" not in text


def test_format_event_alert_skips_redundant_person():
    """If related_person is already in the title (common: 'встреча с
    Машей' + related_person='Маша'), don't render '👤 Маша' — would be
    visual repetition."""
    from datetime import datetime
    now = datetime(2026, 4, 15, 9, 0, 0)
    event = {
        "title": "встреча с Машей",
        "start_at": "2026-04-15 09:15:00",
        "related_person": "Маша",
    }
    text = _format_event_alert(event, now)
    assert text.count("Маш") == 1  # appears in title only, not as 👤 line
