from __future__ import annotations

import logging
from datetime import datetime

from ..core.database import Database

logger = logging.getLogger(__name__)


async def check_reminders(db: Database, send_fn) -> int:
    """Check and send due reminders. Returns count sent."""
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


async def check_birthdays(db: Database, send_fn) -> int:
    """Check upcoming birthdays and notify. Returns count sent."""
    upcoming = db.get_upcoming_birthdays(days=3)
    sent = 0
    today_md = datetime.now().strftime("%m-%d")
    for c in upcoming:
        bday = c.get("birthday", "")
        # Check if birthday is today or coming up
        bday_md = bday[-5:] if len(bday) >= 5 else bday
        name = c["name"]
        relation = f" ({c['relation']})" if c.get("relation") else ""

        if bday_md == today_md:
            text = f"🎂 Сегодня день рождения: {name}{relation}!"
        else:
            text = f"📅 Скоро день рождения: {name}{relation} — {bday}"

        try:
            await send_fn(text)
            sent += 1
        except Exception as e:
            logger.error("Failed to send birthday alert: %s", type(e).__name__)
    return sent


async def check_weather_change(weather_client, last_forecast: dict | None,
                               send_fn) -> dict | None:
    """Check if weather changed significantly. Returns new forecast."""
    if not weather_client:
        return last_forecast

    try:
        forecast = await weather_client.get_forecast(days=1)
    except Exception as e:
        logger.error("Weather check failed: %s", type(e).__name__)
        return last_forecast

    if "error" in forecast:
        return last_forecast

    # First run — just cache
    if last_forecast is None:
        return forecast

    # Compare: alert on new rain
    old_rain = set(h for h, _ in (last_forecast.get("rain_today") or []))
    new_rain = set(h for h, _ in (forecast.get("rain_today") or []))
    new_rain_hours = new_rain - old_rain

    if new_rain_hours:
        hours_str = ", ".join(f"{h}:00" for h in sorted(new_rain_hours))
        try:
            await send_fn(f"🌧 Появился дождь в прогнозе: {hours_str}")
        except Exception as e:
            logger.error("Failed to send weather alert: %s", type(e).__name__)

    return forecast
