from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo WMO weather codes → human-readable Russian
WMO_CODES = {
    0: "ясно", 1: "малооблачно", 2: "облачно", 3: "пасмурно",
    45: "туман", 48: "изморозь",
    51: "мелкий дождь", 53: "дождь", 55: "сильный дождь",
    56: "ледяной дождь", 57: "сильный ледяной дождь",
    61: "небольшой дождь", 63: "дождь", 65: "ливень",
    66: "ледяной дождь", 67: "сильный ледяной дождь",
    71: "небольшой снег", 73: "снег", 75: "сильный снег",
    77: "снежная крупа",
    80: "ливень", 81: "ливень", 82: "сильный ливень",
    85: "снегопад", 86: "сильный снегопад",
    95: "гроза", 96: "гроза с градом", 99: "сильная гроза с градом",
}


def _merge_rain_hours(
    rain: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Collapse consecutive hourly rain entries into (start, end, max_prob) ranges.

    rain items are (hour, prob, weather_code); returned (start_hour, end_hour,
    max_prob) where start/end are inclusive hour indices. Used by the
    scheduler and the briefing to render "13:00-16:00 (до 80%)" instead of
    a flat hour list.
    """
    if not rain:
        return []
    ordered = sorted(rain, key=lambda x: x[0])
    ranges: list[list[int]] = []
    for hour, prob, _code in ordered:
        if ranges and hour == ranges[-1][1] + 1:
            ranges[-1][1] = hour
            if prob > ranges[-1][2]:
                ranges[-1][2] = prob
        else:
            ranges.append([hour, hour, prob])
    return [(s, e, p) for s, e, p in ranges]


class WeatherClient:
    """Open-Meteo API — бесплатный, без ключа."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float, longitude: float, timezone: str = "Europe/Moscow"):
        self.lat = latitude
        self.lon = longitude
        self.tz = timezone
        # Resolve ZoneInfo once so _parse can compute local "now" correctly.
        # datetime.now() defaults to system TZ (UTC in slim Docker images);
        # without this the rain_today filter skips the wrong hours and
        # leaks past-hour entries into the hourly forecast.
        try:
            self._tz_info: ZoneInfo | None = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Invalid weather timezone %r, falling back to system TZ", timezone)
            self._tz_info = None
        self.client = httpx.AsyncClient(timeout=10.0)

    async def get_forecast(self, days: int = 1) -> dict:
        """Получить прогноз погоды на N дней."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "timezone": self.tz,
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,"
                     "precipitation_sum,wind_speed_10m_max,sunrise,sunset",
            "hourly": "temperature_2m,weather_code,precipitation_probability",
            "forecast_days": min(days, 7),
        }

        try:
            resp = await self.client.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Weather API failed: %s", type(e).__name__)
            return {"error": type(e).__name__}

        return self._parse(data, days)

    def _parse(self, data: dict, days: int) -> dict:
        result: dict = {}

        # Current conditions
        current = data.get("current", {})
        if current:
            code = current.get("weather_code", 0)
            result["current"] = {
                "temp": current.get("temperature_2m"),
                "condition": WMO_CODES.get(code, f"код {code}"),
                "wind": current.get("wind_speed_10m"),
                "humidity": current.get("relative_humidity_2m"),
            }

        # Daily forecast — extract each array once with an `or []` fallback
        # so an explicit-None field from Open-Meteo (rare but allowed by the
        # JSON schema) doesn't crash with `len(None)`. dict.get's default
        # only kicks in when the key is MISSING, not when the value is None.
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        codes = daily.get("weather_code") or []
        temp_maxes = daily.get("temperature_2m_max") or []
        temp_mins = daily.get("temperature_2m_min") or []
        precip_sums = daily.get("precipitation_sum") or []
        wind_maxes = daily.get("wind_speed_10m_max") or []
        result["daily"] = []
        for i, date in enumerate(dates[:days]):
            code = codes[i] if i < len(codes) else 0
            result["daily"].append({
                "date": date,
                "temp_max": temp_maxes[i] if i < len(temp_maxes) else None,
                "temp_min": temp_mins[i] if i < len(temp_mins) else None,
                "condition": WMO_CODES.get(code, f"код {code}"),
                "precipitation": precip_sums[i] if i < len(precip_sums) else 0,
                "wind_max": wind_maxes[i] if i < len(wind_maxes) else None,
            })

        # Hourly — find rain windows for today, in the user's local TZ.
        # Times from Open-Meteo are already local because the request passes
        # `timezone=self.tz`, but "now" must be computed in the same TZ —
        # otherwise we compare local forecast hours against UTC/system hours
        # and leak past hours (the "rain at 13:00 sent at 18:56" bug).
        # `or []` defends against a null hourly array shape from the API.
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        precip_probs = hourly.get("precipitation_probability") or []
        weather_codes = hourly.get("weather_code") or []
        now_local = datetime.now(self._tz_info) if self._tz_info else datetime.now()
        today_str = now_local.strftime("%Y-%m-%d")
        now_hour = now_local.hour
        rain_hours: list[tuple[int, int, int]] = []
        for i, t in enumerate(times):
            if len(t) < 13 or not t.startswith(today_str):
                continue
            try:
                hour = int(t[11:13])
            except ValueError:
                continue
            prob = precip_probs[i] if i < len(precip_probs) else 0
            code = weather_codes[i] if i < len(weather_codes) else 0
            if hour >= now_hour and prob and prob >= 50:
                rain_hours.append((hour, int(prob), int(code)))

        result["rain_today"] = rain_hours

        return result

    def format_current(self, forecast: dict) -> str:
        """Форматировать текущую погоду для промпта."""
        if "error" in forecast:
            return f"Погода недоступна: {forecast['error']}"

        current = forecast.get("current", {})
        if not current:
            return "Нет данных о погоде."

        temp = current.get("temp", "?")
        cond = current.get("condition", "?")
        wind = current.get("wind", "?")

        text = f"{temp}°C, {cond}, ветер {wind} км/ч"

        rain = forecast.get("rain_today", [])
        if rain:
            # rain entries are (hour, prob, code); collapse consecutive hours
            # into ranges so the briefing reads "13:00-16:00" not "13, 14, 15".
            ranges = _merge_rain_hours(rain)
            parts = [
                f"{s:02d}:00" if s == e else f"{s:02d}:00-{(e + 1) % 24:02d}:00"
                for s, e, _prob in ranges
            ]
            max_prob = max((p for _s, _e, p in ranges), default=0)
            text += f"\nДождь ожидается: {', '.join(parts)} (до {max_prob}%)"

        return text

    def format_daily(self, forecast: dict) -> str:
        """Форматировать прогноз по дням для брифинга."""
        if "error" in forecast:
            return f"Погода недоступна: {forecast['error']}"

        lines = []
        for day in forecast.get("daily", []):
            t_min = day.get("temp_min", "?")
            t_max = day.get("temp_max", "?")
            cond = day.get("condition", "?")
            precip = day.get("precipitation", 0)
            line = f"{day['date']}: {t_min}..{t_max}°C, {cond}"
            if precip and precip > 0:
                line += f", осадки {precip} мм"
            lines.append(line)

        return "\n".join(lines) or "Нет данных."

    async def close(self):
        await self.client.aclose()
