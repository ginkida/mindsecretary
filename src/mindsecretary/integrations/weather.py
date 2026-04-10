from __future__ import annotations

import logging
from datetime import datetime, timedelta

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


class WeatherClient:
    """Open-Meteo API — бесплатный, без ключа."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, latitude: float, longitude: float, timezone: str = "Europe/Moscow"):
        self.lat = latitude
        self.lon = longitude
        self.tz = timezone
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
            logger.error("Weather API failed: %s", e)
            return {"error": str(e)}

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

        # Daily forecast
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        result["daily"] = []
        for i, date in enumerate(dates[:days]):
            code = (daily.get("weather_code") or [0])[i] if i < len(daily.get("weather_code", [])) else 0
            result["daily"].append({
                "date": date,
                "temp_max": (daily.get("temperature_2m_max") or [None])[i],
                "temp_min": (daily.get("temperature_2m_min") or [None])[i],
                "condition": WMO_CODES.get(code, f"код {code}"),
                "precipitation": (daily.get("precipitation_sum") or [0])[i],
                "wind_max": (daily.get("wind_speed_10m_max") or [None])[i],
            })

        # Hourly — find rain windows for today
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        precip_probs = hourly.get("precipitation_probability", [])
        rain_hours = []
        now_hour = datetime.now().hour
        for i, t in enumerate(times[:24]):  # only today
            hour = int(t[11:13]) if len(t) > 12 else 0
            prob = precip_probs[i] if i < len(precip_probs) else 0
            if hour >= now_hour and prob and prob >= 50:
                rain_hours.append((hour, prob))

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
            hours = ", ".join(f"{h}:00 ({p}%)" for h, p in rain[:3])
            text += f"\nДождь ожидается: {hours}"

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
