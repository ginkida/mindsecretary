"""Tests for integrations/weather.py — TZ-aware rain filter + range merge.

Protects against the "rain at 13:00 sent at 18:56" bug: datetime.now()
returns system-TZ (UTC in slim Docker images), so a user in Asia/Almaty
(UTC+5) at 18:56 would see the filter treat 13:00 local as "future"
because UTC hour == 13.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from mindsecretary.integrations.weather import (
    WeatherClient,
    _merge_rain_hours,
)


def _hourly(start_date: str, hours: list[tuple[int, int, int]]) -> dict:
    """Build a fake Open-Meteo hourly payload slice for a single date."""
    times = [f"{start_date}T{h:02d}:00" for h in range(24)]
    probs = [0] * 24
    codes = [0] * 24
    for hour, prob, code in hours:
        probs[hour] = prob
        codes[hour] = code
    return {
        "time": times,
        "precipitation_probability": probs,
        "weather_code": codes,
    }


def _fake_payload(date: str, hours: list[tuple[int, int, int]]) -> dict:
    return {
        "current": {"temperature_2m": 18, "weather_code": 0,
                    "wind_speed_10m": 5, "relative_humidity_2m": 60},
        "daily": {"time": [date], "temperature_2m_max": [20],
                   "temperature_2m_min": [10], "weather_code": [0],
                   "precipitation_sum": [0], "wind_speed_10m_max": [5]},
        "hourly": _hourly(date, hours),
    }


class TestMergeRainHours:
    def test_empty(self):
        assert _merge_rain_hours([]) == []

    def test_single_hour(self):
        assert _merge_rain_hours([(14, 70, 63)]) == [(14, 14, 70)]

    def test_consecutive_merge(self):
        assert _merge_rain_hours([(13, 60, 63), (14, 80, 65), (15, 70, 63)]) \
            == [(13, 15, 80)]

    def test_gap_creates_two_ranges(self):
        ranges = _merge_rain_hours([(13, 60, 63), (14, 70, 63), (18, 90, 95)])
        assert ranges == [(13, 14, 70), (18, 18, 90)]

    def test_unsorted_input_gets_ordered(self):
        ranges = _merge_rain_hours([(15, 50, 63), (13, 80, 63), (14, 70, 63)])
        assert ranges == [(13, 15, 80)]


class TestRainTodayTZFilter:
    """The filter must use profile TZ, not system TZ."""

    def _client(self, tz: str = "Asia/Almaty") -> WeatherClient:
        return WeatherClient(55.0, 37.0, timezone=tz)

    def test_past_hours_dropped_in_local_tz(self):
        """At 18:56 local (Asia/Almaty, UTC+5), 13:00 rain must be filtered out."""
        client = self._client("Asia/Almaty")
        payload = _fake_payload("2026-04-24", [
            (13, 80, 63),  # past — already happened
            (14, 70, 63),  # past
            (19, 60, 63),  # future from 18:56
            (20, 90, 95),  # future thunderstorm
        ])

        fake_now = datetime(2026, 4, 24, 18, 56, tzinfo=ZoneInfo("Asia/Almaty"))
        with patch("mindsecretary.integrations.weather.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            # _parse calls .strftime on the result, so keep real strftime
            mock_dt.strftime = datetime.strftime
            result = client._parse(payload, days=1)

        hours = [h for h, _, _ in result["rain_today"]]
        assert 13 not in hours
        assert 14 not in hours
        assert 19 in hours
        assert 20 in hours

    def test_current_hour_included(self):
        """Rain at the current hour is still in-progress → report it."""
        client = self._client("Asia/Almaty")
        payload = _fake_payload("2026-04-24", [(18, 80, 63)])
        fake_now = datetime(2026, 4, 24, 18, 30, tzinfo=ZoneInfo("Asia/Almaty"))
        with patch("mindsecretary.integrations.weather.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.strftime = datetime.strftime
            result = client._parse(payload, days=1)

        assert any(h == 18 for h, _, _ in result["rain_today"])

    def test_low_probability_skipped(self):
        client = self._client("Asia/Almaty")
        payload = _fake_payload("2026-04-24", [(20, 40, 63)])  # 40% — below 50
        fake_now = datetime(2026, 4, 24, 10, 0, tzinfo=ZoneInfo("Asia/Almaty"))
        with patch("mindsecretary.integrations.weather.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.strftime = datetime.strftime
            result = client._parse(payload, days=1)
        assert result["rain_today"] == []

    def test_other_dates_ignored(self):
        """Even if days=2 returns tomorrow, rain_today must stay today-only."""
        client = self._client("Asia/Almaty")
        payload = {
            "current": {}, "daily": {"time": []},
            "hourly": {
                "time": ["2026-04-24T20:00", "2026-04-25T13:00"],
                "precipitation_probability": [80, 90],
                "weather_code": [63, 63],
            },
        }
        fake_now = datetime(2026, 4, 24, 10, 0, tzinfo=ZoneInfo("Asia/Almaty"))
        with patch("mindsecretary.integrations.weather.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.strftime = datetime.strftime
            result = client._parse(payload, days=2)
        hours = [h for h, _, _ in result["rain_today"]]
        assert hours == [20]


class TestFormatCurrentRain:
    def test_formats_range(self):
        client = WeatherClient(55.0, 37.0, timezone="Asia/Almaty")
        forecast = {
            "current": {"temperature_2m": 15, "condition": "облачно",
                         "wind": 5, "humidity": 60,
                         "temp": 15},
            "rain_today": [(13, 60, 63), (14, 80, 63), (15, 70, 63)],
        }
        # format_current reads `current` as nested dict — adjust shape
        forecast["current"] = {"temp": 15, "condition": "облачно",
                                "wind": 5, "humidity": 60}
        text = client.format_current(forecast)
        assert "13:00-16:00" in text
        assert "до 80%" in text

    def test_formats_single_hour(self):
        client = WeatherClient(55.0, 37.0, timezone="Asia/Almaty")
        forecast = {
            "current": {"temp": 15, "condition": "облачно",
                         "wind": 5, "humidity": 60},
            "rain_today": [(20, 60, 63)],
        }
        text = client.format_current(forecast)
        assert "20:00" in text
        # single-hour uses bare HH:00 (no range dash) in format_current
        assert "20:00-" not in text


class TestParseDefensiveAgainstNoneArrays:
    """Open-Meteo's JSON schema permits null for missing arrays; pre-fix
    `len(daily.get("X", []))` exploded with TypeError because dict.get's
    default only applies on MISSING keys, not on explicit None values.
    A single null field would crash _parse → propagate to scheduler's
    _check_weather → silent suppression of the entire forecast cycle."""

    def test_null_daily_arrays_dont_crash(self):
        """All daily arrays explicitly None — extract should bail
        gracefully with empty 'daily' list rather than TypeError."""
        client = WeatherClient(55.0, 37.0, timezone="Asia/Almaty")
        payload = {
            "current": {},
            "daily": {
                "time": None,
                "temperature_2m_max": None,
                "temperature_2m_min": None,
                "weather_code": None,
                "precipitation_sum": None,
                "wind_speed_10m_max": None,
            },
            "hourly": {},
        }
        # Pre-fix: TypeError. Post-fix: empty result, no exception.
        result = client._parse(payload, days=1)
        assert result["daily"] == []

    def test_null_hourly_arrays_dont_crash(self):
        """Hourly fields all None — rain_today should be empty, no crash."""
        client = WeatherClient(55.0, 37.0, timezone="Asia/Almaty")
        payload = {
            "current": {},
            "daily": {"time": []},
            "hourly": {
                "time": None,
                "precipitation_probability": None,
                "weather_code": None,
            },
        }
        result = client._parse(payload, days=1)
        assert result["rain_today"] == []

    def test_partial_array_lengths_dont_crash(self):
        """Mismatched lengths (longer time array than weather_code) used
        to work via the `if i < len(...)` guard, but only when the
        outer arrays survived the None-len trap. Confirm short arrays
        still get the per-index fallback."""
        client = WeatherClient(55.0, 37.0, timezone="Asia/Almaty")
        payload = {
            "current": {},
            "daily": {
                "time": ["2026-04-24", "2026-04-25"],
                "weather_code": [63],  # only 1 entry for 2 dates
                "temperature_2m_max": [20],
                "temperature_2m_min": [10],
                "precipitation_sum": [0],
                "wind_speed_10m_max": [5],
            },
            "hourly": {},
        }
        result = client._parse(payload, days=2)
        assert len(result["daily"]) == 2
        # First day got the real code; second fell back to default.
        assert result["daily"][0]["temp_max"] == 20
        assert result["daily"][1]["temp_max"] is None


class TestWeatherClientInit:
    def test_invalid_timezone_falls_back_to_none(self):
        client = WeatherClient(55.0, 37.0, timezone="Nonsense/Invalid")
        assert client._tz_info is None

    def test_valid_timezone_resolved(self):
        client = WeatherClient(55.0, 37.0, timezone="Asia/Almaty")
        assert client._tz_info is not None
        assert "Almaty" in str(client._tz_info)
