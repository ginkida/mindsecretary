"""Tests for llm/client.py — retry logic and stop_reason handling."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import pytest

from mindsecretary.llm.client import AnthropicClient


@pytest.fixture
def client(monkeypatch):
    """AnthropicClient with retry delay stripped for fast tests."""
    monkeypatch.setattr("mindsecretary.llm.client._BASE_DELAY", 0.0)
    return AnthropicClient(api_key="test-key", model="claude-test")


def _ok_response(stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason=stop_reason,
    )


def _conn_error() -> anthropic.APIConnectionError:
    """Build an APIConnectionError without running its __init__ (which requires a real request)."""
    return anthropic.APIConnectionError.__new__(anthropic.APIConnectionError)


def _status_error(code: int) -> anthropic.APIStatusError:
    err = anthropic.APIStatusError.__new__(anthropic.APIStatusError)
    err.status_code = code
    return err


_MSG = [{"role": "user", "content": "hi"}]


class TestRetry:
    @pytest.mark.asyncio
    async def test_succeeds_without_retry(self, client):
        client.client.messages.create = AsyncMock(return_value=_ok_response())
        resp = await client.chat(system="s", messages=_MSG)
        assert resp.text == "ok"
        assert client.client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self, client):
        client.client.messages.create = AsyncMock(
            side_effect=[_conn_error(), _conn_error(), _ok_response()],
        )
        resp = await client.chat(system="s", messages=_MSG)
        assert resp.text == "ok"
        assert client.client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_retryable_5xx(self, client):
        client.client.messages.create = AsyncMock(
            side_effect=[_status_error(503), _ok_response()],
        )
        resp = await client.chat(system="s", messages=_MSG)
        assert resp.text == "ok"
        assert client.client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_429(self, client):
        client.client.messages.create = AsyncMock(
            side_effect=[_status_error(429), _ok_response()],
        )
        resp = await client.chat(system="s", messages=_MSG)
        assert client.client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_on_400(self, client):
        err = _status_error(400)
        client.client.messages.create = AsyncMock(side_effect=err)
        with pytest.raises(anthropic.APIStatusError):
            await client.chat(system="s", messages=_MSG)
        assert client.client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_does_not_retry_on_non_anthropic_error(self, client):
        client.client.messages.create = AsyncMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            await client.chat(system="s", messages=_MSG)
        assert client.client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self, client):
        client.client.messages.create = AsyncMock(side_effect=_conn_error())
        with pytest.raises(anthropic.APIConnectionError):
            await client.chat(system="s", messages=_MSG)
        assert client.client.messages.create.call_count == 3


class TestStopReason:
    @pytest.mark.asyncio
    async def test_warns_on_max_tokens(self, client, caplog):
        client.client.messages.create = AsyncMock(
            return_value=_ok_response(stop_reason="max_tokens"),
        )
        with caplog.at_level(logging.WARNING, logger="mindsecretary.llm.client"):
            await client.chat(system="s", messages=_MSG, max_tokens=512)
        assert any("max_tokens" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_warning_on_end_turn(self, client, caplog):
        client.client.messages.create = AsyncMock(return_value=_ok_response())
        with caplog.at_level(logging.WARNING, logger="mindsecretary.llm.client"):
            await client.chat(system="s", messages=_MSG)
        assert not any("truncated" in r.message for r in caplog.records)
