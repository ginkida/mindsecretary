"""Voyage embed retry behavior in Memory.

Memory wraps Voyage embed in a 3x exp-backoff retry. Transient failures
should not poison memories with embed_failed quarantine; only terminal
failures should fall through to that fallback.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mindsecretary.core.memory import Memory


def _make_memory() -> tuple[Memory, MagicMock]:
    """Build a Memory bound to an in-memory SQLite, with the Voyage client
    swapped for a MagicMock the test fully controls."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    voyage = MagicMock()
    memory = Memory.__new__(Memory)
    memory.db = conn
    memory.voyage = voyage
    memory.model = "voyage-3"
    memory.relevance_w = 0.6
    memory.importance_w = 0.4
    memory._ensure_table()
    return memory, voyage


class TestEmbedRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try_no_sleep(self):
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])

        with patch("mindsecretary.core.memory.asyncio.sleep") as mock_sleep:
            result = await memory._embed_with_retry(["hello"], "document")

        assert voyage.embed.call_count == 1
        assert result.embeddings[0] == emb.tolist()
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_transient_then_succeeds(self):
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        ok = MagicMock(embeddings=[emb.tolist()])

        # Fail twice, then succeed — exercises both retry boundaries.
        voyage.embed.side_effect = [
            ConnectionError("network blip"),
            ConnectionError("network blip 2"),
            ok,
        ]

        with patch("mindsecretary.core.memory.asyncio.sleep") as mock_sleep:
            result = await memory._embed_with_retry(["hi"], "document")

        assert voyage.embed.call_count == 3
        assert result is ok
        # Two backoffs (1s, 2s) before the third attempt succeeded.
        assert mock_sleep.call_count == 2
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_propagates_after_max_retries(self):
        memory, voyage = _make_memory()
        voyage.embed.side_effect = ConnectionError("flaky")

        with patch("mindsecretary.core.memory.asyncio.sleep"):
            with pytest.raises(ConnectionError, match="flaky"):
                await memory._embed_with_retry(["hi"], "document")

        assert voyage.embed.call_count == 3

    @pytest.mark.asyncio
    async def test_save_uses_retry_then_falls_back_to_quarantine(self):
        """If all 3 attempts fail, save() still puts the row in embed_failed
        — terminal failure preserves the quarantine contract."""
        memory, voyage = _make_memory()
        voyage.embed.side_effect = ConnectionError("down hard")

        with patch("mindsecretary.core.memory.asyncio.sleep"):
            mem_id = await memory.save("стоматолог в среду", "todo")

        assert voyage.embed.call_count == 3  # all three attempts ran
        row = memory.db.execute(
            "SELECT status FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        assert row["status"] == "embed_failed"

    @pytest.mark.asyncio
    async def test_save_recovers_when_second_attempt_succeeds(self):
        """Transient failure should NOT quarantine — the whole point of retry."""
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)
        ok = MagicMock(embeddings=[emb.tolist()])
        voyage.embed.side_effect = [ConnectionError("blip"), ok]

        with patch("mindsecretary.core.memory.asyncio.sleep"):
            mem_id = await memory.save("звонок маме завтра", "todo")

        row = memory.db.execute(
            "SELECT status FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        assert row["status"] == "active"
        assert voyage.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_query_embed_uses_retry(self):
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        ok = MagicMock(embeddings=[emb.tolist()])
        voyage.embed.side_effect = [ConnectionError("blip"), ok]

        with patch("mindsecretary.core.memory.asyncio.sleep"):
            arr = await memory._embed_query("найди заметки про работу")

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert voyage.embed.call_count == 2
