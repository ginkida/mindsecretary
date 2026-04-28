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


class TestMemoryUpdateByHint:
    """update_by_hint replaces a memory's content + re-embeds, with strict
    safety on ambiguous matches — memories are too sensitive to overwrite
    a guess."""

    @pytest.mark.asyncio
    async def test_unique_match_updates_content_and_reembeds(self):
        memory, voyage = _make_memory()
        old_emb = np.random.randn(1024).astype(np.float32)
        old_emb /= np.linalg.norm(old_emb)
        new_emb = np.random.randn(1024).astype(np.float32)
        new_emb /= np.linalg.norm(new_emb)

        # Save the original memory first (mock voyage to give old_emb)
        voyage.embed.return_value = MagicMock(embeddings=[old_emb.tolist()])
        mem_id = await memory.save("работает в Yandex", "work", importance=7)

        # Now update — voyage returns new_emb for the new content
        voyage.embed.return_value = MagicMock(embeddings=[new_emb.tolist()])
        result = await memory.update_by_hint("Yandex", "работает в Сбере")

        assert result["status"] == "ok"
        assert result["memory"]["id"] == mem_id
        assert result["memory"]["content"] == "работает в Сбере"

        row = memory.db.execute(
            "SELECT content, embedding, confidence FROM memories WHERE id = ?",
            (mem_id,),
        ).fetchone()
        assert row["content"] == "работает в Сбере"
        # Embedding bytes changed — old_emb.tobytes() != new_emb.tobytes()
        assert row["embedding"] != old_emb.tobytes()
        # Confidence bumped to at least 0.95 (correction = strong signal)
        assert row["confidence"] >= 0.95

    @pytest.mark.asyncio
    async def test_not_found_returns_not_found(self):
        memory, _ = _make_memory()
        result = await memory.update_by_hint("nope", "что-то новое")
        assert result == {"status": "not_found"}

    @pytest.mark.asyncio
    async def test_ambiguous_refuses_with_count_and_samples(self):
        memory, voyage = _make_memory()
        # Distinct embeddings per save so save()'s 0.92 cosine dedup does NOT
        # collapse them into one row — we need 3 active rows for ambiguity.
        embeds = []
        for _ in range(3):
            v = np.random.randn(1024).astype(np.float32)
            v /= np.linalg.norm(v)
            embeds.append(MagicMock(embeddings=[v.tolist()]))
        voyage.embed.side_effect = embeds

        await memory.save("работает в Yandex офисе", "work")
        await memory.save("работает в Yandex удалённо", "work")
        await memory.save("раньше работал в Yandex", "work")

        # Reset call counter; ambiguity must NOT call voyage.embed for the
        # new_content — refusing to update means we don't waste an embed.
        voyage.embed.reset_mock()
        voyage.embed.side_effect = None

        result = await memory.update_by_hint("Yandex", "работает в Сбере")
        assert result["status"] == "ambiguous"
        assert result["count"] == 3
        assert len(result["samples"]) == 3  # capped at 3
        # No embed call — refused before embedding
        assert voyage.embed.call_count == 0

    @pytest.mark.asyncio
    async def test_embed_failure_leaves_row_untouched(self):
        memory, voyage = _make_memory()
        good_emb = np.random.randn(1024).astype(np.float32)
        good_emb /= np.linalg.norm(good_emb)
        voyage.embed.return_value = MagicMock(embeddings=[good_emb.tolist()])
        mem_id = await memory.save("isolated fact", "personal")

        # Make the update embed fail through all 3 retry attempts
        from unittest.mock import patch
        voyage.embed.side_effect = ConnectionError("voyage down")
        with patch("mindsecretary.core.memory.asyncio.sleep"):
            result = await memory.update_by_hint("isolated", "new content")
        assert result == {"status": "embed_failed"}

        # Row content unchanged — no partial corruption
        row = memory.db.execute(
            "SELECT content FROM memories WHERE id = ?", (mem_id,),
        ).fetchone()
        assert row["content"] == "isolated fact"

    @pytest.mark.asyncio
    async def test_invalid_args(self):
        memory, _ = _make_memory()
        assert (await memory.update_by_hint("", "x"))["status"] == "invalid"
        assert (await memory.update_by_hint("y", ""))["status"] == "invalid"
        assert (await memory.update_by_hint("  ", "x"))["status"] == "invalid"

    @pytest.mark.asyncio
    async def test_deleted_memories_excluded(self):
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])

        mem_id = await memory.save("про работу", "work")
        memory.delete(mem_id)  # status='deleted'

        # Hint that would have matched the deleted row → not_found
        result = await memory.update_by_hint("работ", "обновлённое")
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_cyrillic_case_insensitive_hint(self):
        """pylower path same as cancel_reminder — UPPERCASE hint should
        match lowercase stored content."""
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])
        await memory.save("стоматолог по средам", "health")

        result = await memory.update_by_hint("СТОМАТОЛОГ", "новый врач")
        assert result["status"] == "ok"
        assert result["memory"]["content"] == "новый врач"


class TestSearchConfidenceRanking:
    """Search ranking now blends confidence into the final score so
    low-confidence sources (voice transcription ≈ 0.82, photo ≈ 0.78)
    don't outrank a typed fact at equal cosine. Soft formula
    (0.7 + 0.3 * confidence) — uncertain memories demoted, not excluded."""

    @pytest.mark.asyncio
    async def test_higher_confidence_outranks_lower_at_equal_cosine(self):
        memory, voyage = _make_memory()
        # Two memories with the SAME embedding (so equal cosine to query)
        # but different confidence — high-conf should rank first.
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Insert directly to bypass save()'s dedup logic and lock confidence
        for mid, conf in [("low", 0.5), ("high", 1.0)]:
            memory.db.execute(
                "INSERT INTO memories (id, content, embedding, category, "
                "importance, confidence, status) "
                "VALUES (?, ?, ?, 'work', 5, ?, 'active')",
                (mid, f"факт {mid}", emb.tobytes(), conf),
            )
        memory.db.commit()

        # Mock query embedding to match the stored embedding
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])

        results = await memory.search("факт", top_k=2)
        assert len(results) == 2
        # High-confidence memory MUST appear first
        assert results[0]["id"] == "high"
        assert results[1]["id"] == "low"
        # final_score gap reflects the confidence multiplier
        assert results[0]["final_score"] > results[1]["final_score"]

    @pytest.mark.asyncio
    async def test_default_confidence_preserves_legacy_ordering(self):
        """Rows with default confidence=1.0 (treated as 'fully trusted')
        should rank EXACTLY as before — confidence factor adds 0 penalty."""
        memory, voyage = _make_memory()
        emb_a = np.random.randn(1024).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)
        emb_b = np.random.randn(1024).astype(np.float32)
        emb_b /= np.linalg.norm(emb_b)

        for mid, conf, e in [("a", 1.0, emb_a), ("b", 1.0, emb_b)]:
            memory.db.execute(
                "INSERT INTO memories (id, content, embedding, category, "
                "importance, confidence, status) "
                "VALUES (?, ?, ?, 'work', 5, ?, 'active')",
                (mid, f"факт {mid}", e.tobytes(), conf),
            )
        memory.db.commit()

        # Query embedding identical to emb_a — A wins on cosine, B is far.
        voyage.embed.return_value = MagicMock(embeddings=[emb_a.tolist()])
        results = await memory.search("факт", top_k=2)
        assert results[0]["id"] == "a"

        # With default confidence, the multiplier is 0.7 + 0.3 = 1.0 — so
        # final_score equals (cosine*0.6 + (5/10)*0.4) * 1.0 * recency.
        # Lower bound: cosine ≈ 1.0 → 0.6 + 0.2 = 0.8 (before recency).
        # We're not asserting the exact value (recency varies); we just
        # assert the multiplier didn't accidentally penalize defaults.
        assert results[0]["final_score"] > 0.5

    @pytest.mark.asyncio
    async def test_zero_confidence_keeps_floor_above_zero(self):
        """Even at confidence=0 the soft blend leaves 70% of the score —
        uncertain memories are demoted, never silently zeroed out."""
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)
        memory.db.execute(
            "INSERT INTO memories (id, content, embedding, category, "
            "importance, confidence, status) "
            "VALUES ('z', 'unsure fact', ?, 'work', 5, 0.0, 'active')",
            (emb.tobytes(),),
        )
        memory.db.commit()
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])

        results = await memory.search("unsure fact", top_k=1)
        assert len(results) == 1
        # Non-zero result — zero-confidence didn't filter the row out
        assert results[0]["final_score"] > 0


class TestMemoryDeleteByHint:
    """delete_by_hint mirrors update_by_hint's safety contract: ambiguous
    hint refuses; soft-delete preserves /undo."""

    @pytest.mark.asyncio
    async def test_unique_match_soft_deletes(self):
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])
        mem_id = await memory.save("шахматы по выходным", "personal")

        result = await memory.delete_by_hint("шахматы")

        assert result["status"] == "ok"
        assert result["memory"]["id"] == mem_id
        # Status is now 'deleted', not removed — /undo path stays open
        row = memory.db.execute(
            "SELECT status FROM memories WHERE id = ?", (mem_id,),
        ).fetchone()
        assert row["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_undo_after_delete_works(self):
        """Confirms the soft-delete contract: get_last_deleted + restore
        chain still works after delete_by_hint, same as /forget."""
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])
        mem_id = await memory.save("факт про Y", "personal")

        await memory.delete_by_hint("факт про Y")
        last = memory.get_last_deleted()
        assert last is not None
        assert last["id"] == mem_id

        restored = memory.restore(mem_id)
        assert restored is True
        row = memory.db.execute(
            "SELECT status FROM memories WHERE id = ?", (mem_id,),
        ).fetchone()
        assert row["status"] == "active"

    @pytest.mark.asyncio
    async def test_not_found(self):
        memory, _ = _make_memory()
        result = await memory.delete_by_hint("nope")
        assert result == {"status": "not_found"}

    @pytest.mark.asyncio
    async def test_ambiguous_refuses(self):
        memory, voyage = _make_memory()
        embeds = []
        for _ in range(3):
            v = np.random.randn(1024).astype(np.float32)
            v /= np.linalg.norm(v)
            embeds.append(MagicMock(embeddings=[v.tolist()]))
        voyage.embed.side_effect = embeds

        await memory.save("факт A про шахматы", "personal")
        await memory.save("факт B про шахматы", "personal")
        await memory.save("факт C про шахматы", "personal")

        result = await memory.delete_by_hint("шахматы")
        assert result["status"] == "ambiguous"
        assert result["count"] == 3
        assert len(result["samples"]) == 3
        # All three rows still active — no destructive action under ambiguity
        active = memory.db.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'",
        ).fetchone()
        assert active[0] == 3

    @pytest.mark.asyncio
    async def test_invalid_args(self):
        memory, _ = _make_memory()
        assert (await memory.delete_by_hint(""))["status"] == "invalid"
        assert (await memory.delete_by_hint("  "))["status"] == "invalid"

    @pytest.mark.asyncio
    async def test_already_deleted_excluded(self):
        """Already-deleted rows mustn't be matched by hint — otherwise
        a delete after delete would 'succeed' and the second user could
        not understand why /undo restores something they meant gone."""
        memory, voyage = _make_memory()
        emb = np.random.randn(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)
        voyage.embed.return_value = MagicMock(embeddings=[emb.tolist()])
        mem_id = await memory.save("про работу", "work")
        memory.delete(mem_id)

        result = await memory.delete_by_hint("работ")
        assert result == {"status": "not_found"}
