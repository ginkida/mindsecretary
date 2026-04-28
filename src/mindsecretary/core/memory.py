from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import voyageai

logger = logging.getLogger(__name__)

_VOYAGE_MAX_RETRIES = 3
_VOYAGE_BASE_DELAY = 1.0  # seconds


class Memory:
    def __init__(self, db: sqlite3.Connection, voyage_api_key: str,
                 model: str = "voyage-3",
                 relevance_weight: float = 0.6,
                 importance_weight: float = 0.4):
        self.db = db
        self.voyage = voyageai.Client(api_key=voyage_api_key)
        self.model = model
        self.relevance_w = relevance_weight
        self.importance_w = importance_weight
        self._ensure_table()

    def _ensure_table(self):
        # Register pylower defensively — Memory.update_by_hint relies on it.
        # The same function is also registered by Database.__init__, but
        # Memory may be wired to a connection without that path (test
        # fixtures, future refactors). create_function is idempotent on
        # the same connection, so the double-register is safe.
        self.db.create_function(
            "pylower", 1, lambda s: s.lower() if isinstance(s, str) else s,
        )
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                category TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                related_person TEXT,
                related_date TEXT,
                source_type TEXT,
                source_ref TEXT,
                confidence REAL DEFAULT 1.0,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_mem_cat ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
            CREATE INDEX IF NOT EXISTS idx_mem_person ON memories(related_person);
        """)
        self._run_migrations()
        self.db.commit()

    def _run_migrations(self):
        """Idempotent column additions. Narrow exception to 'duplicate column'
        so we don't silently swallow real DB errors (locked, I/O, etc)."""
        migrations = [
            "ALTER TABLE memories ADD COLUMN source_type TEXT",
            "ALTER TABLE memories ADD COLUMN source_ref TEXT",
            "ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0",
            "CREATE INDEX IF NOT EXISTS idx_mem_source_ref ON memories(source_ref)",
        ]
        for sql in migrations:
            try:
                self.db.execute(sql)
            except sqlite3.OperationalError as e:
                # Expected on re-run for ALTER (column already exists).
                # CREATE INDEX IF NOT EXISTS is truly idempotent, won't raise.
                if "duplicate column" not in str(e).lower():
                    raise

    async def _embed_with_retry(self, texts: list[str], input_type: str):
        """Call Voyage embed with 3x exp-backoff retry on transient failures.

        Voyage SDK doesn't expose typed errors for transient vs terminal,
        so we retry on any Exception (matches STT pattern). Terminal failure
        propagates to caller, which falls back to status='embed_failed'.
        """
        last_error: Exception | None = None
        for attempt in range(_VOYAGE_MAX_RETRIES):
            try:
                return await asyncio.to_thread(
                    self.voyage.embed, texts, model=self.model, input_type=input_type,
                )
            except Exception as e:
                last_error = e
                if attempt < _VOYAGE_MAX_RETRIES - 1:
                    delay = _VOYAGE_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Voyage embed attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, _VOYAGE_MAX_RETRIES, type(e).__name__, delay,
                    )
                    await asyncio.sleep(delay)
        logger.error(
            "Voyage embed failed after %d attempts: %s",
            _VOYAGE_MAX_RETRIES, type(last_error).__name__,
        )
        raise last_error  # type: ignore[misc]

    async def _embed(self, texts: list[str]) -> list[np.ndarray]:
        result = await self._embed_with_retry(texts, "document")
        return [np.array(e, dtype=np.float32) for e in result.embeddings]

    async def _embed_query(self, text: str) -> np.ndarray:
        result = await self._embed_with_retry([text], "query")
        return np.array(result.embeddings[0], dtype=np.float32)

    _DEDUP_THRESHOLD = 0.92

    async def save(self, content: str, category: str, importance: int = 5,
                   related_person: str | None = None,
                   related_date: str | None = None,
                   source_type: str | None = None,
                   source_ref: str | None = None,
                   confidence: float = 1.0) -> str:
        memory_id = uuid.uuid4().hex[:8]
        embed_failed = False
        try:
            embedding = (await self._embed([content]))[0]
        except Exception as e:
            logger.error("Voyage embed failed (%s), saving with status=embed_failed", type(e).__name__)
            embedding = np.zeros(1024, dtype=np.float32)
            embed_failed = True

        # Dedup: check if a very similar memory already exists in same category.
        # If so, update its importance (keep the higher) instead of duplicating.
        if not embed_failed and np.any(embedding):
            dup = self._find_duplicate(embedding, category)
            if dup:
                new_imp = max(importance, dup["importance"])
                new_conf = max(float(dup.get("confidence") or 0), confidence)
                self.db.execute(
                    "UPDATE memories SET importance = ?, confidence = ?, "
                    "source_type = COALESCE(source_type, ?), "
                    "source_ref = COALESCE(source_ref, ?), "
                    "last_accessed = datetime('now') WHERE id = ?",
                    (new_imp, new_conf, source_type, source_ref, dup["id"]),
                )
                self.db.commit()
                logger.info("Memory dedup: updated %s instead of creating new", dup["id"])
                return dup["id"]

        # Embed-failed rows get status='embed_failed' so they don't pollute
        # search results. reembed.py / manual reembedding can promote them
        # back to 'active' once embeddings are fixed.
        status = "embed_failed" if embed_failed else "active"
        self.db.execute(
            "INSERT INTO memories (id, content, embedding, category, importance, "
            "related_person, related_date, source_type, source_ref, confidence, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (memory_id, content, embedding.tobytes(), category, importance,
             related_person, related_date, source_type, source_ref, confidence, status),
        )
        self.db.commit()
        return memory_id

    @staticmethod
    def _escape_like(s: str) -> str:
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    async def update_by_hint(self, hint: str, new_content: str) -> dict:
        """Replace the content of a single active memory matched by `hint`.

        Returns one of:
        - {"status": "ok", "memory": {id, content, category}}
        - {"status": "not_found"}
        - {"status": "ambiguous", "count": N, "samples": [{id, content}, ...]}
        - {"status": "embed_failed"}  — Voyage down; row left untouched
        - {"status": "invalid"}       — empty hint or empty new_content

        Multiple-match case refuses to update — memories are more sensitive
        than reminder timing, so we force the caller to disambiguate
        rather than guessing which row to overwrite.
        """
        if (not hint or not hint.strip()
                or not new_content or not new_content.strip()):
            return {"status": "invalid"}

        escaped = self._escape_like(hint.strip().lower())
        rows = self.db.execute(
            "SELECT id, content, category, importance, confidence "
            "FROM memories "
            "WHERE status = 'active' AND pylower(content) LIKE ? ESCAPE '\\'",
            (f"%{escaped}%",),
        ).fetchall()

        if not rows:
            return {"status": "not_found"}
        if len(rows) > 1:
            samples = [
                {"id": r["id"], "content": (r["content"] or "")[:120]}
                for r in rows[:3]
            ]
            return {"status": "ambiguous", "count": len(rows), "samples": samples}

        row = rows[0]
        try:
            new_emb = (await self._embed([new_content]))[0]
        except Exception as e:
            logger.error("Voyage embed failed during update (%s) — row %s untouched",
                         type(e).__name__, row["id"])
            return {"status": "embed_failed"}
        if not np.any(new_emb):
            # Defensive: zero-vector means embed silently degraded; same
            # outcome as exception for the caller — don't corrupt the row.
            return {"status": "embed_failed"}

        # Bump confidence to 0.95 — user just explicitly corrected this fact,
        # treat it like a high-confidence signal regardless of prior source.
        new_conf = max(float(row["confidence"] or 0.0), 0.95)
        self.db.execute(
            "UPDATE memories SET content = ?, embedding = ?, "
            "confidence = ?, last_accessed = datetime('now') WHERE id = ?",
            (new_content, new_emb.tobytes(), new_conf, row["id"]),
        )
        self.db.commit()
        logger.info("Memory %s updated via hint", row["id"])
        return {
            "status": "ok",
            "memory": {
                "id": row["id"],
                "content": new_content,
                "category": row["category"],
            },
        }

    def _find_duplicate(self, embedding: np.ndarray, category: str) -> dict | None:
        """Find an existing memory with very high cosine similarity."""
        rows = self.db.execute(
            "SELECT id, embedding, importance, confidence FROM memories "
            "WHERE status = 'active' AND category = ?",
            (category,),
        ).fetchall()
        if not rows:
            return None

        emb_matrix = np.stack([
            np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
        ])
        norms = np.linalg.norm(emb_matrix, axis=1)
        emb_norm = np.linalg.norm(embedding)
        safe_denom = norms * emb_norm
        safe_denom[safe_denom < 1e-8] = 1.0
        sims = emb_matrix @ embedding / safe_denom

        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self._DEDUP_THRESHOLD:
            return {"id": rows[best_idx]["id"],
                    "importance": rows[best_idx]["importance"],
                    "confidence": rows[best_idx]["confidence"]}
        return None

    @staticmethod
    def _match_reason(score: float) -> str:
        if score >= 0.85:
            return "почти точное совпадение по смыслу"
        if score >= 0.65:
            return "сильное смысловое совпадение"
        if score >= 0.45:
            return "похожая тема"
        return "поднято важностью и общим контекстом"

    async def search(self, query: str, top_k: int = 8,
                     category: str | None = None,
                     min_importance: int = 0) -> list[dict]:
        try:
            query_emb = await self._embed_query(query)
        except Exception as e:
            logger.error("Voyage embed_query failed: %s", type(e).__name__)
            return []

        where = "WHERE status = 'active'"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)
        if min_importance > 0:
            where += " AND importance >= ?"
            params.append(min_importance)

        rows = self.db.execute(
            f"SELECT id, content, embedding, category, importance, "
            f"related_person, related_date, source_type, source_ref, confidence, "
            f"created_at, last_accessed "
            f"FROM memories {where}",
            params,
        ).fetchall()

        if not rows:
            return []

        # Vectorized cosine similarity: build matrix of all embeddings,
        # compute all scores in one numpy operation instead of per-row loop.
        emb_matrix = np.stack([
            np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
        ])
        norms = np.linalg.norm(emb_matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)

        # Avoid division by zero for zero-vector fallback embeddings
        safe_denom = norms * query_norm
        safe_denom[safe_denom < 1e-8] = 1.0
        cosine_scores = emb_matrix @ query_emb / safe_denom

        # Compute final scores incorporating importance weight
        importances = np.array([row["importance"] for row in rows], dtype=np.float32)
        final_scores = cosine_scores * self.relevance_w + (importances / 10) * self.importance_w

        # Recency decay: memories not accessed recently get dampened.
        # Uses last_accessed (or created_at as fallback).
        # Decay from 1.0 (just accessed) to 0.5 floor over ~90 days.
        # Both sides are UTC-naive: stored via SQLite `datetime('now')`
        # (UTC), so compute `now` as UTC-naive too. datetime.now() would
        # drift by the system TZ offset on non-UTC dev machines.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        decay_half_life = 90.0  # days
        for i, row in enumerate(rows):
            ref = row["last_accessed"] or row["created_at"]
            try:
                ref_dt = datetime.fromisoformat(ref.replace(" ", "T") if ref else "")
                days_old = max(0.0, (now - ref_dt).total_seconds() / 86400)
            except (ValueError, TypeError):
                days_old = 0.0
            final_scores[i] *= 0.5 + 0.5 * math.exp(-days_old / decay_half_life)

        # Get top-k indices without full sort (O(n) partial sort vs O(n log n))
        k = min(top_k, len(rows))
        top_indices = np.argpartition(final_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

        top = []
        for idx in top_indices:
            row = rows[idx]
            top.append({
                "id": row["id"],
                "content": row["content"],
                "score": float(cosine_scores[idx]),
                "category": row["category"],
                "importance": row["importance"],
                "related_person": row["related_person"],
                "related_date": row["related_date"],
                "source_type": row["source_type"],
                "source_ref": row["source_ref"],
                "confidence": float(row["confidence"] or 0.0),
                "created_at": row["created_at"],
                "last_accessed": row["last_accessed"],
                "final_score": float(final_scores[idx]),
                "match_reason": self._match_reason(float(cosine_scores[idx])),
            })

        if top:
            ids = [r["id"] for r in top]
            placeholders = ",".join("?" * len(ids))
            self.db.execute(
                f"UPDATE memories SET last_accessed = datetime('now') "
                f"WHERE id IN ({placeholders})",
                ids,
            )
            self.db.commit()

        return top

    def list_recent(self, limit: int = 8, category: str | None = None) -> list[dict]:
        where = "WHERE status = 'active'"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)
        rows = self.db.execute(
            f"SELECT id, content, category, importance, related_person, related_date, "
            f"source_type, source_ref, confidence, created_at, last_accessed "
            f"FROM memories {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_category(self, category: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, content, category, importance, related_person, related_date, "
            "source_type, source_ref, confidence, created_at, last_accessed "
            "FROM memories WHERE category = ? AND status = 'active' "
            "ORDER BY importance DESC",
            (category,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, memory_id: str):
        self.db.execute(
            "UPDATE memories SET status = 'deleted', "
            "last_accessed = datetime('now') WHERE id = ?",
            (memory_id,),
        )
        self.db.commit()

    def get_last_deleted(self) -> dict | None:
        """Get the most recently deleted memory (for /undo)."""
        row = self.db.execute(
            "SELECT id, content, category FROM memories "
            "WHERE status = 'deleted' ORDER BY last_accessed DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def restore(self, memory_id: str) -> bool:
        """Restore a deleted memory back to active."""
        cur = self.db.execute(
            "UPDATE memories SET status = 'active' WHERE id = ? AND status = 'deleted'",
            (memory_id,),
        )
        self.db.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()
        return row[0]
