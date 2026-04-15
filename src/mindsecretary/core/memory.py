from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid

import numpy as np
import voyageai

logger = logging.getLogger(__name__)


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
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                category TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                related_person TEXT,
                related_date TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_mem_cat ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
            CREATE INDEX IF NOT EXISTS idx_mem_person ON memories(related_person);
        """)
        self.db.commit()

    async def _embed(self, texts: list[str]) -> list[np.ndarray]:
        result = await asyncio.to_thread(
            self.voyage.embed, texts, model=self.model, input_type="document",
        )
        return [np.array(e, dtype=np.float32) for e in result.embeddings]

    async def _embed_query(self, text: str) -> np.ndarray:
        result = await asyncio.to_thread(
            self.voyage.embed, [text], model=self.model, input_type="query",
        )
        return np.array(result.embeddings[0], dtype=np.float32)

    async def save(self, content: str, category: str, importance: int = 5,
                   related_person: str | None = None,
                   related_date: str | None = None) -> str:
        memory_id = uuid.uuid4().hex[:8]
        try:
            embedding = (await self._embed([content]))[0]
        except Exception as e:
            logger.error("Voyage embed failed, saving with zero vector: %s", e)
            # Store with zero embedding so the memory isn't lost.
            # It won't appear in similarity search but is preserved in DB.
            embedding = np.zeros(1024, dtype=np.float32)

        self.db.execute(
            "INSERT INTO memories (id, content, embedding, category, importance, "
            "related_person, related_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, content, embedding.tobytes(), category, importance,
             related_person, related_date),
        )
        self.db.commit()
        return memory_id

    async def search(self, query: str, top_k: int = 8,
                     category: str | None = None,
                     min_importance: int = 0) -> list[dict]:
        try:
            query_emb = await self._embed_query(query)
        except Exception as e:
            logger.error("Voyage embed_query failed: %s", e)
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
            f"related_person, related_date, created_at "
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
                "created_at": row["created_at"],
                "final_score": float(final_scores[idx]),
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

    def get_by_category(self, category: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, content, category, importance, related_person, related_date "
            "FROM memories WHERE category = ? AND status = 'active' "
            "ORDER BY importance DESC",
            (category,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, memory_id: str):
        self.db.execute(
            "UPDATE memories SET status = 'deleted' WHERE id = ?", (memory_id,)
        )
        self.db.commit()

    def count(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()
        return row[0]
