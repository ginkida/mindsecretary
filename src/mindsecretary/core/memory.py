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

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    async def save(self, content: str, category: str, importance: int = 5,
                   related_person: str | None = None,
                   related_date: str | None = None) -> str:
        memory_id = uuid.uuid4().hex[:8]
        try:
            embedding = (await self._embed([content]))[0]
        except Exception as e:
            logger.error("Voyage embed failed: %s", e)
            raise

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

        results = []
        for row in rows:
            raw = row[2] if isinstance(row, (list, tuple)) else row["embedding"]
            emb = np.frombuffer(raw, dtype=np.float32)
            score = self._cosine_sim(query_emb, emb)

            def _get(key, idx):
                return row[idx] if isinstance(row, (list, tuple)) else row[key]

            results.append({
                "id": _get("id", 0),
                "content": _get("content", 1),
                "score": score,
                "category": _get("category", 3),
                "importance": _get("importance", 4),
                "related_person": _get("related_person", 5),
                "related_date": _get("related_date", 6),
                "created_at": _get("created_at", 7),
            })

        for r in results:
            r["final_score"] = (
                r["score"] * self.relevance_w
                + (r["importance"] / 10) * self.importance_w
            )

        results.sort(key=lambda x: x["final_score"], reverse=True)
        top = results[:top_k]

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
        return [
            {"id": r[0], "content": r[1], "category": r[2],
             "importance": r[3], "related_person": r[4], "related_date": r[5]}
            for r in rows
        ]

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
