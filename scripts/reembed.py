#!/usr/bin/env python3
"""Re-embed memories saved with zero-vector fallback.

When the Voyage API was down, memories were stored with a zero embedding.
This script finds them and re-embeds so they become searchable again.

Usage:
    python scripts/reembed.py [--db PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np


def find_zero_vectors(db: sqlite3.Connection) -> list[dict]:
    """Find memories whose embedding is all zeros."""
    rows = db.execute(
        "SELECT id, content, category FROM memories WHERE status = 'active'"
    ).fetchall()
    zero = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        if np.allclose(emb, 0):
            zero.append({"id": row["id"], "content": row["content"],
                         "category": row["category"]})
    return zero


def reembed(db_path: Path, dry_run: bool = False):
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    zero_mems = find_zero_vectors(db)
    if not zero_mems:
        print("No zero-vector memories found.")
        return 0

    print(f"Found {len(zero_mems)} memories with zero embeddings:")
    for m in zero_mems:
        print(f"  [{m['id']}] ({m['category']}) {m['content'][:80]}")

    if dry_run:
        print("\n--dry-run: no changes made.")
        return 0

    import os

    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("Error: VOYAGE_API_KEY not set.", file=sys.stderr)
        return 1

    model = os.environ.get("VOYAGE_MODEL", "voyage-3")
    client = voyageai.Client(api_key=api_key)

    # Batch embed (Voyage supports up to 128 texts per call)
    texts = [m["content"] for m in zero_mems]
    batch_size = 64
    updated = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_mems = zero_mems[i:i + batch_size]
        print(f"\nEmbedding batch {i // batch_size + 1} ({len(batch)} texts)...")

        result = client.embed(batch, model=model, input_type="document")
        for mem, emb_list in zip(batch_mems, result.embeddings):
            emb = np.array(emb_list, dtype=np.float32)
            db.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (emb.tobytes(), mem["id"]),
            )
            updated += 1

    db.commit()
    db.close()
    print(f"\nDone: {updated} memories re-embedded.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Re-embed zero-vector memories")
    parser.add_argument("--db", default="data/mindsecretary.db",
                        help="Path to SQLite database")
    parser.add_argument("--dry-run", action="store_true",
                        help="List zero-vector memories without re-embedding")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    sys.exit(reembed(db_path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
