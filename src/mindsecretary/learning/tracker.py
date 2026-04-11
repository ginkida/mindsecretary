from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ..core.database import Database

logger = logging.getLogger(__name__)


class FeedbackTracker:
    """Track implicit and explicit feedback on bot messages."""

    def __init__(self, db: Database):
        self.db = db

    def mark_positive(self, interaction_id: str):
        self.db.db.execute(
            "UPDATE interactions SET feedback = 'positive', feedback_at = datetime('now') "
            "WHERE id = ?", (interaction_id,),
        )
        self.db.db.commit()

    def mark_negative(self, interaction_id: str):
        self.db.db.execute(
            "UPDATE interactions SET feedback = 'negative', feedback_at = datetime('now') "
            "WHERE id = ?", (interaction_id,),
        )
        self.db.db.commit()

    def mark_read(self, interaction_id: str):
        self.db.db.execute(
            "UPDATE interactions SET read_at = datetime('now') WHERE id = ?",
            (interaction_id,),
        )
        self.db.db.commit()

    def record_response_time(self, interaction_id: str, seconds: float):
        self.db.db.execute(
            "UPDATE interactions SET response_time_sec = ? WHERE id = ?",
            (seconds, interaction_id),
        )
        self.db.db.commit()

    def get_feedback_summary(self, days: int = 7) -> dict:
        """Summarize feedback over last N days."""
        since = datetime.now().replace(hour=0, minute=0, second=0)
        since = since - timedelta(days=days)
        # Match SQL datetime('now') format (space-separated); see
        # Database._SQL_TS_FMT docstring for why Python isoformat breaks here.
        since_sql = since.strftime("%Y-%m-%d %H:%M:%S")

        rows = self.db.db.execute(
            "SELECT feedback, COUNT(*) FROM interactions "
            "WHERE direction = 'out' AND timestamp >= ? "
            "GROUP BY feedback",
            (since_sql,),
        ).fetchall()

        summary = {"positive": 0, "negative": 0, "ignored": 0, "total_out": 0}
        for fb, count in rows:
            if fb == "positive":
                summary["positive"] = count
            elif fb == "negative":
                summary["negative"] = count
            elif fb is None:
                summary["ignored"] = count
            summary["total_out"] += count

        # Average response time
        row = self.db.db.execute(
            "SELECT AVG(response_time_sec) FROM interactions "
            "WHERE direction = 'in' AND response_time_sec IS NOT NULL "
            "AND timestamp >= ?",
            (since_sql,),
        ).fetchone()
        summary["avg_response_time"] = row[0] if row[0] else None

        return summary
