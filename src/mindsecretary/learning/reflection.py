from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from ..core.config import Profile
from ..core.database import Database
from ..core.memory import Memory
from ..llm.prompts import WEEKLY_SYSTEM_PROMPT
from ..llm.router import ModelRouter
from .tracker import FeedbackTracker

logger = logging.getLogger(__name__)


class WeeklyReflection:
    """Analyze the week's interactions and generate learnings + review."""

    def __init__(self, router: ModelRouter, memory: Memory, db: Database,
                 profile: Profile):
        self.router = router
        self.memory = memory
        self.db = db
        self.profile = profile
        self.tracker = FeedbackTracker(db)

    async def generate_weekly_review(self) -> str | None:
        """Generate weekly review and extract learnings."""
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        period = f"{week_ago.strftime('%Y-%m-%d')} — {now.strftime('%Y-%m-%d')}"

        # Gather data
        interactions = self.db.get_interactions(since=week_ago, limit=200)
        if len(interactions) < 3:
            logger.info("Too few interactions for weekly review (%d)", len(interactions))
            return None

        events = self.db.get_events(
            date_from=week_ago.strftime("%Y-%m-%d"),
            date_to=now.strftime("%Y-%m-%d"),
        )

        # Habits
        habits_data = self._get_habits_summary(week_ago, now)

        # Current learnings
        learnings = self.memory.get_by_category("learning")

        # Feedback summary
        feedback = self.tracker.get_feedback_summary(days=7)

        # Format interactions (compact)
        interactions_text = self._format_interactions(interactions)
        events_text = "\n".join(
            f"- {e['start_at'][:16]} {e['title']}"
            + (f" (с {e['related_person']})" if e.get("related_person") else "")
            for e in events
        ) or "Нет событий."

        learnings_text = "\n".join(
            f"- {l['content']}" for l in learnings
        ) or "Нет предыдущих learnings."

        habits_text = habits_data or "Привычки не отслеживаются."

        prompt = WEEKLY_SYSTEM_PROMPT.format(
            name=self.profile.name,
            period=period,
            interactions=interactions_text,
            events=events_text,
            habits=habits_text,
            learnings=learnings_text,
        )

        try:
            # Claude Sonnet — weekly analysis
            response = await self.router.chat(
                system=prompt,
                messages=[{"role": "user", "content": "Проанализируй неделю."}],
                max_tokens=2000,


            )
            text = response.text or ""
        except Exception as e:
            logger.error("Weekly review LLM failed: %s", e)
            return None

        # Extract and save learnings from JSON block at the end
        await self._extract_learnings(text)

        # Log as interaction
        self.db.log_interaction(
            direction="out", message_type="weekly_review", content=text,
        )

        return text

    def _format_interactions(self, interactions: list[dict]) -> str:
        lines = []
        for i in interactions:
            ts = i["timestamp"][11:16] if i["timestamp"] and len(i["timestamp"]) > 11 else "??:??"
            day = i["timestamp"][:10] if i["timestamp"] else "?"
            direction = "→" if i["direction"] == "out" else "←"
            msg_type = i.get("message_type", "?")
            content = (i.get("content") or "")[:120]
            fb = f" [{i['feedback']}]" if i.get("feedback") else ""
            lines.append(f"{day} {ts} {direction} ({msg_type}{fb}) {content}")
        return "\n".join(lines[-100:])  # last 100

    def _get_habits_summary(self, since: datetime, until: datetime) -> str:
        rows = self.db.db.execute(
            "SELECT h.name, COUNT(hl.done) as total, SUM(hl.done) as done "
            "FROM habits h LEFT JOIN habit_log hl ON h.id = hl.habit_id "
            "AND hl.date >= ? AND hl.date <= ? "
            "GROUP BY h.id",
            (since.strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d")),
        ).fetchall()
        if not rows:
            return ""
        lines = []
        for r in rows:
            name, total, done = r[0], r[1], r[2] or 0
            lines.append(f"- {name}: {done}/{total}")
        return "\n".join(lines)

    async def _extract_learnings(self, text: str):
        """Extract JSON learnings block from the response and save to memory."""
        # Find JSON array in the response
        import re
        json_match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
        if not json_match:
            # Try without code fence
            json_match = re.search(r"\[[\s\S]*\"content\"[\s\S]*?\]", text)
        if not json_match:
            logger.info("No learnings JSON found in weekly review")
            return

        try:
            learnings = json.loads(json_match.group(1) if json_match.lastindex else json_match.group(0))
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse learnings JSON")
            return

        saved = 0
        for learning in learnings:
            if not isinstance(learning, dict) or "content" not in learning:
                continue
            confidence = learning.get("confidence", 0.5)
            importance = max(1, min(10, int(confidence * 10)))
            try:
                await self.memory.save(
                    content=learning["content"],
                    category="learning",
                    importance=importance,
                )
                saved += 1
            except Exception as e:
                logger.error("Failed to save learning: %s", e)

        if saved:
            logger.info("Saved %d new learnings from weekly review", saved)
