from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from ..core import fmt_local_time, tz_now
from ..core.config import Profile
from ..core.database import Database
from ..core.memory import Memory
from ..core.prompt_safety import sanitize_for_context
from ..llm.prompts import WEEKLY_SYSTEM_PROMPT
from ..llm.client import LLMClient
from .patterns import PatternAnalyzer

logger = logging.getLogger(__name__)


class WeeklyReflection:
    """Analyze the week's interactions and generate learnings + review."""

    def __init__(self, llm: LLMClient, memory: Memory, db: Database,
                 profile: Profile):
        self.llm = llm
        self.memory = memory
        self.db = db
        self.profile = profile

    async def generate_weekly_review(self) -> str | None:
        """Generate weekly review and extract learnings."""
        # Two clocks, on purpose:
        # - `now` / `week_ago` are UTC-naive, matching SQL timestamps stored
        #   via `datetime('now')` — used for range queries below.
        # - `period` is rendered in profile TZ so the user sees their local
        #   calendar dates, not whatever the system clock says.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        week_ago = now - timedelta(days=7)
        local_now = tz_now(self.profile.timezone)
        local_week_ago = local_now - timedelta(days=7)
        period = (
            f"{local_week_ago.strftime('%Y-%m-%d')} — "
            f"{local_now.strftime('%Y-%m-%d')}"
        )

        # Gather data
        interactions = self.db.get_interactions(since=week_ago, limit=200)
        if len(interactions) < 3:
            logger.info("Too few interactions for weekly review (%d)", len(interactions))
            return None

        # Events / habits index by local date (they store start_at / date
        # columns in the user's clock), so pull the week window from the
        # local calendar — otherwise a late-evening UTC run misses the
        # most recent local day at the boundary.
        events = self.db.get_events(
            date_from=local_week_ago.strftime("%Y-%m-%d"),
            date_to=local_now.strftime("%Y-%m-%d"),
        )

        # Habits
        habits_data = self._get_habits_summary(
            local_week_ago.replace(tzinfo=None), local_now.replace(tzinfo=None),
        )

        # Current learnings
        learnings = self.memory.get_by_category("learning")

        # Format interactions (compact)
        s = sanitize_for_context
        interactions_text = self._format_interactions(interactions)
        events_text = "\n".join(
            f"- {e['start_at'][:16]} {s(e['title'], 200)}"
            + (f" (с {s(e['related_person'], 100)})" if e.get("related_person") else "")
            for e in events
        ) or "Нет событий."

        learnings_text = "\n".join(
            f"- {s(l['content'])}" for l in learnings
        ) or "Нет предыдущих learnings."

        habits_text = habits_data or "Привычки не отслеживаются."

        # Deterministic pattern signals — gives the LLM hard numbers to cite
        # instead of inventing statistics from the raw interaction list.
        try:
            patterns_text = PatternAnalyzer(self.db, self.profile).format_for_prompt()
        except Exception as e:
            logger.warning("Pattern analyzer failed: %s", type(e).__name__)
            patterns_text = "Недостаточно данных для автоматических наблюдений."

        prompt = WEEKLY_SYSTEM_PROMPT.format(
            name=self.profile.name,
            period=period,
            interactions=interactions_text,
            events=events_text,
            habits=habits_text,
            learnings=learnings_text,
            patterns=patterns_text,
        )

        try:
            # Claude Sonnet — weekly analysis
            response = await self.llm.chat(
                system=prompt,
                messages=[{"role": "user", "content": "Проанализируй неделю."}],
                max_tokens=2000,


            )
            text = response.text or ""
        except Exception as e:
            logger.error("Weekly review LLM failed: %s", type(e).__name__)
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
        tz = self.profile.timezone
        for i in interactions:
            raw_ts = i.get("timestamp") or ""
            # Convert UTC-stored timestamp to the user's local clock.
            # fmt_local_time returns MM-DD HH:MM when the row is from a
            # different local day, and HH:MM when today — which is the
            # format we want for a week-spanning list either way.
            local_stamp = fmt_local_time(raw_ts, tz)
            direction = "→" if i["direction"] == "out" else "←"
            msg_type = i.get("message_type", "?")
            content = sanitize_for_context(i.get("content") or "", 120)
            fb = f" [{i['feedback']}]" if i.get("feedback") else ""
            lines.append(f"{local_stamp} {direction} ({msg_type}{fb}) {content}")
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
            name, total, done = sanitize_for_context(r[0] or "", 100), r[1], r[2] or 0
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
            content = learning["content"]
            confidence = learning.get("confidence", 0.5)
            importance = max(1, min(10, int(confidence * 10)))
            try:
                # Check if a similar learning already exists
                existing = await self.memory.search(
                    content, top_k=1, category="learning",
                )
                if existing and existing[0].get("score", 0) > 0.85:
                    logger.info("Skipping duplicate learning: %.2f sim with '%s'",
                                existing[0]["score"], existing[0]["content"][:60])
                    continue
                await self.memory.save(
                    content=content,
                    category="learning",
                    importance=importance,
                )
                saved += 1
            except Exception as e:
                logger.error("Failed to save learning: %s", type(e).__name__)

        if saved:
            logger.info("Saved %d new learnings from weekly review", saved)
