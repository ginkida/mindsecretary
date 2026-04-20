from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from ..learning.mood import check_contact_frequency, get_mood_trend
from ..llm.prompts import MAIN_SYSTEM_PROMPT
from ..llm.router import ModelRouter
from ..llm.tools import TOOL_DEFINITIONS, ToolExecutor
from . import DAYS_RU, tz_now
from .config import Profile, Settings
from .database import Database
from .memory import Memory
from .prompt_safety import sanitize_for_context

logger = logging.getLogger(__name__)

# Hard cap on tool calls LLM can request in a single round. Defends against
# runaway LLM output (bug, prompt injection, model misbehavior) burning
# embedding budget. max_tool_rounds * MAX_TOOLS_PER_ROUND is the worst case.
MAX_TOOLS_PER_ROUND = 10


@dataclass
class BrainResponse:
    text: str
    tool_calls_made: int
    total_tokens: int


class Brain:
    def __init__(self, router: ModelRouter, memory: Memory, db: Database,
                 profile: Profile, settings: Settings):
        self.router = router
        self.memory = memory
        self.db = db
        self.profile = profile
        self.settings = settings
        self.tool_executor = ToolExecutor(db, memory)

    async def process(self, user_message: str, message_type: str = "text",
                      metadata: dict | None = None,
                      image_base64: str | None = None) -> BrainResponse:
        self.db.log_interaction(
            direction="in",
            message_type=message_type,
            content=user_message,
            voice_duration_sec=metadata.get("duration_sec") if metadata else None,
            metadata=metadata,
        )

        # Cost circuit breaker — refuse LLM work if daily spend exceeded
        today_cost = self.db.get_today_cost()
        limit = self.settings.daily_cost_limit_usd
        if today_cost >= limit:
            msg = (
                f"⚠️ Дневной лимит API расходов исчерпан "
                f"(${today_cost:.2f} / ${limit:.2f}). Попробуй завтра, или "
                f"увеличь `daily_cost_limit_usd` в config/settings.yaml."
            )
            logger.warning("Cost limit hit: $%.2f / $%.2f", today_cost, limit)
            self.db.log_interaction(
                direction="out", message_type="chat", content=msg,
                metadata={"cost_limit_hit": True},
            )
            return BrainResponse(text=msg, tool_calls_made=0, total_tokens=0)

        system_prompt = await self._build_system_prompt(user_message)

        # Build user message — text or multimodal (text + image)
        if image_base64:
            user_content = [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                }},
            ]
        else:
            user_content = user_message
        messages = [{"role": "user", "content": user_content}]
        total_tools = 0
        total_tokens = 0
        final_text = ""

        for _round in range(self.settings.max_tool_rounds):
            try:
                response = await self.router.chat(
                    system=system_prompt,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    max_tokens=self.settings.max_tokens,
                )
            except Exception as e:
                logger.error("LLM call failed on round %d: %s", _round, type(e).__name__)
                if not final_text:
                    final_text = "Ошибка при обращении к LLM. Попробуй ещё раз."
                break

            inp = response.usage.get("input_tokens", 0)
            outp = response.usage.get("output_tokens", 0)
            total_tokens += inp + outp
            self.db.log_cost("anthropic", input_tokens=inp, output_tokens=outp)

            # Keep the latest non-empty text across rounds. This prevents empty
            # text in a tool-only round from erasing the warm reply that was
            # generated alongside earlier tool calls.
            if response.text and response.text.strip():
                final_text = response.text

            if not response.tool_calls:
                break

            if len(response.tool_calls) > MAX_TOOLS_PER_ROUND:
                logger.warning(
                    "LLM requested %d tool calls in round %d, capping at %d",
                    len(response.tool_calls), _round, MAX_TOOLS_PER_ROUND,
                )
                response.tool_calls = response.tool_calls[:MAX_TOOLS_PER_ROUND]

            # Build assistant message with tool calls (OpenAI format)
            assistant_msg = self._build_assistant_msg(response)
            messages.append(assistant_msg)

            # Execute each tool and append result
            for tc in response.tool_calls:
                total_tools += 1
                result = await self.tool_executor.execute(tc["name"], tc["arguments"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{tc['name']}"),
                    "content": result,
                })
        else:
            # Loop exhausted without break — too many tool rounds
            if not final_text:
                final_text = "Превышен лимит вызовов инструментов."

        self.db.log_interaction(
            direction="out",
            message_type="chat",
            content=final_text,
            metadata={"tool_calls": total_tools, "tokens": total_tokens},
        )

        return BrainResponse(
            text=final_text,
            tool_calls_made=total_tools,
            total_tokens=total_tokens,
        )

    async def _build_system_prompt(self, user_message: str) -> str:
        now = tz_now(self.profile.timezone)
        s = sanitize_for_context

        return MAIN_SYSTEM_PROMPT.format(
            name=self.profile.name,
            profile=self.profile.to_yaml_str(),
            date=now.strftime("%Y-%m-%d"),
            day_of_week=DAYS_RU[now.weekday()],
            time=now.strftime("%H:%M"),
            memories=await self._section_memories(user_message, s),
            today_events=self._section_events(now, s),
            today_goals=self._section_goals(s),
            recent_messages=self._section_recent(s),
            pending_decisions=self._section_decisions(s),
            mood_trend=self._section_mood_trend(),
            theme_clusters=self._section_theme_clusters(s),
            quiet_contacts=self._section_quiet_contacts(s),
            birthdays=self._section_birthdays(now, s),
            style=self.profile.style,
        )

    async def _section_memories(self, user_message: str, s) -> str:
        memories = await self.memory.search(user_message, top_k=self.settings.memory_top_k)
        return "\n".join(
            f"- [{m['category']}] {s(m['content'])}" for m in memories
        ) or "Пока ничего не запомнил."

    def _section_events(self, now: datetime, s) -> str:
        events = self.db.get_events(now.strftime("%Y-%m-%d"))
        return "\n".join(
            f"- {e['start_at'][11:16] if len(e['start_at']) > 10 else '??:??'} "
            f"{s(e['title'], 200)}"
            + (f" ({s(e['related_person'] or '', 100)})" if e.get("related_person") else "")
            for e in events
        ) or "Нет событий."

    def _section_recent(self, s) -> str:
        recent = self.db.get_recent_messages(limit=8)
        return "\n".join(
            f"{'Ты' if m['direction'] == 'in' else 'Бот'}: {s(m['content'], 200)}"
            for m in recent
        ) or "Начало разговора."

    def _section_decisions(self, s) -> str:
        decisions = self.db.get_pending_decisions(limit=5)
        return "\n".join(
            f"- {s(d['description'], 200)}" for d in decisions
        ) or "Нет решений в процессе."

    def _section_mood_trend(self) -> str:
        try:
            trend = get_mood_trend(self.db, days=3)
            return ", ".join(
                f"{m['date'][-5:]}: {m['label']}" for m in trend
            ) or "Нет данных."
        except Exception as e:
            logger.warning("Section mood_trend failed: %s", type(e).__name__)
            return "Нет данных."

    def _section_theme_clusters(self, s) -> str:
        try:
            clusters = self.db.get_theme_clusters(days=30, limit=5)
            return ", ".join(
                f"{s(c['label'], 60)} ({c['count']})" for c in clusters
            ) or "Нет заметных тем."
        except Exception as e:
            logger.warning("Section theme_clusters failed: %s", type(e).__name__)
            return "Нет данных."

    def _section_quiet_contacts(self, s) -> str:
        try:
            alerts = check_contact_frequency(self.db)
            filtered = [
                a for a in alerts
                if a.get("days_since", 0) > self.settings.quiet_contact_days
                and a.get("mention_count", 0) >= self.settings.quiet_contact_min_mentions
            ][:2]
            return "\n".join(
                f"- {s(a['name'], 60)}"
                + (f" ({s(a.get('relation', ''), 40)})" if a.get("relation") else "")
                + f": не общались {a['days_since']} дней"
                for a in filtered
            ) or "Нет тревог."
        except Exception as e:
            logger.warning("Section quiet_contacts failed: %s", type(e).__name__)
            return "Нет данных."

    def _section_goals(self, s) -> str:
        try:
            goals = self.db.get_daily_goals()
            if not goals:
                return "Не поставлены."
            lines = []
            for g in goals:
                emoji = {"pending": "⬜", "completed": "✅",
                         "skipped": "⏭", "partial": "🟡"}.get(g["status"], "⬜")
                prio = {"high": "!"}.get(g.get("priority", ""), "")
                line = f"- {emoji} {s(g['title'], 150)}"
                if prio:
                    line += f" ({prio})"
                lines.append(line)
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Section goals failed: %s", type(e).__name__)
            return "Нет данных."

    def _section_birthdays(self, now: datetime, s) -> str:
        try:
            upcoming = self.db.get_upcoming_birthdays(days=3)
            today_md = now.strftime("%m-%d")
            lines = []
            for c in upcoming[:3]:
                bday = c.get("birthday") or ""
                bday_md = bday[-5:] if len(bday) >= 5 else bday
                name = s(c["name"], 60)
                relation = s(c.get("relation") or "", 40)
                rel_str = f" ({relation})" if relation else ""
                if bday_md == today_md:
                    lines.append(f"- сегодня: {name}{rel_str}")
                else:
                    lines.append(f"- {bday}: {name}{rel_str}")
            return "\n".join(lines) or "Нет ближайших."
        except Exception as e:
            logger.warning("Section birthdays failed: %s", type(e).__name__)
            return "Нет данных."

    @staticmethod
    def _build_assistant_msg(response) -> dict:
        """Build OpenAI-format assistant message with tool calls."""
        tool_calls = []
        for tc in response.tool_calls:
            tool_calls.append({
                "id": tc.get("id", f"call_{tc['name']}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            })
        msg: dict = {"role": "assistant", "content": response.text or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg
