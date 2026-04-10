from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from ..llm.prompts import MAIN_SYSTEM_PROMPT
from ..llm.router import ModelRouter
from ..llm.tools import TOOL_DEFINITIONS, ToolExecutor
from .config import Profile, Settings
from .database import Database
from .memory import Memory

logger = logging.getLogger(__name__)

DAYS_RU = {
    0: "Понедельник", 1: "Вторник", 2: "Среда", 3: "Четверг",
    4: "Пятница", 5: "Суббота", 6: "Воскресенье",
}


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
                logger.error("LLM call failed on round %d: %s", _round, e)
                if not final_text:
                    final_text = "Ошибка при обращении к LLM. Попробуй ещё раз."
                break

            inp = response.usage.get("input_tokens", 0)
            outp = response.usage.get("output_tokens", 0)
            total_tokens += inp + outp
            self.db.log_cost("anthropic", input_tokens=inp, output_tokens=outp)

            if not response.tool_calls:
                final_text = response.text or ""
                break

            # Preserve any text from this round
            if response.text:
                final_text = response.text

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

    @staticmethod
    def _sanitize_for_context(text: str, max_len: int = 500) -> str:
        """Sanitize user-origin text before injecting into system prompt.

        Mitigates prompt injection by stripping instruction-like patterns.
        """
        text = text[:max_len]
        # Remove patterns that look like system instructions
        for prefix in ("## ", "# ", "System:", "SYSTEM:", "Instructions:",
                        "You are", "You must", "Ignore previous", "Forget"):
            text = text.replace(prefix, f"[{prefix.strip()}]")
        return text

    async def _build_system_prompt(self, user_message: str) -> str:
        now = datetime.now()

        memories = await self.memory.search(user_message, top_k=self.settings.memory_top_k)
        memory_text = "\n".join(
            f"- [{m['category']}] {self._sanitize_for_context(m['content'])}"
            for m in memories
        ) or "Пока ничего не запомнил."

        today_str = now.strftime("%Y-%m-%d")
        events = self.db.get_events(today_str)
        events_text = "\n".join(
            f"- {e['start_at'][11:16] if len(e['start_at']) > 10 else '??:??'} "
            f"{self._sanitize_for_context(e['title'], 200)}"
            + (f" ({self._sanitize_for_context(e['related_person'] or '', 100)})"
               if e.get("related_person") else "")
            for e in events
        ) or "Нет событий."

        recent = self.db.get_recent_messages(limit=6)
        recent_text = "\n".join(
            f"{'Ты' if m['direction'] == 'in' else 'Бот'}: "
            f"{self._sanitize_for_context(m['content'], 200)}"
            for m in recent
        ) or "Начало разговора."

        return MAIN_SYSTEM_PROMPT.format(
            name=self.profile.name,
            profile=self.profile.to_yaml_str(),
            date=now.strftime("%Y-%m-%d"),
            day_of_week=DAYS_RU[now.weekday()],
            time=now.strftime("%H:%M"),
            memories=memory_text,
            today_events=events_text,
            recent_messages=recent_text,
        )

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
