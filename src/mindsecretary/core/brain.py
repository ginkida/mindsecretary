from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..learning.mood import check_contact_frequency, get_mood_trend
from ..llm.prompts import MAIN_SYSTEM_PROMPT
from ..llm.client import LLMClient
from ..llm.tools import TOOL_DEFINITIONS, ToolExecutor
from . import DAYS_RU, fmt_local_time, tz_now
from .config import Profile, Settings
from .database import Database
from .memory import Memory
from .prompt_safety import sanitize_for_context

logger = logging.getLogger(__name__)

# Hard cap on tool calls LLM can request in a single round. Defends against
# runaway LLM output (bug, prompt injection, model misbehavior) burning
# embedding budget. max_tool_rounds * MAX_TOOLS_PER_ROUND is the worst case.
MAX_TOOLS_PER_ROUND = 10
MAX_TOOL_RESULT_LEN = 4000

# How many past interactions (user msgs + bot replies + proactive sends)
# to replay into each LLM call as real role=user/assistant turns. This is
# the core of Claude's conversational continuity — older context is reachable
# via the search_conversations tool.
CONVERSATION_HISTORY_TURNS = 20

# Per-turn content cap for replayed history. Keeps prompt cost bounded while
# still giving Claude enough to understand each prior turn. Sanitized for
# injection defense since past content may include user text.
HISTORY_TURN_CHAR_CAP = 800


@dataclass
class BrainResponse:
    text: str
    tool_calls_made: int
    total_tokens: int


class Brain:
    def __init__(self, llm: LLMClient, memory: Memory, db: Database,
                 profile: Profile, settings: Settings):
        self.llm = llm
        self.memory = memory
        self.db = db
        self.profile = profile
        self.settings = settings
        self.tool_executor = ToolExecutor(db, memory)

    async def process(self, user_message: str, message_type: str = "text",
                      metadata: dict | None = None,
                      image_base64: str | None = None) -> BrainResponse:
        # Replay history as real multi-turn BEFORE logging the current message,
        # so the freshly-logged user turn isn't duplicated (history + current).
        history_turns = self._build_history_turns()

        interaction_id = self.db.log_interaction(
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

        system_prompt = await self._build_system_prompt(user_message, message_type)

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
        # Prepend replayed history. Merge pass guarantees no two same-role
        # turns appear back-to-back (Anthropic API requires alternation) —
        # consecutive notifications collapse into a single assistant turn.
        messages = self._merge_consecutive(
            history_turns + [{"role": "user", "content": user_content}]
        )
        total_tools = 0
        total_tokens = 0
        final_text = ""
        request_context = {
            "source_type": message_type,
            "source_ref": interaction_id,
            "metadata": metadata or {},
        }

        for _round in range(self.settings.max_tool_rounds):
            try:
                response = await self.llm.chat(
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
                result = await self.tool_executor.execute(
                    tc["name"], tc["arguments"], request_context=request_context,
                )
                safe_result = sanitize_for_context(result, MAX_TOOL_RESULT_LEN)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{tc['name']}"),
                    "content": safe_result,
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
    def _input_channel_hint(message_type: str) -> str:
        return {
            "text": "Обычный текстовый запрос.",
            "voice": "Расшифровка голосового. Это inbox capture: извлекай факты, договорённости, задачи и даты.",
            "photo": "Фото/скрин/документ. Это inbox capture: извлекай задачи, даты, контакты, суммы и ключевые детали с изображения.",
            "forward": "Пересланное сообщение. Это inbox capture: извлекай action items, follow-ups, события, обещания и людей.",
        }.get(message_type, f"Тип сообщения: {message_type}")

    async def _build_system_prompt(self, user_message: str, message_type: str) -> str:
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
            pending_decisions=self._section_decisions(s),
            mood_trend=self._section_mood_trend(),
            theme_clusters=self._section_theme_clusters(s),
            quiet_contacts=self._section_quiet_contacts(s),
            birthdays=self._section_birthdays(now, s),
            input_channel=self._input_channel_hint(message_type),
            current_context=self._section_ephemeral_state(s),
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

    _NOTIFICATION_LABELS = {
        "morning_briefing": "брифинг",
        "evening_summary": "вечер",
        "diary": "дневник",
        "weekly_review": "неделя",
        "smart_question": "вопрос",
        "open_loops_nudge": "контроль",
        "decision_followup": "решение",
        "birthday_alert": "день рождения",
        "weather_alert": "погода",
        "reminder": "напоминание",
    }

    def _fmt_local_time(self, ts: str, today_local: str) -> str:
        """Thin wrapper over `fmt_local_time` that bakes in the profile TZ."""
        return fmt_local_time(ts, self.profile.timezone, today_local)

    def _build_history_turns(
        self, limit: int = CONVERSATION_HISTORY_TURNS,
    ) -> list[dict]:
        """Replay recent interactions as real role-based LLM turns.

        Returns a list of `{"role": "user"|"assistant", "content": str}` dicts
        suitable for prepending to Brain.process()'s messages list. Unlike the
        old flat "## Разговор" text block this gives Claude native multi-turn
        conversation — each prior user message and bot reply is a distinct
        turn, with proactive notifications (briefings, reminders, evening
        summary, etc.) appearing as assistant turns prefixed with a label +
        local time so Claude knows they weren't replies to the user.

        All content is passed through sanitize_for_context — past user text
        might contain injection attempts that'd be replayed verbatim otherwise.

        Returns [] on DB failure rather than propagating — a transient SQLite
        hiccup shouldn't block the current message from reaching Claude; the
        degraded call (no history) is better than a hard failure.
        """
        try:
            rows = self.db.get_recent_messages(limit=limit)
        except Exception as e:
            logger.warning("History fetch failed: %s", type(e).__name__)
            return []
        if not rows:
            return []
        today_local = tz_now(self.profile.timezone).strftime("%Y-%m-%d")
        turns: list[dict] = []
        for m in rows:
            raw = m.get("content") or ""
            content = sanitize_for_context(raw, HISTORY_TURN_CHAR_CAP)
            if not content.strip():
                continue
            direction = m.get("direction")
            msg_type = m.get("message_type")
            if direction == "in":
                turns.append({"role": "user", "content": content})
            elif msg_type == "notification":
                kind = None
                meta_raw = m.get("metadata")
                if meta_raw:
                    try:
                        kind = json.loads(meta_raw).get("kind")
                    except (json.JSONDecodeError, TypeError):
                        pass
                label = self._NOTIFICATION_LABELS.get(kind, "уведомление")
                ts = self._fmt_local_time(m.get("timestamp") or "", today_local)
                turns.append({
                    "role": "assistant",
                    "content": f"[{label} в {ts}]\n{content}",
                })
            else:
                turns.append({"role": "assistant", "content": content})
        merged = self._merge_consecutive(turns)
        # Anthropic requires messages to start with role=user. If the only
        # history is orphan proactive sends (e.g. morning briefing fired and
        # user hasn't replied yet), the first replayed turn is assistant —
        # which would make [assistant, user_current] and fail the API. Drop
        # the leading assistant turn in that case; its content remains
        # reachable via the search_conversations tool if the user references
        # it directly.
        if merged and merged[0]["role"] == "assistant":
            merged = merged[1:]
        return merged

    @staticmethod
    def _merge_consecutive(turns: list[dict]) -> list[dict]:
        """Collapse consecutive same-role text turns into one.

        Anthropic's messages API requires alternating user/assistant roles —
        two notifications firing back-to-back (e.g. birthday alert + briefing
        at 09:00 with no user reply between) would produce two assistant
        turns in a row and fail the API call. Multimodal content (lists) is
        left alone so we never merge an image turn with a text turn.
        """
        result: list[dict] = []
        for t in turns:
            if (result
                    and result[-1]["role"] == t["role"]
                    and isinstance(result[-1].get("content"), str)
                    and isinstance(t.get("content"), str)):
                result[-1] = {
                    "role": t["role"],
                    "content": result[-1]["content"] + "\n\n" + t["content"],
                }
            else:
                result.append(dict(t))
        return result

    def _section_decisions(self, s) -> str:
        decisions = self.db.get_pending_decisions(limit=5)
        return "\n".join(
            f"- {s(d['description'], 200)}" for d in decisions
        ) or "Нет решений в процессе."

    _EPHEMERAL_LABELS = {
        "location": "Локация",
        "health": "Здоровье",
        "availability": "Занят",
        "energy": "Энергия",
        "activity": "Сейчас",
    }

    def _implicit_state(self, now: datetime) -> list[dict]:
        """Schedule-derived state (work hours on work days → location=на работе).

        Handles normal schedules (start <= end, same-day window) and
        wrap-midnight schedules (e.g. 22:00-06:00). Out-of-range times
        (hour >= 24, minute >= 60, etc.) silently return [] — all datetime
        construction is inside a single try/except so bad profile config
        doesn't crash Brain.process().
        """
        profile = self.profile
        work_days = profile.work_days or [1, 2, 3, 4, 5]
        if now.isoweekday() not in work_days:
            return []
        try:
            sh, sm = map(int, profile.work_start.split(":"))
            eh, em = map(int, profile.work_end.split(":"))
            start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
        except (ValueError, AttributeError, TypeError):
            return []

        # Same-day window (09:00-18:00)
        if start <= end:
            if not (start <= now <= end):
                return []
            effective_end = end
        # Wrap midnight (22:00-06:00): "within shift" is (now >= start) OR
        # (now <= end). Expiration rolls to tomorrow when we're still in
        # the evening half of the shift.
        else:
            if now >= start:
                effective_end = end + timedelta(days=1)
            elif now <= end:
                effective_end = end
            else:
                return []

        return [{
            "key": "location",
            "value": "на работе",
            "expires_at": effective_end.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "implicit",
        }]

    def get_merged_ephemeral_state(self, now: datetime) -> list[dict]:
        """Manual + implicit ephemeral state, manual wins by key.

        Each source is guarded independently — a failure in one (DB error,
        malformed profile) still returns the other. Shared between
        _section_ephemeral_state (for the system prompt) and the /context
        Telegram command (for user inspection).
        """
        try:
            manual = self.db.get_active_ephemeral_state()
        except Exception as e:
            logger.warning("Ephemeral state (manual) failed: %s", type(e).__name__)
            manual = []
        manual_keys = {r["key"] for r in manual}
        try:
            implicit = [
                r for r in self._implicit_state(now)
                if r["key"] not in manual_keys
            ]
        except Exception as e:
            logger.warning("Ephemeral state (implicit) failed: %s", type(e).__name__)
            implicit = []
        return manual + implicit

    def _section_ephemeral_state(self, s) -> str:
        """Format active ephemeral state rows for the system prompt."""
        now = tz_now(self.profile.timezone)
        rows = self.get_merged_ephemeral_state(now)
        if not rows:
            return "Пусто."
        today_date = now.strftime("%Y-%m-%d")
        lines = []
        for row in rows:
            label = self._EPHEMERAL_LABELS.get(row["key"], row["key"])
            value = s(row["value"], 150)
            expires = row.get("expires_at") or ""
            tail = ""
            if len(expires) >= 16:
                if expires.startswith(today_date):
                    tail = f" (до {expires[11:16]})"
                else:
                    tail = f" (до {expires[5:16]})"
            if row.get("source") == "implicit":
                tail += " · по расписанию" if tail else " (по расписанию)"
            lines.append(f"{label}: {value}{tail}")
        return "\n".join(lines)

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
