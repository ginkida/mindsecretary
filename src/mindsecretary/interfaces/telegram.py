from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from collections import deque
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..core import tz_now
from ..core.brain import Brain
from ..learning.mood import check_contact_frequency
from ..voice.stt import GroqSTT

logger = logging.getLogger(__name__)

MAX_VOICE_SIZE = 25 * 1024 * 1024
MAX_VOICE_DURATION = 600
MAX_PHOTO_SIZE = 10 * 1024 * 1024
MAX_TEXT_LENGTH = 10_000
MAX_TRANSCRIPT_LENGTH = 30_000
DOWNLOAD_TIMEOUT = 30.0
TG_MSG_LIMIT = 4096

def _fix_markdown(text: str) -> str:
    """Fix common Markdown issues that cause Telegram parse errors.

    Ensures paired formatting chars (* _ `) and escapes orphans.
    """
    for ch in ("*", "_", "`"):
        if text.count(ch) % 2 != 0:
            # Odd count → escape the last occurrence to make it even
            idx = text.rfind(ch)
            text = text[:idx] + "\\" + text[idx:]
    return text


def _split_message(text: str, limit: int = TG_MSG_LIMIT) -> list[str]:
    """Split long text into Telegram-safe chunks."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        # Try to split at newline
        idx = text.rfind("\n", 0, limit)
        if idx < limit // 2:
            idx = limit
        parts.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return parts


def _memory_source_label(source_type: str | None, source_ref: str | None) -> str:
    label = {
        "text": "текст",
        "voice": "голос",
        "photo": "фото",
        "forward": "пересылка",
    }.get(source_type or "", source_type or "неизвестно")
    if source_ref:
        return f"{label} #{source_ref[:8]}"
    return label


def _format_memory_line(memory: dict, include_match: bool = False) -> str:
    lines = [f"• [{memory['category']}] {memory['content'][:220]}"]
    if include_match:
        lines.append(
            f"  why: {memory.get('match_reason', 'совпадение по смыслу')}, "
            f"score {memory.get('final_score', memory.get('score', 0.0)):.2f}"
        )
    lines.append(
        f"  source: {_memory_source_label(memory.get('source_type'), memory.get('source_ref'))}, "
        f"confidence {float(memory.get('confidence') or 0.0):.2f}, "
        f"created {str(memory.get('created_at') or '')[:16]}"
    )
    return "\n".join(lines)


class TelegramBot:
    def __init__(self, token: str, allowed_user_id: int,
                 brain: Brain, stt: GroqSTT):
        self.token = token
        self.allowed_user_id = allowed_user_id
        self.brain = brain
        self.stt = stt
        self.app: Application | None = None
        # Simple in-memory rate limit — protects against compromised Telegram
        # session spamming expensive (LLM-triggering) handlers.
        self._rate_limit_window: deque[float] = deque()
        self._rate_limit_per_min = brain.settings.rate_limit_per_minute

    def _check_user(self, update: Update) -> bool:
        uid = update.effective_user.id if update.effective_user else 0
        if uid != self.allowed_user_id:
            logger.warning("Unauthorized: user %s", uid)
            return False
        return True

    def _check_rate_limit(self) -> bool:
        """Return True if we're within the per-minute message budget."""
        now = time.time()
        while self._rate_limit_window and now - self._rate_limit_window[0] > 60:
            self._rate_limit_window.popleft()
        if len(self._rate_limit_window) >= self._rate_limit_per_min:
            return False
        self._rate_limit_window.append(now)
        return True

    async def _require_rate_limit(self, update: Update) -> bool:
        if self._check_rate_limit():
            return True
        await update.message.reply_text("Слишком часто, подожди минуту.")
        return False

    async def _reply(self, update: Update, text: str):
        """Send reply with message splitting (no feedback UI — too noisy)."""
        for part in _split_message(text):
            try:
                await update.message.reply_text(
                    _fix_markdown(part), parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                # Fallback without markdown if parsing fails
                await update.message.reply_text(part)

    async def _typing(self, update: Update):
        """Show typing indicator."""
        await update.message.chat.send_action(ChatAction.TYPING)

    # --- Commands ---

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        await update.message.reply_text(
            "Привет! Я MindSecretary — твой персональный секретарь.\n\n"
            "🎤 Голосовые — наговори что угодно\n"
            "📝 Текст — команды и вопросы\n"
            "📸 Фото — визитки, чеки, скриншоты\n"
            "↪️ Пересылка — из других чатов\n\n"
            "Команды:\n"
            "/stats — расходы и статистика\n"
            "/diary — записи дневника\n"
            "/people — твои контакты\n"
            "/goals — цели на сегодня\n"
            "/habits — привычки и streaks\n"
            "/search — поиск по памяти\n"
            "/memory — что именно и почему я помню\n"
            "/loops — что сейчас висит и на контроле\n"
            "/context — текущий контекст (где ты, что с тобой)\n"
            "/undo — восстановить удалённое\n"
            "/export — экспорт данных в JSON\n"
            "/review — запустить недельный обзор\n"
            "/forget — удалить воспоминание",
        )
        # Show notification count
        try:
            count = self.brain.db.count_notifications_today()
            limit = self.brain.profile.notification_limit
            if count > 0:
                await update.message.reply_text(f"📬 Уведомлений сегодня: {count}/{limit}")
        except Exception:
            pass

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        stats = self.brain.db.get_stats()
        lines = [
            "📊 *Статистика*\n",
            f"💰 Сегодня: ${stats['today_cost']:.4f} ({stats['today_tokens']:,} tokens)",
            f"💰 Месяц: ${stats['month_cost']:.4f}",
        ]
        # Per-provider breakdown
        providers = stats.get("providers", {})
        if providers:
            lines.append("")
            for p, d in sorted(providers.items()):
                lines.append(f"  {p}: ${d['cost']:.4f} ({d['tokens']:,} tok)")
        # 7-day trend
        trend = stats.get("week_trend", [])
        if trend:
            trend_str = " ".join(f"${d['cost']:.2f}" for d in trend[-7:])
            lines.append(f"\n📈 7 дней: {trend_str}")
        lines.extend([
            f"\n🧠 Воспоминаний: {stats['memories']}",
            f"👤 Контактов: {stats['contacts']}",
            f"💬 Взаимодействий сегодня: {stats['interactions_today']}",
        ])
        text = "\n".join(lines)
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text)

    async def _handle_diary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        entries = self.brain.db.get_diary_entries(days=7)
        if not entries:
            await update.message.reply_text("Записей в дневнике пока нет.")
            return
        for e in entries[:3]:
            mood = f" | {e['mood']}" if e.get("mood") else ""
            people = f"\n👤 {e['people']}" if e.get("people") else ""
            text = f"📖 *{e['date']}*{mood}{people}\n\n{e['content']}"
            for part in _split_message(text):
                try:
                    await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await update.message.reply_text(part)

    async def _handle_people(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        contacts = self.brain.db.get_contacts("")
        if not contacts:
            await update.message.reply_text("Контактов пока нет.")
            return
        lines = ["👥 *Контакты*\n"]
        now_local = self.brain.db.local_now_naive()
        for c in contacts[:20]:
            line = f"• *{c['name']}*"
            if c.get("relation"):
                line += f" ({c['relation']})"
            if c.get("last_contact"):
                try:
                    last = datetime.fromisoformat(c["last_contact"])
                    # last_contact is a profile-local naive string; compare
                    # against local naive "now" to avoid the Docker-system-
                    # UTC-vs-local offset shifting days by +/-1.
                    days = (now_local - last).days
                    line += f" — {days}д назад"
                except (ValueError, TypeError):
                    pass
            if c.get("notes"):
                line += f"\n  _{c['notes'][:80]}_"
            lines.append(line)
        text = "\n".join(lines)
        for part in _split_message(text):
            try:
                await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(part)

    async def _handle_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        await update.message.reply_text("⏳ Генерирую обзор...")
        await self._typing(update)
        # Access weekly_reflection through the scheduler (set in app.py)
        scheduler = getattr(self, "_scheduler", None)
        if scheduler and scheduler.weekly_reflection:
            text = await scheduler.weekly_reflection.generate_weekly_review()
            if text:
                await self._reply(update, text)
                return
        await update.message.reply_text("Недостаточно данных для обзора.")

    async def _handle_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text(
                "Использование: /search <запрос>\n"
                "Пример: /search встреча с Алексеем",
            )
            return
        if len(query) > 500:
            await update.message.reply_text("Слишком длинный запрос (макс 500 символов).")
            return
        results = await self.brain.memory.search(query, top_k=5)
        if not results:
            await update.message.reply_text("Ничего не нашёл.")
            return
        lines = ["🔍 *Результаты поиска*\n"]
        for r in results:
            score = r.get("final_score", r.get("score", 0))
            cat = r.get("category", "?")
            person = f" ({r['related_person']})" if r.get("related_person") else ""
            lines.append(f"• [{cat}] {r['content'][:150]}{person}\n  _score: {score:.2f}_")
        text = "\n".join(lines)
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text)

    async def _handle_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        query = " ".join(context.args) if context.args else ""
        if len(query) > 500:
            await update.message.reply_text("Слишком длинный запрос (макс 500 символов).")
            return

        if query:
            memories = await self.brain.memory.search(query, top_k=6)
            if not memories:
                await update.message.reply_text("По этому запросу память пуста.")
                return
            lines = ["🧠 *Что помню по запросу*\n"]
            lines.extend(_format_memory_line(m, include_match=True) for m in memories)
        else:
            memories = self.brain.memory.list_recent(limit=8)
            if not memories:
                await update.message.reply_text("Память пока пуста.")
                return
            lines = ["🧠 *Недавняя память*\n"]
            lines.extend(_format_memory_line(m) for m in memories)

        text = "\n".join(lines)
        for part in _split_message(text):
            try:
                await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(part)

    async def _handle_loops(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        loops = self.brain.db.get_open_loops(days_ahead=2, limit_per_section=5)
        promises = self.brain.memory.get_by_category("promise")[:3]
        contact_alerts = [
            a for a in check_contact_frequency(self.brain.db)
            if a.get("days_since", 0) > self.brain.settings.quiet_contact_days
            and a.get("mention_count", 0) >= self.brain.settings.quiet_contact_min_mentions
        ][:3]

        counts = loops.get("counts", {})
        lines = [
            "📌 *Открытые хвосты*\n",
            f"Напоминания просрочены: {counts.get('overdue_reminders', 0)}",
            f"До конца дня: {counts.get('due_today_reminders', 0)}",
            f"События впереди: {counts.get('upcoming_events', 0)}",
            f"Цели на сегодня открыты: {counts.get('pending_goals', 0)}",
            f"Решения с follow-up: {counts.get('due_decisions', 0)}",
        ]

        if loops.get("overdue_reminders"):
            lines.append("\n⏰ Просроченные:")
            lines.extend(
                f"• {r['trigger_at'][5:16]} — {r['text'][:140]}"
                for r in loops["overdue_reminders"]
            )
        if loops.get("upcoming_events"):
            lines.append("\n📅 Ближайшие события:")
            lines.extend(
                f"• {e['start_at'][5:16]} — {e['title'][:140]}"
                + (f" ({e['related_person']})" if e.get("related_person") else "")
                for e in loops["upcoming_events"]
            )
        if loops.get("pending_goals"):
            lines.append("\n🎯 Незакрытые цели:")
            lines.extend(
                f"• {g['title'][:140]} [{g['priority']}]"
                for g in loops["pending_goals"]
            )
        if loops.get("due_decisions"):
            lines.append("\n📋 Follow-up по решениям:")
            lines.extend(
                f"• {d['description'][:140]}"
                for d in loops["due_decisions"]
            )
        if promises:
            lines.append("\n🤝 Обещания в памяти:")
            lines.extend(f"• {p['content'][:140]}" for p in promises)
        if contact_alerts:
            lines.append("\n👥 Тихие контакты:")
            lines.extend(
                f"• {a['name']} — {a['days_since']} дней"
                for a in contact_alerts
            )

        text = "\n".join(lines)
        for part in _split_message(text):
            try:
                await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(part)

    async def _handle_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text(
                "Использование: /forget <что забыть>\n"
                "Пример: /forget аллергия на орехи",
            )
            return
        if len(query) > 500:
            await update.message.reply_text("Слишком длинный запрос (макс 500 символов).")
            return
        results = await self.brain.memory.search(query, top_k=3)
        if not results:
            await update.message.reply_text("Не нашёл такого в памяти.")
            return
        top = results[0]
        confirm_kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Да, удалить", callback_data=f"forget_yes:{top['id']}"),
                InlineKeyboardButton("Отмена", callback_data="forget_no"),
            ]
        ])
        try:
            await update.message.reply_text(
                f"Удалить это?\n\n_{_fix_markdown(top['content'][:300])}_",
                reply_markup=confirm_kb,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            await update.message.reply_text(
                f"Удалить это?\n\n{top['content'][:300]}",
                reply_markup=confirm_kb,
            )

    async def _handle_undo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        last = self.brain.memory.get_last_deleted()
        if not last:
            await update.message.reply_text("Нечего восстанавливать.")
            return
        self.brain.memory.restore(last["id"])
        await update.message.reply_text(
            f"♻️ Восстановлено: {last['content'][:200]}"
        )

    async def _handle_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/context — show active ephemeral state (manual + schedule-derived)."""
        if not self._check_user(update):
            return
        args = [a.lower() for a in (context.args or [])]
        if args and args[0] == "clear":
            target_key = args[1] if len(args) > 1 else None
            n = self.brain.db.clear_ephemeral_state(target_key)
            what = f"ключ '{target_key}'" if target_key else "весь ручной контекст"
            await update.message.reply_text(
                f"🧹 Очищено ({what}, {n} {'запись' if n == 1 else 'записей'}). "
                f"Расписание из профиля остаётся."
            )
            return

        now = tz_now(self.brain.profile.timezone)
        rows = self.brain.get_merged_ephemeral_state(now)
        manual = [r for r in rows if r.get("source") != "implicit"]
        implicit = [r for r in rows if r.get("source") == "implicit"]

        if not rows:
            await update.message.reply_text(
                "Текущего контекста нет.\n\n"
                "Это «здесь и сейчас»: где ты, что с тобой, занят ли. "
                "Бот ставит сам, когда ты упоминаешь свою ситуацию. "
                "Расписание из профиля (рабочие часы) подставляется автоматически."
            )
            return

        labels = {
            "location": "Локация",
            "health": "Здоровье",
            "availability": "Занят",
            "energy": "Энергия",
            "activity": "Сейчас",
        }
        lines = ["🎯 *Текущий контекст*\n"]
        for row in manual:
            label = labels.get(row["key"], row["key"])
            expires = row["expires_at"] or ""
            tail = f" _(до {expires[11:16]})_" if len(expires) >= 16 else ""
            lines.append(f"📍 {label}: {row['value']}{tail}")
        for row in implicit:
            label = labels.get(row["key"], row["key"])
            expires = row.get("expires_at") or ""
            tail = f" _(до {expires[11:16]})_" if len(expires) >= 16 else ""
            lines.append(f"⏰ {label}: {row['value']}{tail} _· по расписанию_")
        lines.append(
            "\n_📍 — ручное, меняется через разговор с ботом_"
        )
        lines.append(
            "_⏰ — из профиля (рабочие часы/дни), меняется в `config/profile.yaml`_"
        )
        lines.append("_/context clear — сбросить ручной контекст._")
        try:
            await update.message.reply_text(
                _fix_markdown("\n".join(lines)), parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            await update.message.reply_text("\n".join(lines))

    async def _handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        await update.message.reply_text("⏳ Готовлю экспорт...")

        db = self.brain.db
        exported_at = tz_now(self.brain.profile.timezone).isoformat()
        data = {
            "exported_at": exported_at,
            "memories": db.db.execute(
                "SELECT id, content, category, importance, related_person, "
                "related_date, source_type, source_ref, confidence, created_at "
                "FROM memories WHERE status = 'active' "
                "ORDER BY created_at DESC"
            ).fetchall(),
            "contacts": db.get_contacts(""),
            "diary": db.get_diary_entries(days=365),
            "events": db.db.execute(
                "SELECT * FROM events ORDER BY start_at DESC LIMIT 500"
            ).fetchall(),
            "decisions": db.db.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC LIMIT 200"
            ).fetchall(),
        }
        # Convert sqlite3.Row objects to dicts
        for key in ("memories", "events", "decisions"):
            data[key] = [dict(r) for r in data[key]]

        json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        doc = io.BytesIO(json_bytes)
        doc.name = (
            f"mindsecretary_export_"
            f"{tz_now(self.brain.profile.timezone).strftime('%Y%m%d')}.json"
        )

        await update.message.reply_document(
            document=doc,
            caption=f"📦 Экспорт: {len(data['memories'])} воспоминаний, "
                    f"{len(data['contacts'])} контактов, "
                    f"{len(data['diary'])} записей дневника",
        )

    async def _handle_habits(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        stats = self.brain.db.get_habit_stats()
        if not stats:
            await update.message.reply_text("Привычек пока нет. Скажи мне что отслеживать!")
            return
        lines = ["📊 *Привычки*\n"]
        for h in stats:
            streak_str = f"🔥 {h['streak']}д" if h['streak'] > 0 else "—"
            lines.append(f"• *{h['name']}* — {streak_str} | неделя: {h['week_rate']}%")
        text = "\n".join(lines)
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text)

    async def _handle_goals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        goals = self.brain.db.get_daily_goals()
        if not goals:
            await update.message.reply_text("На сегодня целей нет. Скажи мне, что планируешь!")
            return
        lines = ["🎯 *Цели на сегодня*\n"]
        for g in goals:
            emoji = {"pending": "⬜", "completed": "✅",
                     "skipped": "⏭", "partial": "🟡"}.get(g["status"], "⬜")
            prio = " ❗" if g.get("priority") == "high" else ""
            line = f"{emoji} {g['title']}{prio}"
            if g.get("reflection"):
                line += f"\n   _{g['reflection'][:80]}_"
            lines.append(line)
        done = sum(1 for g in goals if g["status"] == "completed")
        lines.append(f"\n✅ {done}/{len(goals)} выполнено")
        text = "\n".join(lines)
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text)

    # --- Callback handlers ---

    async def _handle_forget_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if not self._check_user(update):
            return
        if query.data == "forget_no":
            await query.edit_message_text("Отменено.")
            return
        # forget_yes:<memory_id>
        memory_id = query.data.split(":", 1)[1] if ":" in query.data else ""
        if memory_id:
            self.brain.memory.delete(memory_id)
            await query.edit_message_text(f"🗑 Удалено.")
        else:
            await query.edit_message_text("Ошибка: ID не найден.")

    # --- Message handlers ---

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        voice = update.message.voice or update.message.audio
        if not voice:
            return
        if voice.file_size and voice.file_size > MAX_VOICE_SIZE:
            await update.message.reply_text("Слишком большой файл (макс 25 МБ).")
            return
        if voice.duration and voice.duration > MAX_VOICE_DURATION:
            await update.message.reply_text("Слишком длинное сообщение (макс 10 мин).")
            return

        await self._typing(update)

        try:
            file = await context.bot.get_file(voice.file_id)
            audio_bytes = bytes(await asyncio.wait_for(
                file.download_as_bytearray(), timeout=DOWNLOAD_TIMEOUT,
            ))
        except asyncio.TimeoutError:
            await update.message.reply_text("Таймаут загрузки.")
            return
        except Exception as e:
            logger.error("Voice download failed: %s", type(e).__name__)
            await update.message.reply_text("Не удалось скачать голосовое.")
            return

        try:
            transcript = await self.stt.transcribe(audio_bytes)
        except Exception as e:
            logger.error("STT failed: %s", type(e).__name__)
            await update.message.reply_text("Не удалось распознать речь.")
            return
        finally:
            del audio_bytes

        if not transcript.strip():
            await update.message.reply_text("Не удалось распознать речь.")
            return

        if len(transcript) > MAX_TRANSCRIPT_LENGTH:
            await update.message.reply_text(
                f"Транскрипт длинный ({len(transcript)} симв), обрезаю до {MAX_TRANSCRIPT_LENGTH}.",
            )
            transcript = transcript[:MAX_TRANSCRIPT_LENGTH]

        logger.info("Voice transcribed (%d chars)", len(transcript))
        await self._typing(update)

        try:
            response = await asyncio.wait_for(
                self.brain.process(
                    user_message=transcript, message_type="voice",
                    metadata={"duration_sec": voice.duration},
                ),
                timeout=self.brain.settings.process_timeout_sec,
            )
            await self._reply(update, response.text)
        except asyncio.TimeoutError:
            await update.message.reply_text("Таймаут обработки.")
        except Exception as e:
            logger.error("Brain failed: %s", type(e).__name__)
            await update.message.reply_text("Произошла ошибка.")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        msg = update.message
        if not msg.photo:
            return

        # Get the largest photo
        photo = msg.photo[-1]
        if photo.file_size and photo.file_size > MAX_PHOTO_SIZE:
            await msg.reply_text("Фото слишком большое (макс 10 МБ).")
            return

        caption = (
            msg.caption
            or "Разбери фото как inbox: извлеки факты, задачи, контакты, даты и важные детали."
        )[:MAX_TEXT_LENGTH]
        await self._typing(update)

        try:
            file = await context.bot.get_file(photo.file_id)
            photo_bytes = bytes(await asyncio.wait_for(
                file.download_as_bytearray(), timeout=DOWNLOAD_TIMEOUT,
            ))
        except asyncio.TimeoutError:
            await msg.reply_text("Таймаут загрузки фото.")
            return
        except Exception as e:
            logger.error("Photo download failed: %s", type(e).__name__)
            await msg.reply_text("Не удалось скачать фото.")
            return

        image_b64 = base64.b64encode(photo_bytes).decode("utf-8")
        del photo_bytes

        logger.info("Photo received (%d KB), caption: %s", len(image_b64) // 1024, caption[:50])
        await self._typing(update)

        try:
            response = await asyncio.wait_for(
                self.brain.process(
                    user_message=caption, message_type="photo",
                    image_base64=image_b64,
                ),
                timeout=self.brain.settings.process_timeout_sec,
            )
            await self._reply(update, response.text)
        except asyncio.TimeoutError:
            await msg.reply_text("Таймаут обработки фото.")
        except Exception as e:
            logger.error("Brain failed on photo: %s", type(e).__name__)
            await msg.reply_text("Не удалось обработать фото.")

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        text = update.message.text
        if not text:
            return
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        await self._typing(update)

        try:
            response = await asyncio.wait_for(
                self.brain.process(user_message=text, message_type="text"),
                timeout=self.brain.settings.process_timeout_sec,
            )
            await self._reply(update, response.text)
        except asyncio.TimeoutError:
            await update.message.reply_text("Таймаут обработки.")
        except Exception as e:
            logger.error("Brain failed: %s", type(e).__name__)
            await update.message.reply_text("Произошла ошибка.")

    async def _handle_forward(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        msg = update.message
        forward_from = "[Переслано]: "
        origin = msg.forward_origin
        if origin:
            try:
                if hasattr(origin, "sender_user") and origin.sender_user:
                    u = origin.sender_user
                    name = u.first_name or ""
                    if u.last_name:
                        name += f" {u.last_name}"
                    forward_from = f"[Переслано от {name}]: "
                elif hasattr(origin, "sender_user_name") and origin.sender_user_name:
                    forward_from = f"[Переслано от {origin.sender_user_name}]: "
                elif hasattr(origin, "sender_chat") and origin.sender_chat:
                    forward_from = f"[Переслано из {origin.sender_chat.title}]: "
            except Exception:
                pass

        text = (msg.text or msg.caption or "")[:MAX_TEXT_LENGTH]
        full_text = f"{forward_from}{text}"
        if not full_text.strip():
            return

        await self._typing(update)

        try:
            response = await asyncio.wait_for(
                self.brain.process(user_message=full_text, message_type="forward"),
                timeout=self.brain.settings.process_timeout_sec,
            )
            await self._reply(update, response.text)
        except asyncio.TimeoutError:
            await update.message.reply_text("Таймаут обработки.")
        except Exception as e:
            logger.error("Brain failed: %s", type(e).__name__)
            await update.message.reply_text("Произошла ошибка.")

    # --- Build ---

    def build(self) -> Application:
        self.app = Application.builder().token(self.token).build()

        # Commands
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("stats", self._handle_stats))
        self.app.add_handler(CommandHandler("diary", self._handle_diary))
        self.app.add_handler(CommandHandler("people", self._handle_people))
        self.app.add_handler(CommandHandler("review", self._handle_review))
        self.app.add_handler(CommandHandler("search", self._handle_search))
        self.app.add_handler(CommandHandler("memory", self._handle_memory))
        self.app.add_handler(CommandHandler("context", self._handle_context))
        self.app.add_handler(CommandHandler("loops", self._handle_loops))
        self.app.add_handler(CommandHandler("undo", self._handle_undo))
        self.app.add_handler(CommandHandler("forget", self._handle_forget))
        self.app.add_handler(CommandHandler("goals", self._handle_goals))
        self.app.add_handler(CommandHandler("habits", self._handle_habits))
        self.app.add_handler(CommandHandler("export", self._handle_export))

        # Confirmation callback for /forget
        self.app.add_handler(CallbackQueryHandler(self._handle_forget_confirm, pattern="^forget_"))

        # Messages (order matters: forwarded first, then photo, voice, text)
        self.app.add_handler(MessageHandler(filters.FORWARDED, self._handle_forward))
        self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

        return self.app

    async def send_message(self, text: str):
        """Send a proactive message to the user."""
        if not (self.app and self.app.bot):
            return
        for part in _split_message(text):
            try:
                await self.app.bot.send_message(
                    chat_id=self.allowed_user_id, text=_fix_markdown(part),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                await self.app.bot.send_message(
                    chat_id=self.allowed_user_id, text=part,
                )
