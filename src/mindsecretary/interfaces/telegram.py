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

from ..core import fmt_utc_to_local, pluralize_ru, tz_now
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


def _format_memory_line(memory: dict, include_match: bool = False,
                        tz_name: str | None = None) -> str:
    lines = [f"• [{memory['category']}] {memory['content'][:220]}"]
    if include_match:
        lines.append(
            f"  why: {memory.get('match_reason', 'совпадение по смыслу')}, "
            f"score {memory.get('final_score', memory.get('score', 0.0)):.2f}"
        )
    # created_at is UTC (datetime('now')). Convert to profile-local for
    # display. Pre-fix the raw [:16] slice showed UTC time, which read
    # off by N hours for users on positive offsets and could even
    # surface yesterday's date for memories saved just past local
    # midnight in Asia/Almaty.
    created = fmt_utc_to_local(memory.get("created_at") or "", tz_name)
    lines.append(
        f"  source: {_memory_source_label(memory.get('source_type'), memory.get('source_ref'))}, "
        f"confidence {float(memory.get('confidence') or 0.0):.2f}, "
        f"created {created}"
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

    async def _reply(self, update: Update, text: str | None):
        """Send reply with message splitting (no feedback UI — too noisy).

        Empty/whitespace text → silent return. The Brain can produce empty
        text legitimately (e.g. tool-only round where the user message
        triggered save_memory and Claude chose to say nothing). Pre-fix
        this hit Telegram's "message text empty" error, which the outer
        handler caught and surfaced as 'Произошла ошибка' — wrong message
        for a successful no-reply case.
        """
        if not text or not text.strip():
            logger.info("Empty reply suppressed (brain produced no text)")
            return
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
            "/forget — удалить воспоминание\n"
            "/about <имя> — брифинг про человека\n"
            "/learnings — накопленные инсайты из weekly review\n"
            "/snooze <время> — пауза проактивных уведомлений (напр. 2h)\n"
            "/version — версия и базовые счётчики",
        )
        # Show notification count
        try:
            count = self.brain.db.count_notifications_today()
            limit = self.brain.profile.notification_limit
            if count > 0:
                await update.message.reply_text(f"📬 Уведомлений сегодня: {count}/{limit}")
        except Exception:
            pass

    @staticmethod
    def _resolve_version() -> str:
        """Read package version from installed metadata.

        Falls back to "unknown" if the package isn't installed (rare —
        editable install via `pip install -e .` is the entry path) so
        /version never crashes on a corner case.
        """
        try:
            from importlib.metadata import PackageNotFoundError, version
            return version("mindsecretary")
        except (ImportError, PackageNotFoundError):
            return "unknown"

    async def _handle_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        ver = self._resolve_version()
        # Lightweight self-introspection: sizes that prove the bot is
        # alive and accumulating state, without dragging in /stats's
        # full cost-and-token report. Each query catches its own
        # exception so a single broken table doesn't sink the whole
        # response.
        def _safe(fn, default=0):
            try:
                return fn()
            except Exception:
                return default
        memories = _safe(self.brain.memory.count)
        contacts_count = _safe(
            lambda: len(self.brain.db.get_contacts("")),
        )
        pending_reminders = _safe(
            lambda: len(self.brain.db.get_pending_reminders()),
        )
        text = (
            f"🤖 MindSecretary v{ver}\n"
            f"🧠 Воспоминаний: {memories}\n"
            f"👤 Контактов: {contacts_count}\n"
            f"⏰ Pending-напоминаний: {pending_reminders}\n"
            f"🌍 TZ: {self.brain.profile.timezone or 'system'}"
        )
        await update.message.reply_text(text)

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
        # Monthly projection from the 7-day average — gives the user a
        # visible burn-rate trajectory so a creeping cost surprise
        # doesn't only become visible when the daily_cost_limit_usd
        # circuit breaker fires.
        projection = stats.get("month_projection")
        if projection is not None:
            lines.append(f"🔮 Прогноз/мес: ${projection:.2f} (по 7-дн avg)")
        lines.extend([
            f"\n🧠 Воспоминаний: {stats['memories']}",
        ])
        # Top 5 memory categories — quick read on what kinds of facts the
        # bot is accumulating. Capped to 5 so the message stays scannable
        # in Telegram even with many categories in use.
        cat_breakdown = stats.get("memory_categories", [])
        if cat_breakdown:
            for c in cat_breakdown[:5]:
                lines.append(f"  • {c['category']}: {c['count']}")
        lines.extend([
            f"👤 Контактов: {stats['contacts']}",
            f"💬 Взаимодействий сегодня: {stats['interactions_today']}",
        ])
        text = "\n".join(lines)
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text)

    _DIARY_MAX_ENTRIES = 30

    async def _handle_diary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        # Parse optional arg:
        #   /diary               → last 3 from 7-day window (legacy default)
        #   /diary <N>           → last N entries (capped at _DIARY_MAX_ENTRIES)
        #   /diary YYYY-MM-DD    → entry for exactly that date
        arg = (context.args[0] if context.args else "").strip()
        entries: list[dict] = []
        limit = 3

        if arg:
            if len(arg) == 10 and arg.count("-") == 2:
                # Looks like a date — try to fetch a single entry
                entry = self.brain.db.get_diary_entry_by_date(arg)
                if not entry:
                    await update.message.reply_text(
                        f"Нет записи за {arg}.",
                    )
                    return
                entries = [entry]
                limit = 1
            else:
                try:
                    n = int(arg)
                except ValueError:
                    await update.message.reply_text(
                        "Использование:\n"
                        "  /diary — последние 3\n"
                        "  /diary 7 — последние 7 записей\n"
                        "  /diary 2026-04-15 — запись за дату",
                    )
                    return
                limit = max(1, min(self._DIARY_MAX_ENTRIES, n))
                # Pull a wider window to make sure we have enough rows.
                entries = self.brain.db.get_diary_entries(days=limit * 3)
        else:
            entries = self.brain.db.get_diary_entries(days=7)

        if not entries:
            await update.message.reply_text("Записей в дневнике пока нет.")
            return
        for e in entries[:limit]:
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

        tz_name = getattr(self.brain.db, "_timezone", None)
        if query:
            memories = await self.brain.memory.search(query, top_k=6)
            if not memories:
                await update.message.reply_text("По этому запросу память пуста.")
                return
            lines = ["🧠 *Что помню по запросу*\n"]
            lines.extend(
                _format_memory_line(m, include_match=True, tz_name=tz_name)
                for m in memories
            )
        else:
            memories = self.brain.memory.list_recent(limit=8)
            if not memories:
                await update.message.reply_text("Память пока пуста.")
                return
            lines = ["🧠 *Недавняя память*\n"]
            lines.extend(
                _format_memory_line(m, tz_name=tz_name) for m in memories
            )

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
            now_local = self.brain.db.local_now_naive()
            for e in loops["upcoming_events"]:
                # iter 13 added in-progress events (start_at in past, end_at
                # in future) to upcoming_events. Without a marker, /loops
                # at 14:30 reads "Ближайшее: 14:00 встреча" while the
                # user is 30 min into the meeting. Mirror of iter 15's
                # briefing fix.
                start_raw = e.get("start_at") or ""
                in_progress = False
                try:
                    start_dt = datetime.fromisoformat(
                        start_raw.replace(" ", "T"),
                    )
                    in_progress = start_dt <= now_local
                except (ValueError, TypeError):
                    pass
                marker = "▶️ сейчас" if in_progress else start_raw[5:16]
                tail = (
                    f" ({e['related_person']})"
                    if e.get("related_person") else ""
                )
                lines.append(f"• {marker} — {e['title'][:140]}{tail}")
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
            # Pluralize day-count — "31 дней" reads wrong; should be
            # "31 день", "32-34 дня", "35+ дней" (with teens carve-out
            # already in pluralize_ru).
            for a in contact_alerts:
                days = a["days_since"]
                word = pluralize_ru(days, ("день", "дня", "дней"))
                lines.append(f"• {a['name']} — {days} {word}")

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
        # Memory.restore returns False on missing id / already-active.
        # Pre-fix /undo always said "♻️ Восстановлено" — but two /undo
        # in a row would re-restore an already-active row (no-op) and
        # still claim success. Surface the real outcome instead.
        if not self.brain.memory.restore(last["id"]):
            await update.message.reply_text(
                "Уже восстановлено или удалено навсегда."
            )
            return
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
            records_word = pluralize_ru(n, ("запись", "записи", "записей"))
            await update.message.reply_text(
                f"🧹 Очищено ({what}, {n} {records_word}). "
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

    @staticmethod
    def _parse_snooze_duration(arg: str) -> int | None:
        """Parse '30m' / '2h' / '1d' → minutes. Returns None for invalid.

        Cap at 7 days to prevent accidental indefinite snooze ("/snooze 365d"
        would silently kill all proactive notifications for a year).
        """
        import re
        m = re.match(r"^(\d+)([mhd])$", arg.strip().lower())
        if not m:
            return None
        n = int(m.group(1))
        unit = m.group(2)
        minutes = {"m": n, "h": n * 60, "d": n * 60 * 24}.get(unit, 0)
        # Cap at 7 days
        return min(minutes, 7 * 24 * 60) if minutes > 0 else None

    async def _handle_snooze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Mute scheduled proactive notifications for a duration.

        Reminders are NOT affected — they go through a separate code path
        and represent explicit user intent at a chosen time.

        Usage:
            /snooze 30m       — pause for 30 minutes
            /snooze 2h        — pause for 2 hours
            /snooze 1d        — pause for a day (max 7d)
            /snooze off       — clear snooze
            /snooze           — show remaining time, or usage if not snoozed
        """
        if not self._check_user(update):
            return
        from datetime import datetime, timezone, timedelta

        args = " ".join(context.args).strip().lower() if context.args else ""

        if args in ("off", "0", "clear", "stop"):
            self.brain.db.set_snooze_until(None)
            await update.message.reply_text("✅ Snooze отключён.")
            return

        if not args:
            # Status query — show remaining snooze if any
            until = self.brain.db.get_snooze_until()
            if until is None:
                await update.message.reply_text(
                    "Сейчас не на паузе.\n\n"
                    "Использование:\n"
                    "/snooze 30m — пауза 30 минут\n"
                    "/snooze 2h — пауза 2 часа\n"
                    "/snooze 1d — пауза на день (макс 7d)\n"
                    "/snooze off — снять паузу\n\n"
                    "Напоминания продолжают работать (это твой явный intent).",
                )
                return
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            remaining_min = max(0, int((until - now_utc).total_seconds() // 60))
            if remaining_min < 60:
                rem_label = f"{remaining_min} мин"
            else:
                hours = remaining_min // 60
                mins = remaining_min % 60
                rem_label = f"{hours}ч {mins}м" if mins else f"{hours}ч"
            await update.message.reply_text(
                f"⏸ На паузе ещё {rem_label}. /snooze off — снять.",
            )
            return

        minutes = self._parse_snooze_duration(args)
        if minutes is None:
            await update.message.reply_text(
                "Неправильный формат.\n"
                "Примеры: /snooze 30m, /snooze 2h, /snooze 1d, /snooze off",
            )
            return

        # Compute UTC deadline — DB stores UTC-naive (matches the rest
        # of the proactive subsystem's clock convention).
        until = (
            datetime.now(timezone.utc).replace(tzinfo=None)
            + timedelta(minutes=minutes)
        )
        self.brain.db.set_snooze_until(until)

        # Render duration back at the user — confirms what they got.
        if minutes < 60:
            label = f"{minutes} мин"
        elif minutes < 60 * 24:
            hours = minutes // 60
            label = f"{hours}ч"
        else:
            days = minutes // (60 * 24)
            label = f"{days}д"
        await update.message.reply_text(
            f"⏸ Snooze на {label}. Напоминания продолжают работать. "
            f"/snooze off — снять.",
        )

    async def _handle_learnings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show accumulated learnings extracted by the weekly review.

        Weekly review (Sundays at 20:00) writes insights to
        memory.category='learning'. They quietly accumulate but were
        only reachable via `/memory learning` — assuming the user knew
        that category name. /learnings is the direct surface.
        """
        if not self._check_user(update):
            return
        learnings = self.brain.memory.get_by_category("learning")
        if not learnings:
            await update.message.reply_text(
                "Пока без learnings. Они появятся после первого "
                "недельного обзора (воскресенье 20:00) — или запусти "
                "сейчас командой /review.",
            )
            return

        # get_by_category already sorts by importance DESC. Cap at 10
        # so the message fits in one Telegram bubble even after many
        # weekly reviews.
        top = learnings[:10]
        lines = ["📚 *Learnings*\n"]
        tz_name = getattr(self.brain.db, "_timezone", None)
        for m in top:
            content = m["content"][:280]
            imp = m.get("importance") or 5
            conf = float(m.get("confidence") or 0.0)
            # created_at is UTC; convert to local then take the date
            # prefix. Pre-fix raw [:10] showed UTC date, which can be
            # off by a day for memories saved past local midnight on
            # positive UTC offsets.
            created_local = fmt_utc_to_local(
                m.get("created_at") or "", tz_name,
            )
            created = created_local[:10] if created_local != "?" else ""
            tail = f"_imp {imp}_"
            if conf:
                tail += f" · _conf {conf:.2f}_"
            if created:
                tail += f" · _{created}_"
            lines.append(f"• {content}\n  {tail}")
        if len(learnings) > 10:
            lines.append(f"\n_…и ещё {len(learnings) - 10}._")
        text = "\n\n".join(lines)
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text)

    async def _handle_about(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pre-meeting brief: who this person is, last contact, open
        promises, what to ask. Wires up the long-defined PRE_MEETING_PROMPT
        which had no caller before this command. User-invoked, costs an
        LLM call per use — same shape as /review."""
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        from ..core.prompt_safety import sanitize_for_context
        from ..llm.prompts import PRE_MEETING_PROMPT

        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text(
                "Использование: /about <имя>\nПример: /about Маша",
            )
            return
        if len(query) > 100:
            await update.message.reply_text("Слишком длинное имя.")
            return

        contacts = self.brain.db.get_contacts(query)
        if not contacts:
            await update.message.reply_text(
                f"Не нашёл контакта по '{query}'. "
                f"Если человек упоминался — попробуй /memory {query}.",
            )
            return

        # get_contacts is already ORDER BY mention_count DESC — pick the
        # most-frequently-mentioned match. Multi-match is normal for
        # common first names ("Маша" matches all Mashas).
        c = contacts[0]
        s = sanitize_for_context

        # Contact info block — name, relation, birthday, notes, recency.
        info_parts = [f"Имя: {s(c['name'], 100)}"]
        if c.get("relation"):
            info_parts.append(f"Связь: {s(c['relation'], 100)}")
        if c.get("birthday"):
            info_parts.append(f"ДР: {c['birthday']}")
        if c.get("last_contact"):
            info_parts.append(f"Последний контакт: {c['last_contact'][:16]}")
        if c.get("mention_count"):
            info_parts.append(f"Упоминаний: {c['mention_count']}")
        if c.get("notes"):
            info_parts.append(f"Заметки: {s(c['notes'], 500)}")
        contact_info = "\n".join(info_parts)

        # Relevant memories — semantic search by name plus the broader
        # context the LLM might cite (jobs, recent decisions, etc.).
        try:
            memories = await self.brain.memory.search(c["name"], top_k=8)
        except Exception:
            memories = []
        memories_text = "\n".join(
            f"- [{m['category']}] {s(m['content'], 220)}" for m in memories
        ) or "Релевантной памяти нет."

        # Open promises — semantically search inside the 'promise' category
        # to surface anything tied to this person. We don't strictly filter
        # by related_person (most promises don't set it) — the LLM can
        # judge relevance from content.
        try:
            promises = await self.brain.memory.search(
                f"обещание {c['name']}", category="promise", top_k=5,
            )
        except Exception:
            promises = []
        promises_text = "\n".join(
            f"- {s(p['content'], 220)}" for p in promises
        ) or "Незакрытых обещаний не нашёл."

        prompt = PRE_MEETING_PROMPT.format(
            contact_info=contact_info,
            memories=memories_text,
            promises=promises_text,
        )

        await self._typing(update)
        try:
            response = await self.brain.llm.chat(
                system=prompt,
                messages=[
                    {"role": "user",
                     "content": f"Подготовь брифинг про {c['name']}."},
                ],
                max_tokens=600,
            )
            text = (response.text or "").strip() or "Ничего не нашёл."
        except Exception as e:
            logger.error("about brief failed: %s", type(e).__name__)
            await update.message.reply_text("Не удалось подготовить брифинг.")
            return

        for part in _split_message(text):
            try:
                await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(part)

    async def _handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        await update.message.reply_text("⏳ Готовлю экспорт...")

        db = self.brain.db
        exported_at = tz_now(self.brain.profile.timezone).isoformat()
        # User-owned data: everything the user has personally created or
        # the bot has captured on their behalf. Excluded:
        # - ephemeral_state — transient by design (TTL ≤72h, no historic value)
        # - api_costs / preferences — bot-internal accounting, not user content
        # - memories.embedding — BLOB, expensive to serialize and useless
        #   without the original Voyage model+API; users can re-embed.
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
            "reminders": db.db.execute(
                "SELECT * FROM reminders ORDER BY trigger_at DESC LIMIT 500"
            ).fetchall(),
            "daily_goals": db.db.execute(
                "SELECT * FROM daily_goals ORDER BY date DESC, created_at DESC "
                "LIMIT 1000"
            ).fetchall(),
            "habits": db.db.execute(
                "SELECT * FROM habits ORDER BY name"
            ).fetchall(),
            "habit_log": db.db.execute(
                "SELECT * FROM habit_log ORDER BY date DESC LIMIT 2000"
            ).fetchall(),
            # Recent interactions only — full chat history can balloon past
            # Telegram's 50MB doc limit on heavy users. 2000 rows ≈ 40 days
            # at 50 interactions/day, which captures the practical window
            # users would want to migrate.
            "interactions": db.db.execute(
                "SELECT id, direction, message_type, content, "
                "voice_duration_sec, timestamp, metadata "
                "FROM interactions ORDER BY timestamp DESC LIMIT 2000"
            ).fetchall(),
        }
        # Convert sqlite3.Row objects to dicts
        for key in ("memories", "events", "decisions", "reminders",
                    "daily_goals", "habits", "habit_log", "interactions"):
            data[key] = [dict(r) for r in data[key]]

        json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        doc = io.BytesIO(json_bytes)
        doc.name = (
            f"mindsecretary_export_"
            f"{tz_now(self.brain.profile.timezone).strftime('%Y%m%d')}.json"
        )

        # Caption summarizes every category we exported. Pre-fix
        # only 6 of 9 were listed (events/decisions/diary/habit_log
        # silent), making users wonder if those were skipped. All
        # plurals also went through pluralize_ru — pre-fix "1 целей"
        # / "2 воспоминаний" / "3 контактов" all read wrong.
        cap_parts = [
            (len(data["memories"]),
             ("воспоминание", "воспоминания", "воспоминаний")),
            (len(data["contacts"]),
             ("контакт", "контакта", "контактов")),
            (len(data["events"]),
             ("событие", "события", "событий")),
            (len(data["reminders"]),
             ("напоминание", "напоминания", "напоминаний")),
            (len(data["habits"]),
             ("привычка", "привычки", "привычек")),
            (len(data["habit_log"]),
             ("отметка", "отметки", "отметок")),
            (len(data["daily_goals"]),
             ("цель", "цели", "целей")),
            (len(data["decisions"]),
             ("решение", "решения", "решений")),
            (len(data["diary"]),
             ("запись", "записи", "записей")),
            (len(data["interactions"]),
             ("сообщение", "сообщения", "сообщений")),
        ]
        caption = "📦 Экспорт: " + ", ".join(
            f"{n} {pluralize_ru(n, forms)}" for n, forms in cap_parts
        )
        await update.message.reply_document(document=doc, caption=caption)

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
        if not memory_id:
            await query.edit_message_text("Ошибка: ID не найден.")
            return
        # Memory.delete returns False on missing id / already-deleted
        # — race possible between /forget showing the prompt and the
        # user clicking Yes (LLM might have run delete_memory in
        # between, or another /forget action was confirmed first).
        # Pre-fix we showed "🗑 Удалено" regardless; surface the real
        # outcome instead.
        if self.brain.memory.delete(memory_id):
            await query.edit_message_text("🗑 Удалено.")
        else:
            await query.edit_message_text(
                "Уже удалено или не найдено."
            )

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

        # Post-download size check — same rationale as the photo handler:
        # voice.file_size can be None on the Telegram side, so the pre-
        # download guard skips silently. Whisper rejects >25MB anyway, but
        # a 50MB voice would chew memory before the API call and make the
        # error message less informative ("STT failed" vs "too big").
        if len(audio_bytes) > MAX_VOICE_SIZE:
            del audio_bytes
            await update.message.reply_text("Слишком большой файл (макс 25 МБ).")
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

        # Caption is stored as-is (possibly empty) on the interaction
        # row so history replay reads back as the user's actual
        # message. Brain.process injects the inbox-capture instruction
        # at the LLM-facing layer when caption is empty — see
        # PHOTO_DEFAULT_INSTRUCTION in brain.py.
        caption = (msg.caption or "")[:MAX_TEXT_LENGTH]
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

        # Post-download size check: pre-fix only photo.file_size was checked,
        # which Telegram doesn't always populate (`if photo.file_size and …`
        # silently skips when None). A missing header could let an oversized
        # photo through, base64-encode at 4/3 the size, and ship it to Claude
        # — paying for tokens we set the limit to avoid.
        if len(photo_bytes) > MAX_PHOTO_SIZE:
            del photo_bytes
            await msg.reply_text("Фото слишком большое (макс 10 МБ).")
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
        # Reject empty AND whitespace-only — voice/forward already do this,
        # but plain text used to slip through. A "   " send would log a
        # nearly-empty interaction and pay for an LLM round on a message
        # the user clearly didn't intend.
        if not text or not text.strip():
            return
        if len(text) > MAX_TEXT_LENGTH:
            # Mirror the voice handler — pre-fix a 12k-char paste was
            # silently clipped to 10k and the LLM only saw the first
            # half, so its reply ignored the rest with no signal to
            # the user that anything was dropped.
            await update.message.reply_text(
                f"Сообщение длинное ({len(text)} симв), "
                f"обрезаю до {MAX_TEXT_LENGTH}."
            )
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

    @staticmethod
    def _forward_attribution(origin) -> str:
        """Render the "[Переслано от X]: " prefix for a forwarded message.

        Telegram's forward_origin can carry sender_user (regular user),
        sender_user_name (privacy-protected user), or sender_chat (channel
        forward). Defensive try/except so an unexpected origin shape
        doesn't crash the handler — falls back to a generic [Переслано].
        """
        if not origin:
            return "[Переслано]: "
        try:
            if hasattr(origin, "sender_user") and origin.sender_user:
                u = origin.sender_user
                name = u.first_name or ""
                if u.last_name:
                    name += f" {u.last_name}"
                return f"[Переслано от {name}]: "
            if hasattr(origin, "sender_user_name") and origin.sender_user_name:
                return f"[Переслано от {origin.sender_user_name}]: "
            if hasattr(origin, "sender_chat") and origin.sender_chat:
                return f"[Переслано из {origin.sender_chat.title}]: "
        except Exception:
            pass
        return "[Переслано]: "

    async def _handle_forward(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        if not await self._require_rate_limit(update):
            return
        msg = update.message
        forward_from = self._forward_attribution(msg.forward_origin)

        # Forwarded PHOTO branch: filters.FORWARDED is registered before
        # filters.PHOTO so forwarded screenshots / receipts / memes land
        # here, NOT in _handle_photo. Pre-fix this branch fell through
        # to text-only and the image was dropped silently — user
        # forwarded a receipt expecting OCR and bot saw "[Переслано
        # от Маша]: " (caption-only) and answered nonsense.
        if msg.photo:
            photo = msg.photo[-1]
            if photo.file_size and photo.file_size > MAX_PHOTO_SIZE:
                await msg.reply_text("Фото слишком большое (макс 10 МБ).")
                return
            caption_body = (msg.caption or "")[:MAX_TEXT_LENGTH].strip()
            # Caption is optional on forwards. We log the user's actual
            # caption (possibly empty) along with the [Переслано]
            # prefix; Brain.process injects PHOTO_DEFAULT_INSTRUCTION
            # at the LLM-facing layer when the caption body is empty,
            # so history replay reads cleanly.
            full_caption = (
                f"{forward_from}{caption_body}" if caption_body
                else forward_from.rstrip()
            )
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
                logger.error("Forwarded photo download failed: %s", type(e).__name__)
                await msg.reply_text("Не удалось скачать фото.")
                return
            if len(photo_bytes) > MAX_PHOTO_SIZE:
                del photo_bytes
                await msg.reply_text("Фото слишком большое (макс 10 МБ).")
                return
            image_b64 = base64.b64encode(photo_bytes).decode("utf-8")
            del photo_bytes
            logger.info(
                "Forwarded photo received (%d KB), caption: %s",
                len(image_b64) // 1024, caption_body[:50],
            )
            await self._typing(update)
            try:
                response = await asyncio.wait_for(
                    self.brain.process(
                        user_message=full_caption, message_type="photo",
                        image_base64=image_b64,
                    ),
                    timeout=self.brain.settings.process_timeout_sec,
                )
                await self._reply(update, response.text)
            except asyncio.TimeoutError:
                await msg.reply_text("Таймаут обработки фото.")
            except Exception as e:
                logger.error("Brain failed on forwarded photo: %s", type(e).__name__)
                await msg.reply_text("Не удалось обработать фото.")
            return

        raw = msg.text or msg.caption or ""
        text = raw[:MAX_TEXT_LENGTH]
        # Pre-fix the empty-check used full_text, which always has the
        # "[Переслано]:" prefix and was therefore never empty. So a
        # forwarded photo without a caption (or a sticker forward) became
        # "[Переслано]: " and shipped to Brain.process — paying for an
        # LLM round on what's effectively nothing. Check the content
        # portion explicitly.
        if not text.strip():
            return
        if len(raw) > MAX_TEXT_LENGTH:
            # Same UX gap as plain text — silent clip on long forwards
            # (long article forwarded for summary) hid which half got
            # processed. Warn before dispatching to Brain.
            await update.message.reply_text(
                f"Пересланный текст длинный ({len(raw)} симв), "
                f"обрезаю до {MAX_TEXT_LENGTH}."
            )
        full_text = f"{forward_from}{text}"

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
        self.app.add_handler(CommandHandler("version", self._handle_version))
        self.app.add_handler(CommandHandler("about", self._handle_about))
        self.app.add_handler(CommandHandler("learnings", self._handle_learnings))
        self.app.add_handler(CommandHandler("snooze", self._handle_snooze))

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
