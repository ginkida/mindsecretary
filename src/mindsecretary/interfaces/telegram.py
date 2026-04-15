from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from datetime import datetime, timedelta

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

from ..core.brain import Brain
from ..core.enums import Feedback
from ..voice.stt import GroqSTT

logger = logging.getLogger(__name__)

MAX_VOICE_SIZE = 25 * 1024 * 1024
MAX_VOICE_DURATION = 600
MAX_PHOTO_SIZE = 10 * 1024 * 1024
MAX_TEXT_LENGTH = 10_000
DOWNLOAD_TIMEOUT = 30.0
TG_MSG_LIMIT = 4096

FEEDBACK_KB = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("👍", callback_data="fb_positive"),
        InlineKeyboardButton("👎", callback_data="fb_negative"),
    ]
])


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


class TelegramBot:
    def __init__(self, token: str, allowed_user_id: int,
                 brain: Brain, stt: GroqSTT):
        self.token = token
        self.allowed_user_id = allowed_user_id
        self.brain = brain
        self.stt = stt
        self.app: Application | None = None
        # For feedback tracking: map message_id → interaction_id
        self._reply_map: dict[int, str] = {}

    def _check_user(self, update: Update) -> bool:
        uid = update.effective_user.id if update.effective_user else 0
        if uid != self.allowed_user_id:
            logger.warning("Unauthorized: user %s", uid)
            return False
        return True

    async def _reply(self, update: Update, text: str, with_feedback: bool = True):
        """Send reply with optional feedback buttons and message splitting."""
        parts = _split_message(text)
        for i, part in enumerate(parts):
            kb = FEEDBACK_KB if (with_feedback and i == len(parts) - 1) else None
            try:
                sent = await update.message.reply_text(
                    _fix_markdown(part), reply_markup=kb, parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                # Fallback without markdown if parsing fails
                sent = await update.message.reply_text(part, reply_markup=kb)
            # Track last reply for feedback
            if kb and sent:
                interaction_id = self.brain.db.log_interaction(
                    direction="out", message_type="reply_ref", content="",
                )
                if len(self._reply_map) >= 200:
                    # Keep only the newest 100 entries (dict preserves insertion order)
                    self._reply_map = dict(list(self._reply_map.items())[-100:])
                self._reply_map[sent.message_id] = interaction_id

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
            "/export — экспорт данных в JSON\n"
            "/review — запустить недельный обзор\n"
            "/forget — удалить воспоминание",
        )

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        stats = self.brain.db.get_stats()
        await update.message.reply_text(
            f"📊 *Статистика*\n\n"
            f"💰 Сегодня: ${stats['today_cost']:.4f} ({stats['today_tokens']:,} tokens)\n"
            f"💰 Месяц: ${stats['month_cost']:.4f}\n\n"
            f"🧠 Воспоминаний: {stats['memories']}\n"
            f"👤 Контактов: {stats['contacts']}\n"
            f"💬 Взаимодействий сегодня: {stats['interactions_today']}",
            parse_mode=ParseMode.MARKDOWN,
        )

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
        for c in contacts[:20]:
            line = f"• *{c['name']}*"
            if c.get("relation"):
                line += f" ({c['relation']})"
            if c.get("last_contact"):
                try:
                    last = datetime.fromisoformat(c["last_contact"])
                    days = (datetime.now() - last).days
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
        await update.message.reply_text("⏳ Генерирую обзор...")
        await self._typing(update)
        # Access weekly_reflection through the scheduler (set in app.py)
        scheduler = getattr(self, "_scheduler", None)
        if scheduler and scheduler.weekly_reflection:
            text = await scheduler.weekly_reflection.generate_weekly_review()
            if text:
                await self._reply(update, text, with_feedback=False)
                return
        await update.message.reply_text("Недостаточно данных для обзора.")

    async def _handle_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
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

    async def _handle_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
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
        await update.message.reply_text(
            f"Удалить это?\n\n_{top['content'][:300]}_",
            reply_markup=confirm_kb,
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
            return
        await update.message.reply_text("⏳ Готовлю экспорт...")

        db = self.brain.db
        data = {
            "exported_at": datetime.now().isoformat(),
            "memories": db.db.execute(
                "SELECT id, content, category, importance, related_person, "
                "related_date, created_at FROM memories WHERE status = 'active' "
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
        doc.name = f"mindsecretary_export_{datetime.now().strftime('%Y%m%d')}.json"

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

    # --- Feedback callback ---

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

    async def _handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if not self._check_user(update):
            return
        msg_id = query.message.message_id
        interaction_id = self._reply_map.get(msg_id)
        if not interaction_id:
            return
        feedback = Feedback.POSITIVE if query.data == "fb_positive" else Feedback.NEGATIVE
        self.brain.db.db.execute(
            "UPDATE interactions SET feedback = ?, feedback_at = datetime('now') WHERE id = ?",
            (feedback, interaction_id),
        )
        self.brain.db.db.commit()
        emoji = "👍" if feedback == "positive" else "👎"
        await query.edit_message_reply_markup(reply_markup=None)
        logger.info("Feedback: %s for interaction %s", feedback, interaction_id)

    # --- Message handlers ---

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_user(update):
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
        msg = update.message
        if not msg.photo:
            return

        # Get the largest photo
        photo = msg.photo[-1]
        if photo.file_size and photo.file_size > MAX_PHOTO_SIZE:
            await msg.reply_text("Фото слишком большое (макс 10 МБ).")
            return

        caption = (msg.caption or "Что на этом фото?")[:MAX_TEXT_LENGTH]
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
        self.app.add_handler(CommandHandler("forget", self._handle_forget))
        self.app.add_handler(CommandHandler("goals", self._handle_goals))
        self.app.add_handler(CommandHandler("habits", self._handle_habits))
        self.app.add_handler(CommandHandler("export", self._handle_export))

        # Feedback buttons
        self.app.add_handler(CallbackQueryHandler(self._handle_feedback, pattern="^fb_"))
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
