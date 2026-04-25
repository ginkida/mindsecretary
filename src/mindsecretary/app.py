from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .core.brain import Brain
from .core.config import AppConfig
from .core.database import Database
from .core.memory import Memory
from .integrations.weather import WeatherClient
from .interfaces.telegram import TelegramBot
from .llm.client import AnthropicClient
from .learning.reflection import WeeklyReflection
from .proactive.briefing import BriefingGenerator
from .proactive.scheduler import ProactiveScheduler
from .proactive.smart_questions import SmartQuestions
from .voice.stt import GroqSTT


def _setup_logging(log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "mindsecretary.log"

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Suppress noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


logger = logging.getLogger("mindsecretary")


def main():
    # --- Config ---
    try:
        config = AppConfig.load()
    except FileNotFoundError as e:
        print(f"Config not found: {e}")
        print("Copy .env.example to .env and edit config/profile.yaml")
        sys.exit(1)
    except KeyError as e:
        print(f"Missing env var: {e}")
        print("Check your .env file against .env.example")
        sys.exit(1)

    _setup_logging(config.project_root / "data")

    # --- Database ---
    logger.info("Initializing database...")
    db = Database(config.db_path, timezone=config.profile.timezone)

    # --- Memory (shared DB connection) ---
    logger.info("Initializing memory (Voyage AI)...")
    memory = Memory(
        db=db.db,
        voyage_api_key=config.voyage_api_key,
        model=config.settings.embedding_model,
        relevance_weight=config.settings.relevance_weight,
        importance_weight=config.settings.importance_weight,
    )

    # --- Weather ---
    logger.info("Initializing weather (Open-Meteo)...")
    weather = WeatherClient(
        latitude=config.profile.home_coords[0],
        longitude=config.profile.home_coords[1],
        timezone=config.profile.timezone,
    )

    # --- LLM ---
    logger.info("Initializing LLM (Claude %s)...", config.settings.model)
    client = AnthropicClient(
        api_key=config.anthropic_api_key,
        model=config.settings.model,
    )

    # --- STT ---
    logger.info("Initializing STT (Groq Whisper)...")
    stt = GroqSTT(
        api_key=config.groq_api_key,
        model=config.settings.stt_model,
        language=config.settings.stt_language,
    )

    # --- Brain ---
    logger.info("Initializing Brain...")
    brain = Brain(
        llm=client,
        memory=memory,
        db=db,
        profile=config.profile,
        settings=config.settings,
    )
    # Inject weather into tool executor
    brain.tool_executor.weather = weather

    # --- Telegram Bot ---
    logger.info("Building Telegram bot...")
    bot = TelegramBot(
        token=config.telegram_token,
        allowed_user_id=config.telegram_user_id,
        brain=brain,
        stt=stt,
    )
    app = bot.build()

    # --- Proactive Scheduler (starts inside event loop via post_init) ---
    proactive = ProactiveScheduler(
        db=db,
        weather=weather,
        profile=config.profile,
        settings=config.settings,
        send_fn=bot.send_message,
    )
    briefing = BriefingGenerator(
        llm=client,
        memory=memory,
        db=db,
        weather=weather,
        profile=config.profile,
    )
    proactive.briefing_generator = briefing
    proactive.weekly_reflection = WeeklyReflection(
        llm=client, memory=memory, db=db, profile=config.profile,
    )
    proactive.smart_questions = SmartQuestions(
        llm=client, memory=memory, db=db,
        profile=config.profile,
        min_interactions=config.settings.smart_question_min_interactions,
    )

    # Give bot access to scheduler for /review command
    bot._scheduler = proactive

    async def post_init(application):
        proactive.start()
        logger.info("Proactive scheduler started.")

    async def post_shutdown(application):
        proactive.stop()
        await weather.close()
        db.close()
        logger.info("Shutdown complete.")

    app.post_init = post_init
    app.post_shutdown = post_shutdown

    # --- Run ---
    logger.info(
        "MindSecretary starting for %s (%s). "
        "Memories: %d, Contacts: %d.",
        config.profile.name,
        config.profile.city,
        memory.count(),
        len(db.get_contacts("")),
    )
    app.run_polling()


if __name__ == "__main__":
    main()
