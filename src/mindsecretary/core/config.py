from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Resolve the project root — where config/, data/, migrations/ live.

    Resolution order:
    1. MINDSECRETARY_ROOT env var (set by Dockerfile)
    2. cwd if it has a recognizable marker (config/, data/, migrations/, pyproject.toml)
    3. source-relative parents[3] — only works for editable installs (`pip install -e`)

    Plain `pip install` puts the package in site-packages where parents[3]
    resolves to `/usr/local/lib/python3.XX` — not the project root. Docker
    uses WORKDIR=/app with bind-mounted config/ and data/, so the env var
    makes the resolution explicit and predictable.
    """
    env_root = os.environ.get("MINDSECRETARY_ROOT")
    if env_root:
        return Path(env_root)
    cwd = Path.cwd()
    for marker in ("config", "data", "migrations", "pyproject.toml"):
        if (cwd / marker).exists():
            return cwd
    return Path(__file__).resolve().parents[3]


def _check_env_permissions(env_file: Path) -> None:
    """Warn if .env permissions allow group/world access to API keys."""
    if not env_file.exists() or os.name != "posix":
        return
    mode = stat.S_IMODE(env_file.stat().st_mode)
    if mode & 0o077:
        logger.warning(
            ".env permissions are %o — API keys readable beyond owner. "
            "Run: chmod 600 %s",
            mode, env_file,
        )


@dataclass
class Profile:
    name: str
    city: str
    timezone: str
    home_coords: list[float]
    work_coords: list[float]
    wake_up: str
    work_start: str
    work_end: str
    sleep: str
    commute_method: str
    commute_minutes: int
    style: str
    language: str
    notification_limit: int
    quiet_hours: list[str]
    priorities: list[str]
    dislikes: list[str]
    # Work days (ISO weekday: Mon=1..Sun=7). Used with work_start/work_end
    # to derive implicit "at work" ephemeral state during working hours.
    work_days: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    @classmethod
    def from_env(cls) -> Profile | None:
        """Build profile from env vars. Returns None if PROFILE_NAME not set."""
        name = os.environ.get("PROFILE_NAME")
        if not name:
            return None
        coords_raw = os.environ.get("PROFILE_HOME_COORDS", "55.7558,37.6173")
        coords = [float(x.strip()) for x in coords_raw.split(",")]
        work_raw = os.environ.get("PROFILE_WORK_COORDS", coords_raw)
        work_coords = [float(x.strip()) for x in work_raw.split(",")]
        prio_raw = os.environ.get("PROFILE_PRIORITIES", "здоровье,семья,работа,развитие")
        dislike_raw = os.environ.get("PROFILE_DISLIKES", "опаздывать,пустая болтовня,лишние уведомления")
        quiet_raw = os.environ.get("PROFILE_QUIET_HOURS", "23:00,07:00")
        work_days_raw = os.environ.get("PROFILE_WORK_DAYS", "1,2,3,4,5")
        try:
            work_days = [int(x.strip()) for x in work_days_raw.split(",") if x.strip()]
            work_days = [d for d in work_days if 1 <= d <= 7]
        except ValueError:
            work_days = [1, 2, 3, 4, 5]
        return cls(
            name=name,
            city=os.environ.get("PROFILE_CITY", "Москва"),
            timezone=os.environ.get("PROFILE_TIMEZONE", "Europe/Moscow"),
            home_coords=coords,
            work_coords=work_coords,
            wake_up=os.environ.get("PROFILE_WAKE_UP", "07:00"),
            work_start=os.environ.get("PROFILE_WORK_START", "09:00"),
            work_end=os.environ.get("PROFILE_WORK_END", "18:00"),
            sleep=os.environ.get("PROFILE_SLEEP", "23:00"),
            commute_method=os.environ.get("PROFILE_COMMUTE", "метро"),
            commute_minutes=int(os.environ.get("PROFILE_COMMUTE_MIN", "45")),
            style=os.environ.get("PROFILE_STYLE", "кратко, по делу"),
            language=os.environ.get("PROFILE_LANGUAGE", "ru"),
            notification_limit=int(os.environ.get("PROFILE_NOTIFY_LIMIT", "10")),
            quiet_hours=[h.strip() for h in quiet_raw.split(",")],
            priorities=[p.strip() for p in prio_raw.split(",")],
            dislikes=[d.strip() for d in dislike_raw.split(",")],
            work_days=work_days or [1, 2, 3, 4, 5],
        )

    @classmethod
    def from_yaml(cls, path: Path) -> Profile:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        ident = raw["identity"]
        sched = raw["schedule"]
        comm = raw["communication"]
        return cls(
            name=ident["name"],
            city=ident["city"],
            timezone=ident["timezone"],
            home_coords=ident["home_coords"],
            work_coords=ident["work_coords"],
            wake_up=sched["wake_up"],
            work_start=sched["work_start"],
            work_end=sched["work_end"],
            sleep=sched["sleep"],
            commute_method=sched["commute_method"],
            commute_minutes=sched["commute_minutes"],
            style=comm["style"],
            language=comm["language"],
            notification_limit=comm["notification_limit"],
            quiet_hours=comm["quiet_hours"],
            priorities=raw.get("priorities", []),
            dislikes=raw.get("dislikes", []),
            work_days=sched.get("work_days", [1, 2, 3, 4, 5]),
        )

    @classmethod
    def load(cls, root: Path) -> Profile:
        """Load profile: env vars take priority, then YAML fallback."""
        profile = cls.from_env()
        if profile:
            return profile
        yaml_path = root / "config" / "profile.yaml"
        if yaml_path.exists():
            return cls.from_yaml(yaml_path)
        raise FileNotFoundError(
            "Set PROFILE_NAME env var or create config/profile.yaml"
        )

    def to_yaml_str(self) -> str:
        lines = [
            f"Имя: {self.name}",
            f"Город: {self.city}",
            f"Подъём: {self.wake_up}, работа: {self.work_start}-{self.work_end}",
            f"Дорога: {self.commute_method}, {self.commute_minutes} мин",
            f"Стиль общения: {self.style}",
            f"Приоритеты: {', '.join(self.priorities)}",
            f"Не любит: {', '.join(self.dislikes)}",
        ]
        return "\n".join(lines)


@dataclass
class Settings:
    model: str
    max_tokens: int
    max_tool_rounds: int
    stt_model: str
    stt_language: str
    embedding_model: str
    memory_top_k: int
    relevance_weight: float
    importance_weight: float
    # Proactive job toggles (all default on)
    morning_briefing: bool = True
    evening_summary: bool = True
    smart_questions: bool = True
    decision_followups: bool = True
    weekly_review: bool = True
    weather_monitor: bool = True
    birthday_alerts: bool = True
    event_alerts: bool = True
    # Tunable intervals and thresholds
    reminder_check_minutes: int = 5
    weather_check_minutes: int = 60
    # Pre-event alerts: fire `event_alert_lead_minutes` before each calendar
    # event's start_at, scanning every `event_alert_check_minutes` ticks.
    # Lead must be >= check, otherwise events that fall in [now, now+lead]
    # at one tick land in the past at the next and are skipped.
    event_alert_lead_minutes: int = 15
    event_alert_check_minutes: int = 5
    process_timeout_sec: int = 90
    quiet_contact_days: int = 30
    quiet_contact_min_mentions: int = 3
    smart_question_min_interactions: int = 5
    # Cost circuit breaker — hard cap on daily spend across all APIs
    daily_cost_limit_usd: float = 5.0
    # Rate limiting — max messages per minute for the authorized user
    rate_limit_per_minute: int = 20
    # Data retention — delete interactions/api_costs older than this
    data_retention_days: int = 90

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        llm = raw["llm"]
        stt = raw["stt"]
        emb = raw["embeddings"]
        mem = raw["memory"]
        proactive = raw.get("proactive", {})
        tuning = raw.get("tuning", {})
        return cls(
            model=llm["model"],
            max_tokens=llm["max_tokens"],
            max_tool_rounds=llm["max_tool_rounds"],
            stt_model=stt["model"],
            stt_language=stt["language"],
            embedding_model=emb["model"],
            memory_top_k=mem["search_top_k"],
            relevance_weight=mem["relevance_weight"],
            importance_weight=mem["importance_weight"],
            morning_briefing=proactive.get("morning_briefing", True),
            evening_summary=proactive.get("evening_summary", True),
            smart_questions=proactive.get("smart_questions", True),
            decision_followups=proactive.get("decision_followups", True),
            weekly_review=proactive.get("weekly_review", True),
            weather_monitor=proactive.get("weather_monitor", True),
            birthday_alerts=proactive.get("birthday_alerts", True),
            event_alerts=proactive.get("event_alerts", True),
            reminder_check_minutes=tuning.get("reminder_check_minutes", 5),
            weather_check_minutes=tuning.get("weather_check_minutes", 60),
            event_alert_lead_minutes=tuning.get("event_alert_lead_minutes", 15),
            event_alert_check_minutes=tuning.get("event_alert_check_minutes", 5),
            process_timeout_sec=tuning.get("process_timeout_sec", 90),
            quiet_contact_days=tuning.get("quiet_contact_days", 30),
            quiet_contact_min_mentions=tuning.get("quiet_contact_min_mentions", 3),
            smart_question_min_interactions=tuning.get("smart_question_min_interactions", 5),
            daily_cost_limit_usd=tuning.get("daily_cost_limit_usd", 5.0),
            rate_limit_per_minute=tuning.get("rate_limit_per_minute", 20),
            data_retention_days=tuning.get("data_retention_days", 90),
        )


@dataclass
class AppConfig:
    profile: Profile
    settings: Settings
    anthropic_api_key: str
    groq_api_key: str
    voyage_api_key: str
    telegram_token: str
    telegram_user_id: int
    db_path: Path
    project_root: Path

    @classmethod
    def load(cls, root: Path | None = None) -> AppConfig:
        root = root or _project_root()
        env_file = root / ".env"
        load_dotenv(env_file)
        _check_env_permissions(env_file)

        profile = Profile.load(root)
        settings = Settings.from_yaml(root / "config" / "settings.yaml")

        db_dir = root / "data"
        db_dir.mkdir(exist_ok=True)

        return cls(
            profile=profile,
            settings=settings,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            groq_api_key=os.environ["GROQ_API_KEY"],
            voyage_api_key=os.environ["VOYAGE_API_KEY"],
            telegram_token=os.environ["TELEGRAM_TOKEN"],
            telegram_user_id=int(os.environ["TELEGRAM_USER_ID"]),
            db_path=db_dir / "mindsecretary.db",
            project_root=root,
        )
