from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
            notification_limit=int(os.environ.get("PROFILE_NOTIFY_LIMIT", "5")),
            quiet_hours=[h.strip() for h in quiet_raw.split(",")],
            priorities=[p.strip() for p in prio_raw.split(",")],
            dislikes=[d.strip() for d in dislike_raw.split(",")],
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

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        llm = raw["llm"]
        stt = raw["stt"]
        emb = raw["embeddings"]
        mem = raw["memory"]
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
        load_dotenv(root / ".env")

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
