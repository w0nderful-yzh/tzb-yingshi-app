from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "老年安全监测后端"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_enabled: bool = False
    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://app:change-me@127.0.0.1:5432/tzb_yingshi"
    )
    database_echo: bool = False
    ys7_signal_enabled: bool = False
    ys7_webhook_token: SecretStr | None = None
    ys7_queue_maxsize: int = Field(default=1000, ge=1, le=100_000)
    ys7_raw_event_dir: Path = REPOSITORY_ROOT / "backend/storage/ys7/raw"

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
