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
    ys7_media_enabled: bool = False
    ys7_app_key: SecretStr | None = None
    ys7_app_secret: SecretStr | None = None
    ys7_access_token: SecretStr | None = None
    ys7_device_serial: str | None = None
    ys7_channel_no: int = Field(default=1, ge=1)
    ys7_live_protocol: Literal["hls", "rtmp", "flv"] = "flv"
    ys7_live_quality: int = Field(default=2, ge=1, le=3)
    ys7_media_chunk_ms: int = Field(default=5_000, ge=1_000, le=15_000)
    ys7_media_queue_maxsize: int = Field(default=4, ge=1, le=20)
    ys7_elder_alone: bool = False
    sensevoice_enabled: bool = False
    sensevoice_model: str = "iic/SenseVoiceSmall"
    sensevoice_device: str = "cpu"
    sensevoice_max_chunk_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=50 * 1024 * 1024,
    )

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
