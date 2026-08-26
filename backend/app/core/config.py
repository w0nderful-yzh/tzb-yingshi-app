from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
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
    demo_elder_subject: str = "u-elder-001"
    demo_guardian_subject: str = "u-family-001"
    demo_elder_name: str = "演示老人"
    demo_guardian_name: str = "演示家属"
    demo_elder_login: str = "elder"
    demo_guardian_login: str = "guardian"
    demo_elder_password: SecretStr = SecretStr("elder123")
    demo_guardian_password: SecretStr = SecretStr("guardian123")
    auth_session_ttl_hours: int = Field(default=168, ge=1, le=24 * 90)
    ys7_signal_enabled: bool = False
    ys7_webhook_token: SecretStr | None = None
    ys7_queue_maxsize: int = Field(default=1000, ge=1, le=100_000)
    ys7_raw_event_dir: Path = REPOSITORY_ROOT / "backend/storage/ys7/raw"
    ys7_alarm_poll_enabled: bool = False
    ys7_alarm_poll_interval_seconds: float = Field(default=5.0, ge=5.0, le=3600.0)
    ys7_alarm_poll_lookback_seconds: int = Field(default=120, ge=30, le=3600)
    ys7_alarm_poll_page_size: int = Field(default=50, ge=1, le=50)
    ys7_media_enabled: bool = False
    ys7_media_source: Literal["cloud", "app_relay"] = "cloud"
    ys7_app_key: SecretStr | None = None
    ys7_app_secret: SecretStr | None = None
    ys7_access_token: SecretStr | None = None
    ys7_device_serial: str | None = None
    ys7_channel_no: int = Field(default=1, ge=1)
    ys7_live_protocol: Literal["hls", "rtmp", "flv"] = "flv"
    ys7_live_quality: int = Field(default=2, ge=1, le=3)
    ys7_media_queue_maxsize: int = Field(default=32, ge=4, le=200)
    ys7_pcm_relay_queue_maxsize: int = Field(default=8, ge=2, le=100)
    ys7_vad_mode: int = Field(default=2, ge=0, le=3)
    ys7_vad_speech_start_ms: int = Field(default=200, ge=20, le=1_000)
    ys7_vad_silence_end_ms: int = Field(default=700, ge=100, le=5_000)
    streaming_chunk_ms: int = Field(default=600, ge=200, le=3_000)
    ys7_elder_alone: bool = False
    sensevoice_enabled: bool = False
    sensevoice_model: str = "iic/SenseVoiceSmall"
    sensevoice_device: str = "cpu"
    sensevoice_max_chunk_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=50 * 1024 * 1024,
    )
    streaming_asr_enabled: bool = False
    streaming_asr_model: str = "paraformer-zh-streaming"
    streaming_asr_device: str = "cpu"
    streaming_asr_hotwords: str = "验证码 安全账户 屏幕共享 远程控制 涉案资金 转账 汇款 取现"
    streaming_asr_hotword_corrections: dict[str, str] = Field(default_factory=dict)
    fraud_llm_enabled: bool = False
    fraud_llm_base_url: str | None = None
    fraud_llm_api_key: SecretStr | None = None
    fraud_llm_model: str | None = None
    fraud_llm_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    fraud_llm_enable_thinking: bool | None = None
    fraud_llm_queue_maxsize: int = Field(default=32, ge=1, le=1_000)
    fraud_llm_trigger_state_index: int = Field(default=2, ge=1, le=4)
    fraud_llm_max_transcript_chars: int = Field(default=6_000, ge=500, le=20_000)
    fraud_llm_vision_enabled: bool = False
    fraud_llm_max_images: int = Field(default=4, ge=1, le=8)
    fraud_latency_trace_enabled: bool = False
    fraud_classifier_warmup_enabled: bool = True
    sensevoice_warmup_enabled: bool = True
    streaming_asr_warmup_enabled: bool = True
    fraud_preliminary_alert_enabled: bool = False
    fraud_preliminary_min_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    fraud_preliminary_stable_revisions: int = Field(default=2, ge=1, le=10)
    fraud_preliminary_confirm_min_state_index: int = Field(default=2, ge=0, le=5)
    fraud_semantic_retriever_enabled: bool = False
    fraud_recent_risk_enabled: bool = False
    fall_risk_enabled: bool = False
    # Existing multimodal prototype base URL (living-room Camera-led C path).
    fall_risk_base_url: str | None = None
    # Each Radar process represents one RADAR_ROOM and exposes /api/radar/latest.
    fall_risk_radar_room_urls: dict[str, str] = Field(default_factory=dict)
    fall_risk_api_key: SecretStr | None = None
    fall_risk_timeout_seconds: float = Field(default=2.0, ge=0.1, le=30.0)
    fall_risk_camera_led_path: str = "/api/multimodal/camera-led-associated/latest"
    fall_risk_radar_only_path: str = "/api/radar/latest"
    psychology_enabled: bool = False
    psychology_base_url: str | None = None
    psychology_api_key: SecretStr | None = None
    psychology_timeout_seconds: float = Field(default=2.0, ge=0.1, le=30.0)
    psychology_latest_path: str = "/api/psychology/assessments/latest"
    psychology_observation_interval_seconds: float = Field(
        default=900.0,
        ge=60.0,
        le=24 * 3600.0,
    )

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("ys7_vad_speech_start_ms", "ys7_vad_silence_end_ms", "streaming_chunk_ms")
    @classmethod
    def require_frame_aligned_vad_timing(cls, value: int) -> int:
        if value % 20 != 0:
            raise ValueError("VAD/streaming timing must be a multiple of the 20 ms frame")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
