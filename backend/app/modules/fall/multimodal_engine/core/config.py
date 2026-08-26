from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


ENGINE_DIR = Path(__file__).resolve().parents[1]
# Keep this alias while the migrated engine is made repository-native.  All
# runtime output stays below the engine and is ignored by Git.
BACKEND_DIR = ENGINE_DIR
REPOSITORY_DIR = (
    ENGINE_DIR.parents[4] if len(ENGINE_DIR.parents) > 4 else ENGINE_DIR.parent
)
# The competition workspace still owns large Camera assets and local SDKs.
# Every path remains overridable through the existing Settings environment
# fields when the repository is deployed on another machine.
WORKSPACE_DIR = (
    REPOSITORY_DIR.parents[1]
    if len(REPOSITORY_DIR.parents) > 1
    else REPOSITORY_DIR.parent
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Keep the algorithm service isolated from the App backend's database
        # and deployment settings. Machine-local Camera/Radar secrets and
        # overrides belong only in multimodal_engine/.env (ignored by Git).
        env_file=ENGINE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Elder Risk Prototype"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = True
    demo_mode: bool = True

    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_database: str = "elder_risk_prototype"
    mysql_username: str = "elder_risk_app"
    mysql_password: SecretStr = SecretStr("")

    db_pool_size: int = Field(default=5, ge=1, le=20)
    db_max_overflow: int = Field(default=5, ge=0, le=20)
    db_pool_recycle_seconds: int = Field(default=1800, ge=60)

    frontend_origin: str = "http://127.0.0.1:8088"

    ezviz_base_url: str = "https://open.ys7.com/api/lapp"
    ezviz_app_key: SecretStr = SecretStr("")
    ezviz_app_secret: SecretStr = SecretStr("")
    ezviz_device_verify_code: SecretStr = SecretStr("")
    ezviz_request_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    ezviz_token_refresh_skew_seconds: int = Field(default=60, ge=0, le=600)

    radar_integration_enabled: bool = True
    radar_service_url: str = "http://127.0.0.1:8010"
    radar_poll_interval_seconds: float = Field(default=0.05, gt=0, le=60)
    radar_request_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    radar_risk_events_enabled: bool = True
    radar_formal_predictions_enabled: bool = False

    # Phase-1 multimodal decision layer. These values affect only the new
    # fusion score and never alter either single-modality model.
    fusion_camera_weight: float = Field(default=0.6, gt=0, le=1)
    fusion_radar_weight: float = Field(default=0.4, gt=0, le=1)
    fusion_sync_tolerance_seconds: float = Field(default=2.0, gt=0, le=30)
    fusion_medium_threshold: float = Field(default=0.35, gt=0, lt=1)
    fusion_high_threshold: float = Field(default=0.65, gt=0, lt=1)
    fusion_default_method: Literal[
        "fixed_weighted",
        "quality_weighted",
        "radar_quality_adaptive",
    ] = (
        "fixed_weighted"
    )
    fusion_ema_alpha: float = Field(default=0.35, gt=0, le=1)
    fusion_watch_exit_threshold: float = Field(default=0.25, ge=0, lt=1)
    fusion_high_exit_threshold: float = Field(default=0.50, ge=0, lt=1)
    fusion_imminent_threshold: float = Field(default=0.80, gt=0, le=1)
    fusion_watch_confirmation_windows: int = Field(default=2, ge=1, le=20)
    fusion_high_confirmation_windows: int = Field(default=3, ge=1, le=20)
    fusion_normal_confirmation_windows: int = Field(default=2, ge=1, le=20)
    fusion_conflict_score_gap: float = Field(default=0.45, gt=0, le=1)
    fusion_minimum_modality_quality: float = Field(default=0.25, ge=0, le=1)
    # Radar Eligibility Gate. It changes only whether Radar may enter Fusion;
    # it does not alter Camera, TCN, checkpoints or the default 0.6/0.4 weights.
    fusion_radar_eligibility_enabled: bool = True
    fusion_radar_eligibility_history_seconds: float = Field(
        default=1.2, gt=0, le=10
    )
    fusion_radar_eligibility_minimum_track_samples: int = Field(
        default=2, ge=2, le=30
    )
    fusion_radar_eligibility_minimum_point_count: int = Field(
        default=3, ge=1, le=200
    )
    fusion_radar_eligibility_reference_point_count: int = Field(
        default=20, ge=1, le=500
    )
    fusion_radar_eligibility_minimum_track_stability: float = Field(
        default=0.60, ge=0, le=1
    )
    fusion_radar_eligibility_minimum_quality: float = Field(
        default=0.25, ge=0, le=1
    )
    fusion_radar_eligibility_maximum_velocity_jump_mps: float = Field(
        default=1.5, gt=0, le=20
    )
    fusion_radar_eligibility_height_tolerance_m: float = Field(
        default=0.50, gt=0, le=5
    )
    fusion_shadow_log_enabled: bool = True
    fusion_shadow_log_path: Path = BACKEND_DIR / "runtime" / "fusion_shadow.jsonl"
    fusion_shadow_log_max_mb: int = Field(default=20, ge=1, le=1024)
    fusion_shadow_log_backup_count: int = Field(default=5, ge=1, le=50)
    fusion_shadow_sampler_enabled: bool = True
    fusion_shadow_sample_interval_seconds: float = Field(default=0.5, gt=0, le=60)
    # Shadow-only temporal/association experiment. It cannot create alerts and
    # does not change Fixed Fusion, either modality model or score thresholds.
    fusion_temporal_window_seconds: float = Field(default=2.0, gt=0, le=10)
    fusion_temporal_confirmation_windows: int = Field(default=2, ge=1, le=20)
    # Camera-led TI tracking evidence augmentation. These are configurable
    # shadow evidence gates, not changes to BioSTGCN/TCN/Fusion thresholds.
    fusion_associated_window_seconds: float = Field(default=1.2, gt=0, le=10)
    fusion_associated_minimum_track_samples: int = Field(default=2, ge=2, le=30)
    fusion_associated_minimum_point_count: int = Field(default=3, ge=1, le=200)
    fusion_associated_minimum_track_stability: float = Field(
        default=0.60, ge=0, le=1
    )
    fusion_associated_weak_vertical_velocity_mps: float = -0.10
    fusion_associated_strong_vertical_velocity_mps: float = -0.35
    fusion_associated_weak_height_drop_m: float = -0.05
    fusion_associated_strong_height_drop_m: float = -0.20
    fusion_alignment_shadow_enabled: bool = True
    fusion_alignment_calibration_path: Path = (
        ENGINE_DIR / "calibrations" / "living_room_grid9_shadow_v0.json"
    )
    # Formal FusionFinding persistence stays disabled until independently validated.
    fusion_risk_events_enabled: bool = False
    fusion_offline_replay_preview_path: Path = (
        WORKSPACE_DIR
        / "摔倒预测多模态"
        / "evidence_replay"
        / "outputs"
        / "phase15_complete"
        / "evidence_preview.json"
    )

    fall_inference_project_dir: Path = (
        WORKSPACE_DIR / "摔倒预测模块" / "修改" / "MCF_LE2I_Final"
    )
    fall_inference_python: Path = (
        WORKSPACE_DIR
        / "摔倒预测模块"
        / "修改"
        / ".venv-rtmpose-full"
        / "Scripts"
        / "python.exe"
    )
    fall_inference_runtime_dir: Path = BACKEND_DIR / "runtime" / "fall_inference"
    fall_inference_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    fall_inference_max_upload_mb: int = Field(default=250, ge=1, le=2048)
    fall_inference_device: str = "cpu"
    fall_inference_live_protocol: int = Field(default=2, ge=2, le=4)
    fall_inference_live_quality: int = Field(default=2, ge=1, le=2)
    fall_inference_live_capture_seconds: int = Field(default=10, ge=6, le=60)
    fall_inference_live_address_expire_seconds: int = Field(
        default=120,
        ge=30,
        le=604800,
    )
    fall_inference_live_capture_timeout_seconds: int = Field(
        default=60,
        ge=15,
        le=300,
    )
    fall_live_monitor_enabled: bool = False
    fall_live_source_mode: Literal[
        "ezviz_opensdk", "ezviz_standard", "browser_capture"
    ] = (
        "ezviz_opensdk"
    )
    fall_live_ezviz_opensdk_root: Path = (
        WORKSPACE_DIR
        / "tmp"
        / "ezviz-sdk"
        / "extracted"
        / "EZPCOpenSDK_v5.13.1_build20250714"
    )
    fall_live_ezviz_opensdk_stream_type: int = Field(default=2, ge=1, le=2)
    # EZUIKit capturePicture is a snapshot API rather than a decoded 30 FPS
    # frame callback.  Keep the default at the measured sustainable rate so
    # the bounded queue does not continuously discard frames.
    fall_live_browser_capture_fps: float = Field(default=2.0, ge=0.5, le=30.0)
    fall_live_browser_max_frame_kb: int = Field(default=768, ge=32, le=4096)
    # RTMW3D WholeBody confidence is much lower than a 2D detector score at
    # room scale.  0.10 rejects collapsed poses while retaining body poses
    # accepted by the original RTMDet > 0.30 extraction path.
    fall_live_min_keypoint_confidence: float = Field(default=0.10, ge=0.0, le=1.0)
    fall_live_min_valid_pose_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    fall_live_detector_interval: int = Field(default=5, ge=1, le=15)
    fall_live_pose_batch_size: int = Field(default=5, ge=1, le=8)
    fall_live_required_source_frames: int = Field(default=45, ge=2, le=90)
    fall_live_max_source_gap_frames: int = Field(default=1, ge=1, le=30)
    fall_live_device_id: str = ""
    fall_live_channel_no: int = Field(default=1, ge=1)
    fall_live_stride_frames: int = Field(default=8, ge=1, le=90)
    fall_live_max_queue_frames: int = Field(default=12, ge=1, le=30)
    fall_live_stream_stall_timeout_seconds: float = Field(default=6.0, ge=2, le=60)
    fall_live_reconnect_seconds: float = Field(default=2.0, ge=0.5, le=300)
    # Optional raw Camera Evidence sidecar for Camera-Radar calibration.
    # Disabled by default and never consumed by inference/Fusion.
    fall_alignment_shadow_log_path: Path | None = None
    # OpenSDK camera-frame physiological sidecar. It is opt-in and shadow-only;
    # none of these settings alter Camera, Radar or Fusion decisions.
    rppg_enabled: bool = False
    rppg_sqi_threshold: float = Field(default=0.7, ge=0, le=1)
    rppg_min_valid_seconds: float = Field(default=60.0, ge=10, le=600)
    fall_live_event_cooldown_seconds: int = Field(default=30, ge=1, le=3600)
    fall_live_risk_events_enabled: bool = True
    fall_live_auto_create_session: bool = True

    @property
    def database_url(self) -> URL:
        return URL.create(
            drivername="mysql+pymysql",
            username=self.mysql_username,
            password=self.mysql_password.get_secret_value(),
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
            query={"charset": "utf8mb4"},
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
