from __future__ import annotations

import json
import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from radar_module.acquisition.ti_reader import (
    JsonlReplayAdapter,
    RadarSourceAdapter,
    TiOfficialOutputAdapter,
    TiRadarReader,
)
from radar_module.contracts import (
    RadarFrame,
    RadarPoint,
    RadarRiskResult,
    RadarTarget,
    Room,
    SourceMode,
)
from radar_module.inference.risk_prediction import RadarRiskPredictor
from radar_module.inference.research_live_v2 import (
    RadarResearchLivePredictorV2,
    ResearchLiveResultV2,
)
from radar_module.inference.tcn_live_v1 import (
    RadarTcnLivePredictorV1,
    TcnLiveResultV1,
)
from radar_module.inference.pointnet_live_v1 import (
    PointNetLiveResultV1,
    RadarPointNetLivePredictorV1,
)
from radar_module.inference.calibrated_tcn_live_v1 import (
    CalibratedTcnLivePredictorV1,
    CalibratedTcnLiveResultV1,
)
from radar_module.inference.fall_risk_assessment_v1 import (
    FallRiskAssessmentResultV1,
    RadarRiskAssessmentLiveV1,
)
from radar_module.inference.descent_live_v1 import (
    DescentLiveResultV1,
    RadarDescentLivePredictorV1,
)
from radar_module.dataset.point_iwr6843_adaptation_v1 import SEQUENCE_VERSION
from radar_module.model.radar_lstm import LoadedRadarModel, RadarLSTM
from radar_module.preprocess.feature_extraction import RadarFeatureExtractor
from radar_module.preprocess.temporal_features_v2 import FEATURE_VERSION_V2


RADAR_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class RadarApiSettings:
    replay_root: Path
    checkpoint_path: Path
    device_id: str = "iwr6843isk-01"
    room: Room = Room.LIVING_ROOM
    max_distance_m: float | None = 8.0
    risk_threshold: float = 0.7
    consecutive_windows: int = 3
    device: str = "cpu"
    ti_official_command: tuple[str, ...] | None = None
    ti_official_cwd: Path | None = None
    research_shadow_enabled: bool = False
    research_checkpoint_path: Path | None = None
    research_confirmation_windows: int = 3
    tcn_shadow_enabled: bool = False
    tcn_checkpoint_path: Path | None = None
    tcn_checkpoint_sha256: str | None = None
    tcn_confirmation_windows: int = 3
    pointnet_shadow_enabled: bool = False
    pointnet_checkpoint_path: Path | None = None
    pointnet_checkpoint_sha256: str | None = None
    pointnet_confirmation_windows: int = 3
    calibrated_tcn_shadow_enabled: bool = False
    calibrated_tcn_checkpoint_path: Path | None = None
    calibrated_tcn_checkpoint_sha256: str | None = None
    calibrated_tcn_calibration_path: Path | None = None
    calibrated_tcn_calibration_method: str = "real_gaussian"
    calibrated_tcn_confirmation_windows: int = 3
    calibrated_tcn_recovery_windows: int = 2
    calibrated_tcn_recovery_window_seconds: float = 1.5
    calibrated_tcn_persist_confirm_seconds: float = 0.0
    risk_assessment_shadow_enabled: bool = False
    risk_assessment_window_seconds: float = 60.0
    descent_shadow_enabled: bool = False
    descent_checkpoint_path: Path | None = None
    descent_checkpoint_sha256: str | None = None
    descent_confirmation_windows: int = 3
    descent_calibration_path: Path | None = None

    @classmethod
    def from_environment(cls) -> "RadarApiSettings":
        replay_root = Path(
            os.getenv("RADAR_REPLAY_ROOT", str(RADAR_DIR / "data" / "replay"))
        ).resolve()
        checkpoint_path = Path(
            os.getenv(
                "RADAR_CHECKPOINT_PATH",
                str(RADAR_DIR / "checkpoints" / "radar_lstm_test.pt"),
            )
        ).resolve()
        command_json = os.getenv("TI_OFFICIAL_OUTPUT_COMMAND_JSON", "").strip()
        command: tuple[str, ...] | None = None
        if command_json:
            parsed = json.loads(command_json)
            if not isinstance(parsed, list) or not all(
                isinstance(item, str) and item for item in parsed
            ):
                raise ValueError(
                    "TI_OFFICIAL_OUTPUT_COMMAND_JSON must be a JSON string array"
                )
            command = tuple(parsed)
        cwd_value = os.getenv("TI_OFFICIAL_OUTPUT_CWD", "").strip()
        research_checkpoint_value = os.getenv(
            "RADAR_RESEARCH_CHECKPOINT_PATH",
            str(
                RADAR_DIR
                / "checkpoints"
                / "radar_lstm_research_dguha_prefall_dense_pw32_v2.pt"
            ),
        ).strip()
        tcn_checkpoint_value = os.getenv(
            "RADAR_TCN_CHECKPOINT_PATH",
            str(
                RADAR_DIR
                / "checkpoints"
                / "experiments_v5"
                / "tcn_hard_negative"
                / "tcn_0p5_1p0_specificity_operating_point_v1.pt"
            ),
        ).strip()
        tcn_checkpoint_sha256 = os.getenv(
            "RADAR_TCN_CHECKPOINT_SHA256",
            "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef",
        ).strip()
        pointnet_checkpoint_value = os.getenv(
            "RADAR_POINTNET_CHECKPOINT_PATH",
            str(
                RADAR_DIR
                / "checkpoints"
                / "experiments_v9"
                / "pointnet_formal_conservative"
                / "pointnet_gru_radar_branch_v1.pt"
            ),
        ).strip()
        pointnet_checkpoint_sha256 = os.getenv(
            "RADAR_POINTNET_CHECKPOINT_SHA256", ""
        ).strip()
        calibrated_tcn_calibration_value = os.getenv(
            "RADAR_CALIBRATED_TCN_CALIBRATION_PATH", ""
        ).strip()
        return cls(
            replay_root=replay_root,
            checkpoint_path=checkpoint_path,
            device_id=os.getenv("RADAR_DEVICE_ID", "iwr6843isk-01").strip(),
            room=Room(os.getenv("RADAR_ROOM", Room.LIVING_ROOM.value)),
            max_distance_m=float(os.getenv("RADAR_MAX_DISTANCE_M", "8")),
            risk_threshold=float(os.getenv("RADAR_RISK_THRESHOLD", "0.7")),
            consecutive_windows=int(
                os.getenv("RADAR_CONSECUTIVE_WINDOWS", "3")
            ),
            device=os.getenv("RADAR_TORCH_DEVICE", "cpu"),
            ti_official_command=command,
            ti_official_cwd=Path(cwd_value).resolve() if cwd_value else None,
            research_shadow_enabled=_environment_flag(
                "RADAR_RESEARCH_SHADOW_ENABLED", False
            ),
            research_checkpoint_path=(
                Path(research_checkpoint_value).resolve()
                if research_checkpoint_value
                else None
            ),
            research_confirmation_windows=int(
                os.getenv("RADAR_RESEARCH_CONFIRMATION_WINDOWS", "3")
            ),
            tcn_shadow_enabled=_environment_flag(
                "RADAR_TCN_SHADOW_ENABLED", False
            ),
            tcn_checkpoint_path=(
                Path(tcn_checkpoint_value).resolve()
                if tcn_checkpoint_value
                else None
            ),
            tcn_checkpoint_sha256=(
                tcn_checkpoint_sha256 or None
            ),
            tcn_confirmation_windows=int(
                os.getenv("RADAR_TCN_CONFIRMATION_WINDOWS", "3")
            ),
            pointnet_shadow_enabled=_environment_flag(
                "RADAR_POINTNET_SHADOW_ENABLED", False
            ),
            pointnet_checkpoint_path=(
                Path(pointnet_checkpoint_value).resolve()
                if pointnet_checkpoint_value
                else None
            ),
            pointnet_checkpoint_sha256=(pointnet_checkpoint_sha256 or None),
            pointnet_confirmation_windows=int(
                os.getenv("RADAR_POINTNET_CONFIRMATION_WINDOWS", "3")
            ),
            calibrated_tcn_shadow_enabled=_environment_flag(
                "RADAR_CALIBRATED_TCN_SHADOW_ENABLED", False
            ),
            calibrated_tcn_checkpoint_path=(
                Path(tcn_checkpoint_value).resolve()
                if tcn_checkpoint_value
                else None
            ),
            calibrated_tcn_checkpoint_sha256=(
                tcn_checkpoint_sha256 or None
            ),
            calibrated_tcn_calibration_path=(
                Path(calibrated_tcn_calibration_value).resolve()
                if calibrated_tcn_calibration_value
                else None
            ),
            calibrated_tcn_calibration_method=os.getenv(
                "RADAR_CALIBRATED_TCN_CALIBRATION_METHOD", "real_gaussian"
            ).strip(),
            calibrated_tcn_confirmation_windows=int(
                os.getenv("RADAR_CALIBRATED_TCN_CONFIRMATION_WINDOWS", "3")
            ),
            calibrated_tcn_recovery_windows=int(
                os.getenv("RADAR_CALIBRATED_TCN_RECOVERY_WINDOWS", "2")
            ),
            calibrated_tcn_recovery_window_seconds=float(
                os.getenv("RADAR_CALIBRATED_TCN_RECOVERY_WINDOW_SECONDS", "1.5")
            ),
            calibrated_tcn_persist_confirm_seconds=float(
                os.getenv("RADAR_CALIBRATED_TCN_PERSIST_CONFIRM_SECONDS", "0.0")
            ),
            risk_assessment_shadow_enabled=_environment_flag(
                "RADAR_RISK_ASSESSMENT_SHADOW_ENABLED", False
            ),
            risk_assessment_window_seconds=float(
                os.getenv("RADAR_RISK_ASSESSMENT_WINDOW_SECONDS", "60")
            ),
            descent_shadow_enabled=_environment_flag(
                "RADAR_DESCENT_SHADOW_ENABLED", False
            ),
            descent_checkpoint_path=(
                Path(
                    os.getenv(
                        "RADAR_DESCENT_CHECKPOINT_PATH",
                        str(
                            RADAR_DIR
                            / "checkpoints"
                            / "experiments_v10"
                            / "descent_detection_tcn_v1.pt"
                        ),
                    ).strip()
                ).resolve()
            ),
            descent_checkpoint_sha256=os.getenv(
                "RADAR_DESCENT_CHECKPOINT_SHA256",
                "82ba9c7dbb4862609ac36e02dd183df87fb8c966957c2c8b3f1e9cbb3df22ca4",
            ).strip()
            or None,
            descent_confirmation_windows=int(
                os.getenv("RADAR_DESCENT_CONFIRMATION_WINDOWS", "3")
            ),
            descent_calibration_path=(
                Path(
                    os.getenv(
                        "RADAR_DESCENT_CALIBRATION_PATH",
                        str(
                            RADAR_DIR
                            / "reports"
                            / "domain_calibration_v1_full"
                            / "calibrated_normalization_descent_iwr6843_fall102.json"
                        ),
                    ).strip()
                ).resolve()
            ),
        )


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str = Field(min_length=1, max_length=255)
    speed: float = Field(default=1.0, gt=0, le=100)
    loop: bool = False


class RadarRuntime:
    def __init__(
        self,
        *,
        settings: RadarApiSettings,
        loaded_model: LoadedRadarModel,
        research_predictor: RadarResearchLivePredictorV2 | None = None,
        tcn_predictor: RadarTcnLivePredictorV1 | None = None,
        pointnet_predictor: RadarPointNetLivePredictorV1 | None = None,
        calibrated_tcn_predictor: CalibratedTcnLivePredictorV1 | None = None,
        risk_assessment_predictor: RadarRiskAssessmentLiveV1 | None = None,
        descent_predictor: RadarDescentLivePredictorV1 | None = None,
    ) -> None:
        self.settings = settings
        self.loaded_model = loaded_model
        self.research_predictor = research_predictor
        self.tcn_predictor = tcn_predictor
        self.pointnet_predictor = pointnet_predictor
        self.calibrated_tcn_predictor = calibrated_tcn_predictor
        self.risk_assessment_predictor = risk_assessment_predictor
        self.descent_predictor = descent_predictor
        self.extractor = RadarFeatureExtractor()
        self._predictor = self._new_predictor()
        self._reader: TiRadarReader | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._latest: RadarRiskResult | None = None
        self._last_error: str | None = None
        self._latest_research: ResearchLiveResultV2 | None = None
        self._research_last_error: str | None = None
        self._latest_tcn: TcnLiveResultV1 | None = None
        self._tcn_last_error: str | None = None
        self._latest_pointnet: PointNetLiveResultV1 | None = None
        self._pointnet_last_error: str | None = None
        self._latest_calibrated_tcn: CalibratedTcnLiveResultV1 | None = None
        self._calibrated_tcn_last_error: str | None = None
        self._latest_risk_assessment: dict[str, object] | None = None
        self._risk_assessment_last_error: str | None = None
        self._latest_descent: DescentLiveResultV1 | None = None
        self._descent_last_error: str | None = None
        self._frame_timestamps: deque[float] = deque(maxlen=512)
        self._latest_point_count: int | None = None
        self._latest_alignment_evidence: list[dict[str, object]] = []
        self._radar_config_name: str | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    @property
    def source_mode(self) -> SourceMode | None:
        with self._lock:
            return self._reader.source_mode if self._reader is not None else None

    @property
    def latest(self) -> RadarRiskResult | None:
        with self._lock:
            return self._latest

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def latest_research(self) -> ResearchLiveResultV2 | None:
        with self._lock:
            return self._latest_research

    @property
    def latest_tcn(self) -> TcnLiveResultV1 | None:
        with self._lock:
            return self._latest_tcn

    @property
    def latest_pointnet(self) -> PointNetLiveResultV1 | None:
        with self._lock:
            return self._latest_pointnet

    @property
    def latest_calibrated_tcn(self) -> CalibratedTcnLiveResultV1 | None:
        with self._lock:
            return self._latest_calibrated_tcn

    @property
    def calibrated_tcn_last_error(self) -> str | None:
        with self._lock:
            return self._calibrated_tcn_last_error

    @property
    def latest_risk_assessment(self) -> dict[str, object] | None:
        with self._lock:
            return self._latest_risk_assessment

    @property
    def risk_assessment_last_error(self) -> str | None:
        with self._lock:
            return self._risk_assessment_last_error

    @property
    def latest_descent(self) -> DescentLiveResultV1 | None:
        with self._lock:
            return self._latest_descent

    @property
    def latest_alignment_evidence(self) -> list[dict[str, object]]:
        """Return a copy of the shadow-only target evidence."""

        with self._lock:
            return [dict(item) for item in self._latest_alignment_evidence]

    @property
    def descent_last_error(self) -> str | None:
        with self._lock:
            return self._descent_last_error

    def start_source(self, source_adapter: RadarSourceAdapter) -> None:
        self.stop_source()
        reader = TiRadarReader(
            source_adapter=source_adapter,
            device_id=self.settings.device_id,
            room=self.settings.room,
            max_distance_m=self.settings.max_distance_m,
        )
        reader.start()
        with self._lock:
            self._reader = reader
            self._predictor = self._new_predictor()
            self._latest = None
            self._last_error = None
            self._latest_research = None
            self._research_last_error = None
            self._latest_tcn = None
            self._tcn_last_error = None
            self._latest_pointnet = None
            self._pointnet_last_error = None
            self._latest_calibrated_tcn = None
            self._calibrated_tcn_last_error = None
            self._latest_risk_assessment = None
            self._risk_assessment_last_error = None
            self._latest_descent = None
            self._descent_last_error = None
            self._frame_timestamps.clear()
            self._latest_point_count = None
            self._latest_alignment_evidence = []
            self._radar_config_name = None
            if self.research_predictor is not None:
                self.research_predictor.reset()
            if self.tcn_predictor is not None:
                self.tcn_predictor.reset()
            if self.pointnet_predictor is not None:
                self.pointnet_predictor.reset()
            if self.calibrated_tcn_predictor is not None:
                self.calibrated_tcn_predictor.reset()
            if self.risk_assessment_predictor is not None:
                self.risk_assessment_predictor.reset()
            if self.descent_predictor is not None:
                self.descent_predictor.reset()
            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._run,
                name=f"radar-{reader.source_mode.value.lower()}-worker",
                daemon=True,
            )
            self._worker.start()

    def start_replay(
        self,
        file_path: Path,
        *,
        speed: float,
        loop: bool,
    ) -> None:
        self.start_source(
            JsonlReplayAdapter(file_path, speed=speed, loop=loop)
        )

    def start_real(self) -> None:
        command = self.settings.ti_official_command
        if command is None:
            raise RuntimeError(
                "TI official output command is not configured; set "
                "TI_OFFICIAL_OUTPUT_COMMAND_JSON to the official/bridge command"
            )
        self.start_source(
            TiOfficialOutputAdapter(
                command=command,
                cwd=self.settings.ti_official_cwd,
            )
        )

    def stop_source(self) -> None:
        with self._lock:
            self._stop_event.set()
            reader = self._reader
            worker = self._worker
        if reader is not None:
            reader.stop()
        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=3)
        with self._lock:
            self._reader = None
            self._worker = None
            self._latest = None
            self._latest_research = None
            self._latest_tcn = None
            self._latest_pointnet = None
            self._latest_calibrated_tcn = None
            self._latest_risk_assessment = None
            self._latest_descent = None
            self._frame_timestamps.clear()
            self._latest_point_count = None
            self._latest_alignment_evidence = []
            self._radar_config_name = None
            if self.research_predictor is not None:
                self.research_predictor.reset()
            if self.tcn_predictor is not None:
                self.tcn_predictor.reset()
            if self.pointnet_predictor is not None:
                self.pointnet_predictor.reset()
            if self.calibrated_tcn_predictor is not None:
                self.calibrated_tcn_predictor.reset()
            if self.risk_assessment_predictor is not None:
                self.risk_assessment_predictor.reset()
            if self.descent_predictor is not None:
                self.descent_predictor.reset()

    def health_payload(self) -> dict[str, object]:
        source_mode = self.source_mode
        active_model_mode = (
            "RESEARCH_WEAK_SUPERVISION"
            if self.pointnet_predictor is not None
            else self.tcn_predictor.model_mode
            if self.tcn_predictor is not None
            else self.loaded_model.model_mode.value
        )
        active_feature_version = (
            SEQUENCE_VERSION
            if self.pointnet_predictor is not None
            else FEATURE_VERSION_V2
            if self.tcn_predictor is not None
            else self.loaded_model.feature_version
        )
        with self._lock:
            frame_rate_hz = self._observed_frame_rate_hz()
            point_count = self._latest_point_count
            radar_config_name = self._radar_config_name
        return {
            "status": "ok" if self.last_error is None else "degraded",
            "radar_connected": self.is_running,
            "model_loaded": True,
            "source_mode": source_mode.value if source_mode else None,
            "model_mode": active_model_mode,
            "feature_version": active_feature_version,
            "frame_rate_hz": frame_rate_hz,
            "point_count": point_count,
            "radar_config_name": radar_config_name,
            "last_error": self.last_error,
            "research_shadow_enabled": self.research_predictor is not None,
            "research_shadow_ready": self.latest_research is not None,
            "research_shadow_error": self._research_last_error,
            "tcn_shadow_enabled": self.tcn_predictor is not None,
            "tcn_shadow_ready": self.latest_tcn is not None,
            "tcn_shadow_error": self._tcn_last_error,
            "tcn_model_version": (
                self.tcn_predictor.model_version
                if self.tcn_predictor is not None
                else None
            ),
            "tcn_checkpoint_sha256": (
                self.tcn_predictor.checkpoint_sha256
                if self.tcn_predictor is not None
                else None
            ),
            "tcn_threshold": (
                self.tcn_predictor.threshold
                if self.tcn_predictor is not None
                else None
            ),
            "calibrated_tcn_shadow_enabled": (
                self.calibrated_tcn_predictor is not None
            ),
            "calibrated_tcn_shadow_ready": (
                self.latest_calibrated_tcn is not None
            ),
            "calibrated_tcn_shadow_error": self.calibrated_tcn_last_error,
            "calibrated_tcn_checkpoint_sha256": (
                self.calibrated_tcn_predictor.checkpoint_sha256
                if self.calibrated_tcn_predictor is not None
                else None
            ),
            "calibrated_tcn_threshold": (
                self.calibrated_tcn_predictor.threshold
                if self.calibrated_tcn_predictor is not None
                else None
            ),
            "risk_assessment_shadow_enabled": (
                self.risk_assessment_predictor is not None
            ),
            "risk_assessment_shadow_ready": (
                self.latest_risk_assessment is not None
            ),
            "risk_assessment_shadow_error": self.risk_assessment_last_error,
            "descent_shadow_enabled": self.descent_predictor is not None,
            "descent_shadow_ready": self.latest_descent is not None,
            "descent_shadow_error": self.descent_last_error,
            "descent_model_version": (
                self.descent_predictor.model_version
                if self.descent_predictor is not None
                else None
            ),
            "descent_checkpoint_sha256": (
                self.descent_predictor.checkpoint_sha256
                if self.descent_predictor is not None
                else None
            ),
            "descent_threshold": (
                self.descent_predictor.threshold
                if self.descent_predictor is not None
                else None
            ),
            "pointnet_shadow_enabled": self.pointnet_predictor is not None,
            "pointnet_shadow_ready": self.latest_pointnet is not None,
            "pointnet_shadow_error": self._pointnet_last_error,
            "pointnet_model_version": (
                self.pointnet_predictor.model_version
                if self.pointnet_predictor is not None
                else None
            ),
            "pointnet_checkpoint_sha256": (
                self.pointnet_predictor.checkpoint_sha256
                if self.pointnet_predictor is not None
                else None
            ),
            "pointnet_threshold": (
                self.pointnet_predictor.threshold
                if self.pointnet_predictor is not None
                else None
            ),
        }

    def _run(self) -> None:
        with self._lock:
            reader = self._reader
        if reader is None:
            return
        try:
            while not self._stop_event.is_set():
                frame = reader.read()
                if frame is None:
                    if isinstance(reader.source_adapter, JsonlReplayAdapter):
                        if reader.source_adapter.finished:
                            break
                    continue
                with self._lock:
                    epoch = frame.timestamp.timestamp()
                    if (
                        self._frame_timestamps
                        and epoch < self._frame_timestamps[-1]
                    ):
                        self._frame_timestamps.clear()
                    self._frame_timestamps.append(epoch)
                    while (
                        len(self._frame_timestamps) > 2
                        and epoch - self._frame_timestamps[0] > 2.0
                    ):
                        self._frame_timestamps.popleft()
                    self._latest_point_count = len(frame.points)
                    self._radar_config_name = frame.radar_config_name
                if self.research_predictor is not None:
                    try:
                        research_result = self.research_predictor.consume(frame)
                        if research_result is not None:
                            with self._lock:
                                self._latest_research = research_result
                                self._research_last_error = None
                    except (RuntimeError, TypeError, ValueError) as exc:
                        with self._lock:
                            self._research_last_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        self.research_predictor.reset()
                if self.tcn_predictor is not None:
                    try:
                        tcn_result = self.tcn_predictor.consume(frame)
                        if tcn_result is not None:
                            with self._lock:
                                self._latest_tcn = tcn_result
                                self._tcn_last_error = None
                    except (RuntimeError, TypeError, ValueError) as exc:
                        with self._lock:
                            self._tcn_last_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        self.tcn_predictor.reset()
                if self.pointnet_predictor is not None:
                    try:
                        pointnet_result = self.pointnet_predictor.consume(frame)
                        if pointnet_result is not None:
                            with self._lock:
                                self._latest_pointnet = pointnet_result
                                self._pointnet_last_error = None
                    except (RuntimeError, TypeError, ValueError) as exc:
                        with self._lock:
                            self._pointnet_last_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        self.pointnet_predictor.reset()
                if self.calibrated_tcn_predictor is not None:
                    try:
                        calibrated_result = (
                            self.calibrated_tcn_predictor.consume(frame)
                        )
                        if calibrated_result is not None:
                            with self._lock:
                                self._latest_calibrated_tcn = calibrated_result
                                self._calibrated_tcn_last_error = None
                    except (RuntimeError, TypeError, ValueError) as exc:
                        with self._lock:
                            self._calibrated_tcn_last_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        self.calibrated_tcn_predictor.reset()
                if self.risk_assessment_predictor is not None:
                    try:
                        risk_result = self.risk_assessment_predictor.consume(frame)
                        if risk_result is not None:
                            with self._lock:
                                self._latest_risk_assessment = risk_result
                                self._risk_assessment_last_error = None
                    except (RuntimeError, TypeError, ValueError) as exc:
                        with self._lock:
                            self._risk_assessment_last_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        self.risk_assessment_predictor.reset()
                if self.descent_predictor is not None:
                    try:
                        descent_result = self.descent_predictor.consume(frame)
                        if descent_result is not None:
                            with self._lock:
                                self._latest_descent = descent_result
                                self._descent_last_error = None
                    except (RuntimeError, TypeError, ValueError) as exc:
                        with self._lock:
                            self._descent_last_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                        self.descent_predictor.reset()
                feature = self.extractor.extract(frame)
                result = self._predictor.consume(feature)
                with self._lock:
                    self._latest = result
                    self._latest_alignment_evidence = (
                        _build_alignment_radar_evidence(
                            frame,
                            tcn=self._latest_tcn,
                        )
                    )
        except BaseException as exc:
            if not self._stop_event.is_set():
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            reader.stop()
            with self._lock:
                if self._reader is reader:
                    self._reader = None
                    self._worker = None
                    # 回放结束后不把最后一帧继续伪装为实时数据。
                    self._latest = None
                    self._latest_research = None
                    self._latest_tcn = None
                    self._latest_pointnet = None
                    self._latest_calibrated_tcn = None
                    self._latest_risk_assessment = None
                    self._latest_descent = None
                    self._frame_timestamps.clear()
                    self._latest_point_count = None
                    self._latest_alignment_evidence = []
                    self._radar_config_name = None

    def _observed_frame_rate_hz(self) -> float | None:
        if len(self._frame_timestamps) < 2:
            return None
        elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if elapsed <= 0:
            return None
        return (len(self._frame_timestamps) - 1) / elapsed

    def _new_predictor(self) -> RadarRiskPredictor:
        return RadarRiskPredictor(
            self.loaded_model,
            risk_threshold=self.settings.risk_threshold,
            consecutive_windows=self.settings.consecutive_windows,
            device=self.settings.device,
        )


def _build_alignment_radar_evidence(
    frame: RadarFrame,
    *,
    tcn: TcnLiveResultV1 | None,
) -> list[dict[str, object]]:
    """Build additive per-track evidence without changing radar inference."""

    score = (
        float(tcn.pre_fall_score)
        if tcn is not None and tcn.score_valid
        else None
    )
    quality_by_state = {
        "GOOD": 1.0,
        "DEGRADED": 0.5,
        "INSUFFICIENT_DATA": 0.0,
    }
    radar_quality = (
        quality_by_state.get(str(tcn.data_quality), 0.0)
        if tcn is not None
        else 0.0
    )
    targets = frame.targets or _targets_from_points(frame)
    point_counts: dict[int, int] = {}
    points_by_track: dict[int, list[RadarPoint]] = {}
    for point in frame.points:
        if point.track_id is not None:
            point_counts[point.track_id] = point_counts.get(point.track_id, 0) + 1
            points_by_track.setdefault(point.track_id, []).append(point)

    common: dict[str, object] = {
        "frame_number": frame.frame_number,
        "source_timestamp": frame.source_timestamp or frame.timestamp.isoformat(),
        "radar_score": score,
        "radar_quality": float(radar_quality),
        "radar_state": tcn.risk_state if tcn is not None else "UNKNOWN",
        "radar_score_timestamp": tcn.timestamp if tcn is not None else None,
        "radar_config_name": frame.radar_config_name,
        "shadow_only": True,
    }
    if not targets:
        return [
            {
                **common,
                "track_id": None,
                "x": None,
                "y": None,
                "z": None,
                "vx": None,
                "vy": None,
                "vz": None,
                "point_count": 0,
                "point_cloud_spread_m": None,
            }
        ]
    return [
        {
            **common,
            "track_id": target.track_id,
            "x": target.x,
            "y": target.y,
            "z": target.z,
            "vx": target.velocity_x,
            "vy": target.velocity_y,
            "vz": target.velocity_z,
            "point_count": point_counts.get(target.track_id, 0),
            "point_cloud_spread_m": _point_cloud_spread_m(
                points_by_track.get(target.track_id, [])
            ),
            "target_confidence": target.confidence,
        }
        for target in targets
    ]


def _point_cloud_spread_m(points: list[RadarPoint]) -> float | None:
    """RMS 3-D radius for one TI track; observational sidecar only."""

    if not points:
        return None
    center_x = sum(point.x for point in points) / len(points)
    center_y = sum(point.y for point in points) / len(points)
    center_z = sum(point.z for point in points) / len(points)
    mean_squared_radius = sum(
        (point.x - center_x) ** 2
        + (point.y - center_y) ** 2
        + (point.z - center_z) ** 2
        for point in points
    ) / len(points)
    return float(math.sqrt(mean_squared_radius))


def _targets_from_points(frame: RadarFrame) -> tuple[RadarTarget, ...]:
    grouped: dict[int, list[RadarPoint]] = {}
    for point in frame.points:
        if point.track_id is not None:
            grouped.setdefault(point.track_id, []).append(point)
    targets: list[RadarTarget] = []
    for track_id, points in sorted(grouped.items()):
        targets.append(
            RadarTarget(
                track_id=track_id,
                x=float(sum(point.x for point in points) / len(points)),
                y=float(sum(point.y for point in points) / len(points)),
                z=float(sum(point.z for point in points) / len(points)),
            )
        )
    return tuple(targets)


def create_app(settings: RadarApiSettings | None = None) -> FastAPI:
    resolved_settings = settings or RadarApiSettings.from_environment()
    resolved_settings.replay_root.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint(resolved_settings.checkpoint_path)
    loaded_model = RadarLSTM.load_checkpoint(
        resolved_settings.checkpoint_path,
        expected_feature_version=RadarFeatureExtractor.feature_version,
        expected_feature_names=RadarFeatureExtractor.feature_names,
        expected_window_size=30,
        expected_input_size=8,
        device=resolved_settings.device,
    )
    research_predictor = None
    tcn_predictor = None
    pointnet_predictor = None
    calibrated_tcn_predictor = None
    if (
        resolved_settings.research_shadow_enabled
        and (
            resolved_settings.tcn_shadow_enabled
            or resolved_settings.pointnet_shadow_enabled
        )
    ):
        raise ValueError(
            "legacy research shadow cannot run with TCN/PointNet shadow inference"
        )
    if resolved_settings.research_shadow_enabled:
        if resolved_settings.research_checkpoint_path is None:
            raise ValueError(
                "RADAR_RESEARCH_CHECKPOINT_PATH is required when research shadow is enabled"
            )
        research_predictor = RadarResearchLivePredictorV2(
            resolved_settings.research_checkpoint_path,
            confirmation_windows=(
                resolved_settings.research_confirmation_windows
            ),
            device=resolved_settings.device,
        )
    if resolved_settings.tcn_shadow_enabled:
        if resolved_settings.tcn_checkpoint_path is None:
            raise ValueError(
                "RADAR_TCN_CHECKPOINT_PATH is required when TCN shadow is enabled"
            )
        if resolved_settings.tcn_checkpoint_sha256 is None:
            raise ValueError(
                "RADAR_TCN_CHECKPOINT_SHA256 is required when TCN shadow is enabled"
            )
        tcn_predictor = RadarTcnLivePredictorV1(
            resolved_settings.tcn_checkpoint_path,
            expected_checkpoint_sha256=(
                resolved_settings.tcn_checkpoint_sha256
            ),
            confirmation_windows=resolved_settings.tcn_confirmation_windows,
            device=resolved_settings.device,
        )
    if resolved_settings.pointnet_shadow_enabled:
        if resolved_settings.pointnet_checkpoint_path is None:
            raise ValueError(
                "RADAR_POINTNET_CHECKPOINT_PATH is required when PointNet shadow is enabled"
            )
        if resolved_settings.pointnet_checkpoint_sha256 is None:
            raise ValueError(
                "RADAR_POINTNET_CHECKPOINT_SHA256 is required when PointNet shadow is enabled"
            )
        pointnet_predictor = RadarPointNetLivePredictorV1(
            resolved_settings.pointnet_checkpoint_path,
            expected_checkpoint_sha256=resolved_settings.pointnet_checkpoint_sha256,
            confirmation_windows=resolved_settings.pointnet_confirmation_windows,
            device=resolved_settings.device,
        )
    if resolved_settings.calibrated_tcn_shadow_enabled:
        if resolved_settings.calibrated_tcn_checkpoint_path is None:
            raise ValueError(
                "calibrated TCN shadow requires a TCN checkpoint path"
            )
        if resolved_settings.calibrated_tcn_checkpoint_sha256 is None:
            raise ValueError(
                "calibrated TCN shadow requires a TCN checkpoint SHA256"
            )
        if resolved_settings.calibrated_tcn_calibration_path is None:
            raise ValueError(
                "calibrated TCN shadow requires RADAR_CALIBRATED_TCN_CALIBRATION_PATH"
            )
        calibrated_tcn_predictor = CalibratedTcnLivePredictorV1(
            resolved_settings.calibrated_tcn_checkpoint_path,
            expected_checkpoint_sha256=(
                resolved_settings.calibrated_tcn_checkpoint_sha256
            ),
            calibration_path=(
                resolved_settings.calibrated_tcn_calibration_path
            ),
            calibration_method=(
                resolved_settings.calibrated_tcn_calibration_method
            ),
            confirmation_windows=(
                resolved_settings.calibrated_tcn_confirmation_windows
            ),
            recovery_windows=(
                resolved_settings.calibrated_tcn_recovery_windows
            ),
            recovery_window_seconds=(
                resolved_settings.calibrated_tcn_recovery_window_seconds
            ),
            persist_confirm_seconds=(
                resolved_settings.calibrated_tcn_persist_confirm_seconds
            ),
            emit_formal_alert=False,
            device=resolved_settings.device,
        )
    risk_assessment_predictor = None
    if resolved_settings.risk_assessment_shadow_enabled:
        risk_assessment_predictor = RadarRiskAssessmentLiveV1(
            assessment_window_seconds=(
                resolved_settings.risk_assessment_window_seconds
            ),
        )
    descent_predictor = None
    if resolved_settings.descent_shadow_enabled:
        if resolved_settings.descent_checkpoint_path is None:
            raise ValueError(
                "descent shadow requires RADAR_DESCENT_CHECKPOINT_PATH"
            )
        if resolved_settings.descent_checkpoint_sha256 is None:
            raise ValueError(
                "descent shadow requires RADAR_DESCENT_CHECKPOINT_SHA256"
            )
        descent_predictor = RadarDescentLivePredictorV1(
            resolved_settings.descent_checkpoint_path,
            expected_checkpoint_sha256=(
                resolved_settings.descent_checkpoint_sha256
            ),
            confirmation_windows=(
                resolved_settings.descent_confirmation_windows
            ),
            calibration_path=(
                resolved_settings.descent_calibration_path
                if resolved_settings.descent_calibration_path.is_file()
                else None
            ),
            device=resolved_settings.device,
        )
    runtime = RadarRuntime(
        settings=resolved_settings,
        loaded_model=loaded_model,
        research_predictor=research_predictor,
        tcn_predictor=tcn_predictor,
        pointnet_predictor=pointnet_predictor,
        calibrated_tcn_predictor=calibrated_tcn_predictor,
        risk_assessment_predictor=risk_assessment_predictor,
        descent_predictor=descent_predictor,
    )

    app = FastAPI(
        title="IWR6843ISK Radar Risk Inference Framework",
        version="0.1.0",
        description=(
            "比赛MVP技术链路。TEST_CHECKPOINT输出仅用于DEMO，"
            "不代表真实跌倒预测能力。"
        ),
    )
    app.state.radar_runtime = runtime

    @app.get("/health")
    def health() -> dict[str, object]:
        return runtime.health_payload()

    @app.get("/api/radar/latest")
    def latest() -> dict[str, object]:
        def attach_alignment(payload: dict[str, object]) -> dict[str, object]:
            payload["alignment_evidence"] = runtime.latest_alignment_evidence
            payload["alignment_shadow_only"] = True
            return payload

        risk_assessment = runtime.latest_risk_assessment
        descent = runtime.latest_descent
        if runtime.calibrated_tcn_predictor is not None:
            calibrated = runtime.latest_calibrated_tcn
            if calibrated is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Calibrated TCN shadow inference has no current result",
                )
            tcn = runtime.latest_tcn
            payload = {
                "calibrated_tcn_prediction": calibrated.to_dict(),
                "tcn_baseline": tcn.to_dict() if tcn is not None else None,
            }
            if risk_assessment is not None:
                payload["fall_risk_assessment"] = risk_assessment
            if descent is not None:
                payload["descent_prediction"] = descent.to_dict()
            return attach_alignment(payload)
        if runtime.pointnet_predictor is not None:
            pointnet = runtime.latest_pointnet
            if pointnet is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="PointNet shadow inference has no current result",
                )
            tcn = runtime.latest_tcn
            payload = {
                "pointnet_prediction": pointnet.to_dict(),
                "tcn_baseline": tcn.to_dict() if tcn is not None else None,
            }
            if risk_assessment is not None:
                payload["fall_risk_assessment"] = risk_assessment
            return attach_alignment(payload)
        if runtime.tcn_predictor is not None:
            tcn = runtime.latest_tcn
            if tcn is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="TCN shadow inference has no current result",
                )
            payload = {"tcn_prediction": tcn.to_dict()}
            if risk_assessment is not None:
                payload["fall_risk_assessment"] = risk_assessment
            return attach_alignment(payload)

        result = runtime.latest
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="radar source has no current result",
            )
        payload = result.to_dict()
        research = runtime.latest_research
        if research is not None:
            combined_risk = max(
                research.pre_fall_score,
                research.fall_risk_score,
            )
            payload["risk_score"] = combined_risk
            if payload["human_state"] != "NO_PERSON":
                payload["human_state"] = (
                    "FALL_RISK"
                    if research.fall_risk_score >= 0.30
                    or research.prediction_state == "IMMINENT"
                    else "NORMAL"
                )
            payload["event_triggered"] = (
                research.action_risk_event_triggered
                or research.prediction_state == "IMMINENT"
            )
        payload["research"] = (
            research.to_dict() if research is not None else None
        )
        return attach_alignment(payload)

    @app.post("/api/radar/replay", status_code=status.HTTP_202_ACCEPTED)
    def replay(request: ReplayRequest) -> dict[str, object]:
        replay_path = _resolve_replay_file(
            resolved_settings.replay_root,
            request.file_name,
        )
        try:
            runtime.start_replay(
                replay_path,
                speed=request.speed,
                loop=request.loop,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "started",
            "source_mode": SourceMode.REPLAY.value,
            "file_name": request.file_name,
            "speed": request.speed,
            "loop": request.loop,
        }

    @app.post("/api/radar/real", status_code=status.HTTP_202_ACCEPTED)
    def real() -> dict[str, object]:
        try:
            runtime.start_real()
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "started",
            "source_mode": SourceMode.REAL.value,
        }

    @app.post("/api/radar/stop")
    def stop() -> dict[str, object]:
        runtime.stop_source()
        return {"status": "stopped"}

    return app


def _ensure_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        return
    RadarLSTM.create_test_checkpoint(
        checkpoint_path,
        feature_version=RadarFeatureExtractor.feature_version,
        feature_names=RadarFeatureExtractor.feature_names,
        window_size=30,
        input_size=8,
        seed=20260724,
    )


def _resolve_replay_file(replay_root: Path, file_name: str) -> Path:
    if Path(file_name).is_absolute():
        raise HTTPException(status_code=400, detail="absolute paths are not allowed")
    root = replay_root.resolve()
    candidate = (root / file_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="replay path must stay inside RADAR_REPLAY_ROOT",
        ) from exc
    if candidate.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="replay file must be JSONL")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="replay file not found")
    return candidate


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


app = create_app()
