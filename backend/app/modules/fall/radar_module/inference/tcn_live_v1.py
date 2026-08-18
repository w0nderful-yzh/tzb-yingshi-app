from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from radar_module.contracts import RadarFrame
from radar_module.model.research_training_v2 import RESEARCH_MODEL_MODE
from radar_module.model.temporal_models_v3 import (
    EXPERIMENT_MODEL_VERSION,
    TemporalBinaryModel,
)
from radar_module.preprocess.safe_normalize import (
    safe_normalize,
    zero_variance_feature_mask,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
    TemporalFeatureWindowV2,
    WINDOW_SIZE_V2,
)


TCN_LIVE_SCHEMA_VERSION = "radar_tcn_live_v1"
TCN_ARCHITECTURE = "causal_tcn"
TCN_SHADOW_DISCLAIMER = (
    "毫米波TCN短时跌倒风险预测处于实时影子验证阶段，不触发正式告警"
)


@dataclass(frozen=True, slots=True)
class TcnLiveResultV1:
    schema_version: str
    timestamp: str
    emitted_at: str
    device_id: str
    room: str
    source_mode: str
    risk_state: str
    pre_fall_score: float
    score_valid: bool
    consecutive_high_windows: int
    event_triggered: bool
    event_id: str | None
    unknown_reason: str | None
    data_quality: str
    missing_frame_ratio: float
    longest_unresolved_gap_seconds: float
    centroid_z: float | None
    vertical_velocity: float | None
    height_delta_0_6s: float | None
    feature_point_count: float | None
    model_version: str
    model_mode: str
    architecture: str
    checkpoint_sha256: str
    feature_version: str
    threshold: float
    threshold_policy: str
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str
    shadow_only: bool = True
    alert_suppressed: bool = True
    disclaimer: str = TCN_SHADOW_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["prediction_horizon_seconds"] = list(
            self.prediction_horizon_seconds
        )
        return payload


class RadarTcnLivePredictorV1:
    """Run the frozen causal TCN on timestamped decoded radar point clouds."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        confirmation_windows: int = 3,
        device: str | torch.device = "cpu",
    ) -> None:
        if confirmation_windows < 2:
            raise ValueError("confirmation_windows must be at least two")
        expected_sha256 = expected_checkpoint_sha256.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("expected_checkpoint_sha256 must contain 64 hex digits")

        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"TCN checkpoint does not exist: {self.checkpoint_path}"
            )
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        if self.checkpoint_sha256 != expected_sha256:
            raise ValueError(
                "TCN checkpoint SHA256 mismatch: "
                f"expected {expected_sha256}, got {self.checkpoint_sha256}"
            )

        self.device = torch.device(device)
        checkpoint = _load_and_validate_checkpoint(
            self.checkpoint_path, self.device
        )
        self.model = TemporalBinaryModel(
            architecture=TCN_ARCHITECTURE,
            input_size=len(FEATURE_NAMES_V2),
            hidden_size=int(checkpoint["hidden_size"]),
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.mean = np.asarray(
            checkpoint["normalization_mean"], dtype=np.float32
        )
        self.std = np.asarray(
            checkpoint["normalization_std"], dtype=np.float32
        )
        self.threshold = float(checkpoint["decision_threshold"])
        self.threshold_policy = str(
            checkpoint.get("decision_threshold_policy", "checkpoint")
        )
        self.model_version = str(checkpoint["model_version"])
        self.model_mode = str(checkpoint["model_mode"])
        self.prediction_horizon_seconds = tuple(
            float(value) for value in checkpoint["prediction_horizon_seconds"]
        )
        self.positive_anchor = str(checkpoint["positive_anchor"])
        self.confirmation_windows = int(confirmation_windows)
        self.extractor = RadarTemporalFeatureExtractorV2()

        self._frames: deque[RadarFrame] = deque()
        self._stream_key: tuple[str, object, object] | None = None
        self._last_frame_timestamp: datetime | None = None
        self._last_evaluation_timestamp: datetime | None = None
        self._consecutive_high_windows = 0
        self._event_latched = False

    def reset(self) -> None:
        self._frames.clear()
        self._stream_key = None
        self._last_frame_timestamp = None
        self._last_evaluation_timestamp = None
        self._clear_decision_state()

    def consume(self, frame: RadarFrame) -> TcnLiveResultV1 | None:
        if not isinstance(frame, RadarFrame):
            raise TypeError("TCN live predictor requires RadarFrame")
        stream_key = (frame.device_id, frame.room, frame.source_mode)
        if self._stream_key is not None and stream_key != self._stream_key:
            self.reset()
        if self._last_frame_timestamp is not None:
            gap = (frame.timestamp - self._last_frame_timestamp).total_seconds()
            if gap <= 0.0:
                raise ValueError("radar frame timestamps must be strictly increasing")
            if gap > 1.0:
                self.reset()
        self._stream_key = stream_key
        self._last_frame_timestamp = frame.timestamp
        self._frames.append(frame)
        while self._frames and (
            frame.timestamp - self._frames[0].timestamp
        ).total_seconds() > 2.2:
            self._frames.popleft()

        if self._last_evaluation_timestamp is not None and (
            frame.timestamp - self._last_evaluation_timestamp
        ).total_seconds() < 0.095:
            return None
        self._last_evaluation_timestamp = frame.timestamp
        observed_history_seconds = (
            frame.timestamp - self._frames[0].timestamp
        ).total_seconds()
        required_history_seconds = (
            WINDOW_SIZE_V2 - 1
        ) / self.extractor.target_sample_rate_hz
        if observed_history_seconds + 1e-9 < required_history_seconds:
            self._clear_decision_state()
            return self._unknown_result(frame, reason="WARMUP")

        window = self.extractor.transform(
            tuple(self._frames), end_timestamp=frame.timestamp
        )
        if window.names != FEATURE_NAMES_V2 or window.version != FEATURE_VERSION_V2:
            raise ValueError("live feature contract changed after predictor startup")
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            self._clear_decision_state()
            return self._unknown_result(
                frame,
                reason="INSUFFICIENT_DATA",
                window=window,
            )

        score = self._infer_score(window)
        high = score >= self.threshold
        if high:
            self._consecutive_high_windows += 1
        else:
            self._clear_decision_state()
        if self._consecutive_high_windows >= self.confirmation_windows:
            risk_state = "IMMINENT"
        elif high:
            risk_state = "WATCH"
        else:
            risk_state = "NORMAL"

        event_triggered = False
        event_id = None
        if risk_state == "IMMINENT" and not self._event_latched:
            event_triggered = True
            event_id = f"radar-prefall-{uuid4().hex}"
            self._event_latched = True
        return self._result(
            frame,
            risk_state=risk_state,
            pre_fall_score=score,
            score_valid=True,
            event_triggered=event_triggered,
            event_id=event_id,
            unknown_reason=None,
            window=window,
        )

    def _infer_score(self, window: TemporalFeatureWindowV2) -> float:
        values = np.asarray(window.values, dtype=np.float32)
        # Derive the zero-variance mask from the current std each call so that
        # a runtime calibration override (calibrated_tcn_live_v1) is respected.
        zero_variance_mask = zero_variance_feature_mask(self.std)
        normalized = safe_normalize(
            values,
            self.mean,
            self.std,
            zero_variance_mask=zero_variance_mask,
        )
        with torch.inference_mode():
            logit = self.model(
                torch.from_numpy(normalized).unsqueeze(0).to(self.device)
            )
            score = torch.sigmoid(logit).item()
        return float(np.clip(score, 0.0, 1.0))

    def _clear_decision_state(self) -> None:
        self._consecutive_high_windows = 0
        self._event_latched = False

    def _unknown_result(
        self,
        frame: RadarFrame,
        *,
        reason: str,
        window: TemporalFeatureWindowV2 | None = None,
    ) -> TcnLiveResultV1:
        return self._result(
            frame,
            risk_state="UNKNOWN",
            pre_fall_score=0.0,
            score_valid=False,
            event_triggered=False,
            event_id=None,
            unknown_reason=reason,
            window=window,
        )

    def _result(
        self,
        frame: RadarFrame,
        *,
        risk_state: str,
        pre_fall_score: float,
        score_valid: bool,
        event_triggered: bool,
        event_id: str | None,
        unknown_reason: str | None,
        window: TemporalFeatureWindowV2 | None,
    ) -> TcnLiveResultV1:
        centroid_z = None
        vertical_velocity = None
        height_delta_0_6s = None
        feature_point_count = None
        if (
            window is not None
            and window.data_quality is not TemporalDataQuality.INSUFFICIENT_DATA
        ):
            last = np.asarray(window.values, dtype=np.float32)[-1]
            feature_indices = {
                name: index for index, name in enumerate(FEATURE_NAMES_V2)
            }
            centroid_z = float(last[feature_indices["centroid_z"]])
            vertical_velocity = float(last[feature_indices["vertical_velocity"]])
            height_delta_0_6s = float(
                last[feature_indices["centroid_z_delta_0_6s"]]
            )
            feature_point_count = float(last[feature_indices["point_count"]])
        return TcnLiveResultV1(
            schema_version=TCN_LIVE_SCHEMA_VERSION,
            timestamp=frame.timestamp.isoformat(),
            emitted_at=datetime.now(timezone.utc).isoformat(),
            device_id=frame.device_id,
            room=frame.room.value,
            source_mode=frame.source_mode.value,
            risk_state=risk_state,
            pre_fall_score=float(pre_fall_score),
            score_valid=score_valid,
            consecutive_high_windows=self._consecutive_high_windows,
            event_triggered=event_triggered,
            event_id=event_id,
            unknown_reason=unknown_reason,
            data_quality=(
                window.data_quality.value
                if window is not None
                else TemporalDataQuality.INSUFFICIENT_DATA.value
            ),
            missing_frame_ratio=(
                float(window.missing_frame_ratio) if window is not None else 1.0
            ),
            longest_unresolved_gap_seconds=(
                float(window.longest_unresolved_gap_seconds)
                if window is not None
                else 0.0
            ),
            centroid_z=centroid_z,
            vertical_velocity=vertical_velocity,
            height_delta_0_6s=height_delta_0_6s,
            feature_point_count=feature_point_count,
            model_version=self.model_version,
            model_mode=self.model_mode,
            architecture=TCN_ARCHITECTURE,
            checkpoint_sha256=self.checkpoint_sha256,
            feature_version=FEATURE_VERSION_V2,
            threshold=self.threshold,
            threshold_policy=self.threshold_policy,
            prediction_horizon_seconds=self.prediction_horizon_seconds,
            positive_anchor=self.positive_anchor,
        )


def _load_and_validate_checkpoint(
    path: Path, device: str | torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("TCN checkpoint root must be a mapping")
    if checkpoint.get("model_version") != EXPERIMENT_MODEL_VERSION:
        raise ValueError("unsupported TCN checkpoint model_version")
    if checkpoint.get("model_mode") != RESEARCH_MODEL_MODE:
        raise ValueError("TCN checkpoint is not weak-supervision research")
    if checkpoint.get("model_architecture") != TCN_ARCHITECTURE:
        raise ValueError("TCN checkpoint architecture must be causal_tcn")
    if checkpoint.get("task_type") != "prefall_prediction":
        raise ValueError("TCN checkpoint task_type must be prefall_prediction")
    if bool(checkpoint.get("deployment_eligible", True)):
        raise ValueError("TCN shadow validation requires deployment_eligible=false")
    if not bool(checkpoint.get("shadow_only", False)):
        raise ValueError("TCN shadow validation requires shadow_only=true")
    if checkpoint.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("TCN checkpoint feature version is incompatible")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("TCN checkpoint feature names/order are incompatible")
    if int(checkpoint.get("window_size", -1)) != WINDOW_SIZE_V2:
        raise ValueError("TCN checkpoint window size is incompatible")
    if int(checkpoint.get("input_size", -1)) != len(FEATURE_NAMES_V2):
        raise ValueError("TCN checkpoint input size is incompatible")
    if int(checkpoint.get("hidden_size", 0)) <= 0:
        raise ValueError("TCN checkpoint hidden size is invalid")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("TCN checkpoint state_dict is missing")

    mean = np.asarray(checkpoint.get("normalization_mean"), dtype=np.float32)
    std = np.asarray(checkpoint.get("normalization_std"), dtype=np.float32)
    if mean.shape != (len(FEATURE_NAMES_V2),) or std.shape != mean.shape:
        raise ValueError("TCN checkpoint normalization shape is incompatible")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("TCN checkpoint normalization values are invalid")
    threshold = float(checkpoint.get("decision_threshold", np.nan))
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("TCN checkpoint decision threshold is invalid")
    horizon = tuple(
        float(value)
        for value in checkpoint.get("prediction_horizon_seconds", ())
    )
    if len(horizon) != 2 or not 0.0 < horizon[0] <= horizon[1]:
        raise ValueError("TCN checkpoint prediction horizon is invalid")
    if not str(checkpoint.get("positive_anchor", "")).strip():
        raise ValueError("TCN checkpoint positive anchor is missing")
    return checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
