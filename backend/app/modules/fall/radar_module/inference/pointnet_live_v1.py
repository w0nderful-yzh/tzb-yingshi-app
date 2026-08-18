from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from radar_module.contracts import RadarFrame
from radar_module.dataset.point_iwr6843_adaptation_v1 import FEATURE_NAMES, SEQUENCE_VERSION
from radar_module.model.pointnet_formal_prediction_v1 import FORMAL_MODEL_VERSION
from radar_module.model.point_temporal import PointTemporalEncoder, PointTemporalPredictionHead
from radar_module.preprocess.pointcloud_sequence import PointCloudSequenceBuilder


POINTNET_LIVE_SCHEMA_VERSION = "radar_pointnet_live_v1"
POINTNET_ARCHITECTURE = "pointnet_gru"
POINTNET_DISCLAIMER = (
    "毫米波PointNet-GRU输出是短时运动风险证据，用于多模态融合，不单独构成高可靠跌倒判断"
)


@dataclass(frozen=True, slots=True)
class PointNetLiveResultV1:
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
    observed_frame_count: int
    point_count: int
    snr_available_fraction: float
    model_version: str
    model_variant: str
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
    disclaimer: str = POINTNET_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["prediction_horizon_seconds"] = list(self.prediction_horizon_seconds)
        return payload


class RadarPointNetLivePredictorV1:
    """Independent PointNet-GRU live inference; it does not alter UART/TLV parsing."""

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
        expected = expected_checkpoint_sha256.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("expected_checkpoint_sha256 must contain 64 hex digits")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        if self.checkpoint_sha256 != expected:
            raise ValueError("PointNet checkpoint SHA256 mismatch")
        checkpoint = _load_and_validate(self.checkpoint_path)
        self.device = torch.device(device)
        encoder = PointTemporalEncoder(
            input_size=5,
            frame_hidden_size=int(checkpoint["frame_hidden_size"]),
            temporal_hidden_size=int(checkpoint["temporal_hidden_size"]),
        )
        self.model = PointTemporalPredictionHead(encoder, horizon_count=1)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
        self.threshold = float(checkpoint["decision_threshold"])
        self.threshold_policy = str(checkpoint["decision_threshold_policy"])
        self.model_version = str(checkpoint["model_version"])
        self.model_variant = str(checkpoint["variant"])
        self.prediction_horizon_seconds = tuple(float(v) for v in checkpoint["prediction_horizon_seconds"])
        self.positive_anchor = str(checkpoint["positive_anchor"])
        self.confirmation_windows = int(confirmation_windows)
        self.builder = PointCloudSequenceBuilder()
        self._frames: deque[RadarFrame] = deque()
        self._stream_key: tuple[object, ...] | None = None
        self._last_timestamp: datetime | None = None
        self._last_evaluation: datetime | None = None
        self._consecutive_high = 0
        self._event_latched = False

    def reset(self) -> None:
        self._frames.clear()
        self._stream_key = None
        self._last_timestamp = None
        self._last_evaluation = None
        self._clear_decision()

    def consume(self, frame: RadarFrame) -> PointNetLiveResultV1 | None:
        if not isinstance(frame, RadarFrame):
            raise TypeError("PointNet live predictor requires RadarFrame")
        stream_key = (frame.device_id, frame.room, frame.source_mode)
        if self._stream_key is not None and self._stream_key != stream_key:
            self.reset()
        if self._last_timestamp is not None:
            gap = (frame.timestamp - self._last_timestamp).total_seconds()
            if gap <= 0:
                raise ValueError("radar frame timestamps must be strictly increasing")
            if gap > 1.0:
                self.reset()
        self._stream_key = stream_key
        self._last_timestamp = frame.timestamp
        self._frames.append(frame)
        while self._frames and (frame.timestamp - self._frames[0].timestamp).total_seconds() > 2.2:
            self._frames.popleft()
        if self._last_evaluation is not None and (frame.timestamp - self._last_evaluation).total_seconds() < 0.095:
            return None
        self._last_evaluation = frame.timestamp
        if (frame.timestamp - self._frames[0].timestamp).total_seconds() + 1e-9 < 1.9:
            self._clear_decision()
            return self._result(frame, score=0.0, score_valid=False, state="UNKNOWN", reason="WARMUP", sequence=None)
        sequence = self.builder.transform(tuple(self._frames), end_timestamp=frame.timestamp)
        observed = _observed_point_frame_count(sequence)
        if observed < 10:
            self._clear_decision()
            return self._result(
                frame, score=0.0, score_valid=False, state="UNKNOWN",
                reason="INSUFFICIENT_POINT_FRAMES", sequence=sequence,
            )
        raw = sequence.values[..., :5].copy()
        normalized = (raw - self.mean[None, None, :]) / self.std[None, None, :]
        # Live official parser supplies SNR in dB, matching the corrected Fall-102 contract.
        normalized[~sequence.point_mask] = 0.0
        with torch.inference_mode():
            logit = self.model(
                torch.from_numpy(normalized).unsqueeze(0).to(self.device),
                torch.from_numpy(sequence.point_mask).unsqueeze(0).to(self.device),
                torch.from_numpy(sequence.frame_mask).unsqueeze(0).to(self.device),
            ).squeeze()
            score = float(torch.sigmoid(logit).item())
        high = score >= self.threshold
        if high:
            self._consecutive_high += 1
        else:
            self._clear_decision()
        state = "IMMINENT" if self._consecutive_high >= self.confirmation_windows else "WATCH" if high else "NORMAL"
        event_triggered = state == "IMMINENT" and not self._event_latched
        event_id = f"radar-pointnet-prefall-{uuid4().hex}" if event_triggered else None
        if event_triggered:
            self._event_latched = True
        return self._result(
            frame, score=score, score_valid=True, state=state, reason=None,
            sequence=sequence, event_triggered=event_triggered, event_id=event_id,
        )

    def _clear_decision(self) -> None:
        self._consecutive_high = 0
        self._event_latched = False

    def _result(
        self, frame: RadarFrame, *, score: float, score_valid: bool, state: str,
        reason: str | None, sequence: Any, event_triggered: bool = False,
        event_id: str | None = None,
    ) -> PointNetLiveResultV1:
        observed = _observed_point_frame_count(sequence) if sequence is not None else 0
        point_count = len(frame.points)
        snr_present = sum(point.snr is not None for point in frame.points)
        quality = "GOOD" if observed == 20 else "DEGRADED" if observed >= 10 else "INSUFFICIENT_DATA"
        return PointNetLiveResultV1(
            schema_version=POINTNET_LIVE_SCHEMA_VERSION,
            timestamp=frame.timestamp.isoformat(), emitted_at=datetime.now(timezone.utc).isoformat(),
            device_id=frame.device_id, room=frame.room.value, source_mode=frame.source_mode.value,
            risk_state=state, pre_fall_score=float(np.clip(score, 0.0, 1.0)),
            score_valid=score_valid, consecutive_high_windows=self._consecutive_high,
            event_triggered=event_triggered, event_id=event_id, unknown_reason=reason,
            data_quality=quality, missing_frame_ratio=1.0 - observed / 20.0,
            observed_frame_count=observed, point_count=point_count,
            snr_available_fraction=snr_present / max(point_count, 1),
            model_version=self.model_version, model_variant=self.model_variant,
            model_mode="RESEARCH_WEAK_SUPERVISION", architecture=POINTNET_ARCHITECTURE,
            checkpoint_sha256=self.checkpoint_sha256, feature_version=SEQUENCE_VERSION,
            threshold=self.threshold, threshold_policy=self.threshold_policy,
            prediction_horizon_seconds=self.prediction_horizon_seconds,
            positive_anchor=self.positive_anchor,
        )


def _observed_point_frame_count(sequence: Any) -> int:
    """Count frames that contain measured points, not timestamp-only empty frames."""

    return int(np.sum(sequence.frame_mask & np.any(sequence.point_mask, axis=1)))


def _load_and_validate(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_version") != FORMAL_MODEL_VERSION:
        raise ValueError("unsupported PointNet checkpoint")
    if checkpoint.get("model_role") != "pointnet_gru_short_horizon_radar_evidence":
        raise ValueError("checkpoint role mismatch")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("PointNet feature order mismatch")
    if checkpoint.get("sequence_version") != SEQUENCE_VERSION or int(checkpoint.get("input_size", 0)) != 5:
        raise ValueError("PointNet sequence contract mismatch")
    if int(checkpoint.get("time_steps", 0)) != 20 or int(checkpoint.get("max_points", 0)) != 64:
        raise ValueError("PointNet tensor contract mismatch")
    if bool(checkpoint.get("iwr_fall_recordings_used_as_prediction_positive", True)):
        raise ValueError("IWR fall labels are forbidden")
    if not bool(checkpoint.get("selected_for_radar_branch", False)):
        raise ValueError("checkpoint was not locked by formal evaluation")
    if not bool(checkpoint.get("shadow_only", False)) or bool(checkpoint.get("deployment_eligible", True)):
        raise ValueError("PointNet branch must remain research shadow evidence")
    mean = np.asarray(checkpoint.get("normalization_mean"), dtype=np.float32)
    std = np.asarray(checkpoint.get("normalization_std"), dtype=np.float32)
    if mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("PointNet normalization contract invalid")
    threshold = float(checkpoint.get("decision_threshold", np.nan))
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("PointNet threshold invalid")
    return checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
