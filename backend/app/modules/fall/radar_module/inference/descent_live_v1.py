"""Descent early-detection live shadow predictor.

This is the real-time bridge for the descent early-detection model
(``descent_detection_tcn_v1.pt``), which detects an in-progress body descent
(fall-in-progress) and fires a warning before floor impact. It is distinct
from the frozen B0/calibrated TCN pre-fall predictors:

- TCN (B0 / calibrated): predicts the pre-fall window (fall-imminent before
  descent onset) using DGUHA 0.5-1.0s-before-onset supervision.
- Descent detection: detects *the descent itself* (fall has started, body is
  falling) using windows inside the descent interval [onset+0.3s, floor-0.3s].

The checkpoint uses the same ``TemporalBinaryModel`` architecture
(``causal_tcn``, 20x19 input), the same ``radar_features_v2`` contract, and a
separate set of normalization statistics / threshold.

Contract
--------
- SHA256 of the checkpoint is validated.
- Feature version/order/window/input size are validated against v2.
- Output is ``shadow_only=true``, ``alert_suppressed=true``.
- Never modifies the frozen checkpoint, UART/TLV parsing, or live chain.

Version: radar_descent_live_v1
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.contracts import RadarFrame
from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.safe_normalize import (
    safe_normalize,
    zero_variance_feature_mask,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
    WINDOW_SIZE_V2,
)


DESCENT_LIVE_SCHEMA_VERSION = "radar_descent_live_v1"
DESCENT_ARCHITECTURE = "causal_tcn"
DESCENT_DISCLAIMER = (
    "毫米波下降早期检测shadow输出，识别正在发生的失控下坠，触地前预警；"
    "不触发正式告警，不代表失衡前预测"
)


@dataclass(frozen=True, slots=True)
class DescentLiveResultV1:
    schema_version: str
    timestamp: str
    emitted_at: str
    device_id: str
    room: str
    source_mode: str
    descent_score: float
    score_valid: bool
    risk_state: str
    consecutive_high_windows: int
    event_triggered: bool
    event_id: str | None
    unknown_reason: str | None
    data_quality: str
    model_version: str
    model_mode: str
    architecture: str
    checkpoint_sha256: str
    feature_version: str
    threshold: float
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str
    shadow_only: bool = True
    alert_suppressed: bool = True
    disclaimer: str = DESCENT_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["prediction_horizon_seconds"] = list(
            self.prediction_horizon_seconds
        )
        return payload


class RadarDescentLivePredictorV1:
    """Run the frozen descent-detection TCN on timestamped point clouds."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        confirmation_windows: int = 3,
        calibration_path: str | Path | None = None,
        calibration_method: str = "iwr6843_fall102",
        device: str | torch.device = "cpu",
    ) -> None:
        if confirmation_windows < 2:
            raise ValueError("confirmation_windows must be at least two")
        expected = expected_checkpoint_sha256.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("expected_checkpoint_sha256 must be 64 hex digits")

        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        if self.checkpoint_sha256 != expected:
            raise ValueError(
                f"descent checkpoint SHA256 mismatch: expected {expected}, "
                f"got {self.checkpoint_sha256}"
            )

        ckpt = _load_and_validate(self.checkpoint_path)
        self.device = torch.device(device)
        self.model = TemporalBinaryModel(
            architecture=str(ckpt["model_architecture"]),
            input_size=int(ckpt["input_size"]),
            hidden_size=int(ckpt["hidden_size"]),
        )
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.mean = np.asarray(ckpt["normalization_mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["normalization_std"], dtype=np.float32)
        self.threshold = float(ckpt["decision_threshold"])
        self.model_version = str(ckpt["model_version"])
        self.model_mode = str(ckpt["model_mode"])
        self.confirmation_windows = int(confirmation_windows)
        self.extractor = RadarTemporalFeatureExtractorV2()

        # Optional same-sensor domain calibration: override normalization at
        # runtime (same pattern as calibrated_tcn_live_v1). This lets the
        # DGUHA-trained descent model run on the IWR6843 domain without
        # collapsing to ~0.
        self.calibration_method = calibration_method
        self.calibration_path: Path | None = None
        if calibration_path is not None:
            calib = _load_calibration(calibration_path)
            self.mean = np.asarray(calib["mean"], dtype=np.float32)
            self.std = np.asarray(calib["std"], dtype=np.float32)
            if self.mean.shape != (len(FEATURE_NAMES_V2),) or (
                self.std.shape != self.mean.shape
            ):
                raise ValueError("descent calibration mean/std shape incompatible")
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
                raise ValueError("descent calibration mean/std must be finite")
            if np.any(self.std < 0):
                raise ValueError("descent calibration std must be non-negative")
            self.calibration_path = Path(calibration_path).resolve()

        # Zero-variance features (e.g. quality masks constant in the
        # calibration domain) are forced to normalized value 0 at inference, so
        # a real-sensor outlier (e.g. one empty frame) cannot produce ~1e9 and
        # underflow the sigmoid.
        self.zero_variance_mask = zero_variance_feature_mask(self.std)

        self._frames: deque[RadarFrame] = deque()
        self._stream_key: tuple[str, str, str] | None = None
        self._last_frame_timestamp: datetime | None = None
        self._last_eval_timestamp: datetime | None = None
        self._consecutive_high = 0
        self._event_latched = False

    def reset(self) -> None:
        self._frames.clear()
        self._stream_key = None
        self._last_frame_timestamp = None
        self._last_eval_timestamp = None
        self._consecutive_high = 0
        self._event_latched = False

    def consume(self, frame: RadarFrame) -> DescentLiveResultV1 | None:
        if not isinstance(frame, RadarFrame):
            raise TypeError("descent live predictor requires RadarFrame")
        stream_key = (frame.device_id, frame.room.value, frame.source_mode.value)
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

        if self._last_eval_timestamp is not None and (
            frame.timestamp - self._last_eval_timestamp
        ).total_seconds() < 0.095:
            return None
        self._last_eval_timestamp = frame.timestamp

        required_history = (
            WINDOW_SIZE_V2 - 1
        ) / self.extractor.target_sample_rate_hz
        observed = (
            frame.timestamp - self._frames[0].timestamp
        ).total_seconds()
        if observed + 1e-9 < required_history:
            self._clear_decision_state()
            return self._unknown(frame, reason="WARMUP")

        window = self.extractor.transform(
            tuple(self._frames), end_timestamp=frame.timestamp
        )
        if window.names != FEATURE_NAMES_V2 or window.version != FEATURE_VERSION_V2:
            raise ValueError("live feature contract changed after startup")
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            self._clear_decision_state()
            return self._unknown(frame, reason="INSUFFICIENT_DATA", window=window)

        score = self._infer_score(window)
        high = score >= self.threshold
        if high:
            self._consecutive_high += 1
        else:
            self._clear_decision_state()
        if self._consecutive_high >= self.confirmation_windows:
            risk_state = "IMMINENT"
        elif high:
            risk_state = "WATCH"
        else:
            risk_state = "NORMAL"

        event_triggered = False
        event_id = None
        if risk_state == "IMMINENT" and not self._event_latched:
            event_triggered = True
            event_id = f"descent-prefall-{_uuid4_hex()}"
            self._event_latched = True

        return DescentLiveResultV1(
            schema_version=DESCENT_LIVE_SCHEMA_VERSION,
            timestamp=frame.timestamp.isoformat(),
            emitted_at=datetime.now(timezone.utc).isoformat(),
            device_id=frame.device_id,
            room=frame.room.value,
            source_mode=frame.source_mode.value,
            descent_score=float(score),
            score_valid=True,
            risk_state=risk_state,
            consecutive_high_windows=self._consecutive_high,
            event_triggered=event_triggered,
            event_id=event_id,
            unknown_reason=None,
            data_quality=window.data_quality.value,
            model_version=self.model_version,
            model_mode=self.model_mode,
            architecture=DESCENT_ARCHITECTURE,
            checkpoint_sha256=self.checkpoint_sha256,
            feature_version=FEATURE_VERSION_V2,
            threshold=self.threshold,
            prediction_horizon_seconds=(0.0, 1.5),
            positive_anchor="descent_interval",
        )

    def _infer_score(self, window: Any) -> float:
        values = np.asarray(window.values, dtype=np.float32)
        normalized = safe_normalize(
            values,
            self.mean,
            self.std,
            zero_variance_mask=self.zero_variance_mask,
        )
        with torch.inference_mode():
            logit = self.model(
                torch.from_numpy(normalized).unsqueeze(0).to(self.device)
            )
            score = torch.sigmoid(logit).item()
        return float(np.clip(score, 0.0, 1.0))

    def _clear_decision_state(self) -> None:
        self._consecutive_high = 0
        self._event_latched = False

    def _unknown(
        self,
        frame: RadarFrame,
        *,
        reason: str,
        window: Any | None = None,
    ) -> DescentLiveResultV1:
        return DescentLiveResultV1(
            schema_version=DESCENT_LIVE_SCHEMA_VERSION,
            timestamp=frame.timestamp.isoformat(),
            emitted_at=datetime.now(timezone.utc).isoformat(),
            device_id=frame.device_id,
            room=frame.room.value,
            source_mode=frame.source_mode.value,
            descent_score=0.0,
            score_valid=False,
            risk_state="UNKNOWN",
            consecutive_high_windows=0,
            event_triggered=False,
            event_id=None,
            unknown_reason=reason,
            data_quality=(
                window.data_quality.value
                if window is not None
                else TemporalDataQuality.INSUFFICIENT_DATA.value
            ),
            model_version=self.model_version,
            model_mode=self.model_mode,
            architecture=DESCENT_ARCHITECTURE,
            checkpoint_sha256=self.checkpoint_sha256,
            feature_version=FEATURE_VERSION_V2,
            threshold=self.threshold,
            prediction_horizon_seconds=(0.0, 1.5),
            positive_anchor="descent_interval",
        )


def _load_and_validate(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("descent checkpoint root must be a mapping")
    if ckpt.get("model_architecture") != "causal_tcn":
        raise ValueError("descent checkpoint must be causal_tcn")
    if ckpt.get("task_type") != "descent_early_detection":
        raise ValueError("descent checkpoint task_type mismatch")
    if bool(ckpt.get("deployment_eligible", True)):
        raise ValueError("descent checkpoint must be deployment_eligible=false")
    if not bool(ckpt.get("shadow_only", False)):
        raise ValueError("descent checkpoint must be shadow_only=true")
    if ckpt.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("descent checkpoint feature version incompatible")
    if tuple(ckpt.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("descent checkpoint feature names/order incompatible")
    if int(ckpt.get("window_size", -1)) != WINDOW_SIZE_V2:
        raise ValueError("descent checkpoint window size incompatible")
    if int(ckpt.get("input_size", -1)) != len(FEATURE_NAMES_V2):
        raise ValueError("descent checkpoint input size incompatible")
    mean = np.asarray(ckpt.get("normalization_mean"), dtype=np.float32)
    std = np.asarray(ckpt.get("normalization_std"), dtype=np.float32)
    if mean.shape != (len(FEATURE_NAMES_V2),) or std.shape != mean.shape:
        raise ValueError("descent checkpoint normalization shape incompatible")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("descent checkpoint normalization invalid")
    threshold = float(ckpt.get("decision_threshold", np.nan))
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("descent checkpoint threshold invalid")
    return ckpt


def _load_calibration(path: str | Path) -> dict[str, Any]:
    """Load a same-sensor normalization calibration file (runtime override)."""
    calibration_path = Path(path).resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "radar_calibrated_normalization_v1":
        raise ValueError("unsupported calibration schema version")
    if data.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("calibration feature version incompatible")
    if tuple(data.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("calibration feature names/order incompatible")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uuid4_hex() -> str:
    import uuid

    return uuid.uuid4().hex[:12]
