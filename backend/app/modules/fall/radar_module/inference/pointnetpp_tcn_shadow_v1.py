from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.contracts import RadarFrame
from radar_module.model.pointnetpp_tcn_v1 import (
    ARCHITECTURE,
    INPUT_FEATURES,
    MODEL_VERSION,
    PointNetPlusPlusTcnPrefall,
)
from radar_module.preprocess.pointcloud_sequence import PointCloudSequenceBuilder


@dataclass(frozen=True, slots=True)
class RadarEncoderEvidenceV1:
    """Adapter contract consumed by the unchanged RadarEvidence layer."""

    score: float | None
    quality: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class PointNetPlusPlusTcnShadowPredictorV1:
    """Independent shadow encoder; it never writes Fusion or alert state."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(checkpoint_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.checkpoint_sha256 = _sha256(self.path)
        if self.checkpoint_sha256 != expected_sha256.lower():
            raise ValueError("PointNet++-TCN checkpoint SHA256 mismatch")
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        _validate(checkpoint)
        self.device = torch.device(device)
        self.model = PointNetPlusPlusTcnPrefall().to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
        self.threshold = float(checkpoint["decision_threshold"])
        self.builder = PointCloudSequenceBuilder()
        self._frames: deque[RadarFrame] = deque()
        self._stream_key: tuple[object, ...] | None = None
        self._last_timestamp: datetime | None = None
        self._last_evaluation: datetime | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._stream_key = None
        self._last_timestamp = None
        self._last_evaluation = None

    def consume(self, frame: RadarFrame) -> RadarEncoderEvidenceV1 | None:
        stream_key = (frame.device_id, frame.room, frame.source_mode)
        if self._stream_key is not None and stream_key != self._stream_key:
            self.reset()
        if self._last_timestamp is not None:
            gap = (frame.timestamp - self._last_timestamp).total_seconds()
            if gap <= 0:
                raise ValueError("radar timestamps must be strictly increasing")
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
        if (frame.timestamp - self._frames[0].timestamp).total_seconds() < 1.9:
            return RadarEncoderEvidenceV1(None, 0.0, frame.timestamp)
        sequence = self.builder.transform(tuple(self._frames), end_timestamp=frame.timestamp)
        observed = int(np.sum(sequence.frame_mask & np.any(sequence.point_mask, axis=1)))
        if observed < 10:
            return RadarEncoderEvidenceV1(None, 0.0, frame.timestamp)
        quality = 1.0 if observed == 20 else 0.6
        values = sequence.values[..., :5].copy()
        normalized = (values - self.mean[None, None, :]) / self.std[None, None, :]
        normalized[..., 4][sequence.values[..., 5] <= 0] = 0.0
        normalized[~sequence.point_mask] = 0.0
        with torch.inference_mode():
            score = torch.sigmoid(
                self.model(
                    torch.from_numpy(normalized).unsqueeze(0).to(self.device),
                    torch.from_numpy(sequence.point_mask).unsqueeze(0).to(self.device),
                    torch.from_numpy(sequence.frame_mask).unsqueeze(0).to(self.device),
                )
            ).item()
        return RadarEncoderEvidenceV1(float(np.clip(score, 0.0, 1.0)), quality, frame.timestamp)


def _validate(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported PointNet++-TCN model version")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("architecture mismatch")
    if tuple(checkpoint.get("feature_names", ())) != INPUT_FEATURES:
        raise ValueError("point feature order mismatch")
    if not bool(checkpoint.get("evaluation_locked", False)):
        raise ValueError("checkpoint threshold is not evaluation-locked")
    if bool(checkpoint.get("approved_to_replace_current_encoder", False)):
        raise ValueError("this class is reserved for a non-integrated shadow candidate")
    if not bool(checkpoint.get("shadow_only", False)) or bool(checkpoint.get("deployment_eligible", True)):
        raise ValueError("candidate must remain shadow-only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
