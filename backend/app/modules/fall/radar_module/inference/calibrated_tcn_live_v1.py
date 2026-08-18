"""Domain-calibrated TCN live shadow predictor with decision gating.

This is an *experimental shadow branch* that reuses the frozen causal TCN
weights but replaces the checkpoint's stored normalization statistics with a
domain-calibrated mean/std computed from real IWR6843 sensor replays. The
calibrated score stream is then fed through :class:`DecisionGateV1` so that
controlled-lowering episodes (sit/squat/bend followed by recovery) are
suppressed instead of escalating.

Contract:
- The frozen TCN checkpoint is NOT modified; the original architecture, state
  dict, threshold and feature contract are unchanged.
- The checkpoint's stored normalization is overridden at *runtime* by the
  calibration file. SHA256 of the checkpoint is still validated.
- Output is shadow-only and alert-suppressed. It never emits a formal alert.
- This branch is for making real-sensor scores visible and less noisy during
  a live demo; it is NOT deployment validation.

Version: radar_calibrated_tcn_live_v1
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from radar_module.contracts import RadarFrame
from radar_module.inference.decision_gate_v1 import (
    DecisionGateV1,
    GateDecisionV1,
    GateState,
)
from radar_module.inference.tcn_live_v1 import (
    RadarTcnLivePredictorV1,
    TcnLiveResultV1,
)
from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
    WINDOW_SIZE_V2,
)


CALIBRATED_TCN_SCHEMA_VERSION = "radar_calibrated_tcn_live_v1"
CALIBRATED_NORMALIZATION_SCHEMA_VERSION = "radar_calibrated_normalization_v1"
CALIBRATED_TCN_DISCLAIMER = (
    "域校准TCN shadow输出，仅用于实时演示；不触发正式告警，不代表已验证的跌倒预测"
)


@dataclass(frozen=True, slots=True)
class CalibratedTcnLiveResultV1:
    schema_version: str
    timestamp: str
    emitted_at: str
    device_id: str
    room: str
    source_mode: str
    # calibrated TCN score (0..1)
    pre_fall_score: float
    score_valid: bool
    # raw TCN risk state under calibrated normalization
    tcn_risk_state: str
    # decision-gate state (NORMAL/WATCH/IMMINENT/SUPPRESSED_RECOVERY/CONFIRMED)
    gate_state: str
    formal_alert: bool
    suppressed_reason: str | None
    recovery_window_active: bool
    recovery_count: int
    consecutive_high_windows: int
    threshold_crossed_at: str | None
    confirmed_at: str | None
    confirmation_latency_seconds: float | None
    unknown_reason: str | None
    data_quality: str
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
    normalization_source: str
    calibration_method: str
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str
    shadow_only: bool = True
    alert_suppressed: bool = True
    disclaimer: str = CALIBRATED_TCN_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["prediction_horizon_seconds"] = list(
            self.prediction_horizon_seconds
        )
        return payload


class CalibratedTcnLivePredictorV1:
    """Reuse frozen TCN weights with domain-calibrated normalization + gate.

    The TCN checkpoint is loaded and SHA256-validated exactly as the normal
    live predictor. After loading, ``normalization_mean``/``normalization_std``
    are overridden by the calibration file. Scores are then passed through a
    :class:`DecisionGateV1` for recovery suppression.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        calibration_path: str | Path,
        calibration_method: str = "real_gaussian",
        confirmation_windows: int = 3,
        recovery_windows: int = 2,
        recovery_window_seconds: float = 1.5,
        persist_confirm_seconds: float = 0.0,
        emit_formal_alert: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        # Build the frozen predictor (validates checkpoint SHA256 + contracts).
        self._tcn = RadarTcnLivePredictorV1(
            checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            confirmation_windows=confirmation_windows,
            device=device,
        )
        # Override normalization at runtime with calibrated stats.
        self.calibration = _load_calibration(calibration_path, calibration_method)
        self._tcn.mean = np.asarray(
            self.calibration["mean"], dtype=np.float32
        )
        self._tcn.std = np.asarray(self.calibration["std"], dtype=np.float32)
        if self._tcn.mean.shape != (len(FEATURE_NAMES_V2),) or (
            self._tcn.std.shape != self._tcn.mean.shape
        ):
            raise ValueError("calibration mean/std shape incompatible")
        if not np.isfinite(self._tcn.mean).all() or not np.isfinite(
            self._tcn.std
        ).all():
            raise ValueError("calibration mean/std must be finite")
        if np.any(self._tcn.std <= 0):
            raise ValueError("calibration std must be positive")

        self.calibration_method = calibration_method
        self.calibration_path = Path(calibration_path).resolve()
        self.gate = DecisionGateV1(
            threshold=float(self._tcn.threshold),
            confirmation_windows=confirmation_windows,
            recovery_windows=recovery_windows,
            recovery_window_seconds=recovery_window_seconds,
            persist_confirm_seconds=persist_confirm_seconds,
            emit_formal_alert=emit_formal_alert,
        )
        self._threshold_crossed_at: datetime | None = None
        self._confirmed_at: datetime | None = None

    @property
    def checkpoint_sha256(self) -> str:
        return self._tcn.checkpoint_sha256

    @property
    def model_version(self) -> str:
        return self._tcn.model_version

    @property
    def model_mode(self) -> str:
        return self._tcn.model_mode

    @property
    def threshold(self) -> float:
        return self._tcn.threshold

    @property
    def prediction_horizon_seconds(self) -> tuple[float, float]:
        return self._tcn.prediction_horizon_seconds

    @property
    def positive_anchor(self) -> str:
        return self._tcn.positive_anchor

    def reset(self) -> None:
        self._tcn.reset()
        self.gate.reset()
        self._threshold_crossed_at = None
        self._confirmed_at = None

    def consume(self, frame: RadarFrame) -> CalibratedTcnLiveResultV1 | None:
        tcn_result = self._tcn.consume(frame)
        if tcn_result is None:
            return None
        gate_decision: GateDecisionV1 | None = None
        if tcn_result.score_valid:
            result_timestamp = _parse_timestamp(tcn_result.timestamp)
            gate_decision = self.gate.consume(
                timestamp=result_timestamp,
                score=tcn_result.pre_fall_score,
            )
            if tcn_result.pre_fall_score >= self.threshold:
                if self._threshold_crossed_at is None:
                    self._threshold_crossed_at = result_timestamp
                    self._confirmed_at = None
                if (
                    gate_decision.state in {GateState.IMMINENT, GateState.CONFIRMED}
                    and self._confirmed_at is None
                ):
                    self._confirmed_at = result_timestamp
            elif gate_decision.state in {
                GateState.NORMAL,
                GateState.SUPPRESSED_RECOVERY,
            }:
                self._threshold_crossed_at = None
                self._confirmed_at = None
        else:
            # Data insufficient / warmup: reset gate so no stale episode leaks.
            self.gate.reset()
            self._threshold_crossed_at = None
            self._confirmed_at = None

        return self._result(tcn_result, gate_decision)

    def _result(
        self,
        tcn: TcnLiveResultV1,
        gate: GateDecisionV1 | None,
    ) -> CalibratedTcnLiveResultV1:
        if gate is None:
            gate_state = GateState.NORMAL.value
            formal_alert = False
            suppressed_reason = None
            recovery_active = False
            recovery_count = 0
            consecutive = int(tcn.consecutive_high_windows)
        else:
            gate_state = gate.state.value
            formal_alert = bool(gate.formal_alert)
            suppressed_reason = gate.suppressed_reason
            recovery_active = bool(gate.recovery_window_active)
            recovery_count = int(gate.recovery_count)
            consecutive = int(gate.consecutive_high_windows)
        threshold_crossed_at = (
            self._threshold_crossed_at.isoformat()
            if self._threshold_crossed_at is not None
            else None
        )
        confirmed_at = (
            self._confirmed_at.isoformat()
            if self._confirmed_at is not None
            else None
        )
        confirmation_latency_seconds = None
        if self._threshold_crossed_at is not None and self._confirmed_at is not None:
            confirmation_latency_seconds = (
                self._confirmed_at - self._threshold_crossed_at
            ).total_seconds()
        return CalibratedTcnLiveResultV1(
            schema_version=CALIBRATED_TCN_SCHEMA_VERSION,
            timestamp=tcn.timestamp,
            emitted_at=tcn.emitted_at,
            device_id=tcn.device_id,
            room=tcn.room,
            source_mode=tcn.source_mode,
            pre_fall_score=tcn.pre_fall_score,
            score_valid=tcn.score_valid,
            tcn_risk_state=tcn.risk_state,
            gate_state=gate_state,
            formal_alert=formal_alert,
            suppressed_reason=suppressed_reason,
            recovery_window_active=recovery_active,
            recovery_count=recovery_count,
            consecutive_high_windows=consecutive,
            threshold_crossed_at=threshold_crossed_at,
            confirmed_at=confirmed_at,
            confirmation_latency_seconds=confirmation_latency_seconds,
            unknown_reason=tcn.unknown_reason,
            data_quality=tcn.data_quality,
            centroid_z=tcn.centroid_z,
            vertical_velocity=tcn.vertical_velocity,
            height_delta_0_6s=tcn.height_delta_0_6s,
            feature_point_count=tcn.feature_point_count,
            model_version=tcn.model_version,
            model_mode=tcn.model_mode,
            architecture=tcn.architecture,
            checkpoint_sha256=tcn.checkpoint_sha256,
            feature_version=tcn.feature_version,
            threshold=tcn.threshold,
            normalization_source=str(self.calibration_path),
            calibration_method=self.calibration_method,
            prediction_horizon_seconds=tcn.prediction_horizon_seconds,
            positive_anchor=tcn.positive_anchor,
        )


def _load_calibration(path: str | Path, method: str) -> dict[str, Any]:
    calibration_path = Path(path).resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != CALIBRATED_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration schema version")
    if data.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("calibration feature version incompatible")
    if tuple(data.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("calibration feature names/order incompatible")
    return data


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("calibrated TCN timestamp must be timezone-aware")
    return parsed


def _build_parser() -> Any:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="CalibratedTCN shadow predictor CLI smoke test."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--method", default="real_gaussian")
    return parser


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    args = _build_parser().parse_args()
    predictor = CalibratedTcnLivePredictorV1(
        args.checkpoint,
        expected_checkpoint_sha256=args.sha256,
        calibration_path=args.calibration,
        calibration_method=args.method,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "checkpoint_sha256": predictor.checkpoint_sha256,
                "threshold": predictor.threshold,
                "calibration": predictor.calibration_method,
            },
            ensure_ascii=False,
        )
    )
