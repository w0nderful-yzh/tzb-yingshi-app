"""Rule-based fall-risk assessment from IWR6843 point-cloud features.

Scope
-----
This is a *fall-risk assessment*, not a fall detector and not a pre-fall
predictor. It estimates a short-term room-level risk level (LOW / MODERATE /
HIGH) from clinically-motivated balance and behaviour signals that are
measurable from the existing ``radar_features_v2`` point-cloud feature stream:

- Postural sway / instability: jitter of body height (centroid_z / z_p90)
  during low-motion standing. Larger sway is a clinically validated fall-risk
  indicator (e.g. the MIT OLST study).
- Behavioural activity: slower movements, reduced activity, lower Doppler
  activity. Frailty and reduced mobility are established fall-risk factors.
- Recent descent events: a recent fast downward motion raises risk.

Method
------
Pure transparent rules over aggregated window statistics. No training, no
checkpoint, no label migration, no modification of the frozen model or the
live inference chain. Output is shadow-only and alert-suppressed.

Rationale for this design
-------------------------
OLST (single-leg stance) is clinically validated: shorter hold time and larger
sway predict fall risk. Its radar RDM features do not transfer to IWR6843
point clouds, but the *concept* (sway + mobility = risk) transfers directly to
the point-cloud features we already compute. Because this is rule-based, it
needs no labelled point-cloud data and can run on the real sensor now.

Contract
--------
- Consumes ``TemporalFeatureWindowV2`` (20 x 19 radar_features_v2).
- Maintains a rolling assessment window (default 60 s).
- Emits ``FallRiskAssessmentResultV1`` with level, score, and component
  breakdown.
- ``shadow_only=true``, ``alert_suppressed=true``.

Version: radar_fall_risk_assessment_v1
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    TemporalDataQuality,
    TemporalFeatureWindowV2,
)


ASSESSMENT_SCHEMA_VERSION = "radar_fall_risk_assessment_v1"
ASSESSMENT_DISCLAIMER = (
    "基于点云稳定性和行为模式的规则化跌倒风险评估，shadow输出，不触发正式告警；"
    "不是跌倒检测或失衡前预测"
)


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FallRiskAssessmentResultV1:
    schema_version: str
    timestamp: str
    device_id: str
    room: str
    risk_level: RiskLevel
    risk_score: float | None
    # component scores in [0,1]
    sway_risk: float | None
    mobility_risk: float | None
    descent_risk: float | None
    # diagnostics
    assessment_window_seconds: float
    valid_window_count: int
    observed_duration_seconds: float
    unknown_reason: str | None = None
    shadow_only: bool = True
    alert_suppressed: bool = True
    disclaimer: str = ASSESSMENT_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "room": self.room,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "sway_risk": self.sway_risk,
            "mobility_risk": self.mobility_risk,
            "descent_risk": self.descent_risk,
            "assessment_window_seconds": self.assessment_window_seconds,
            "valid_window_count": self.valid_window_count,
            "observed_duration_seconds": self.observed_duration_seconds,
            "unknown_reason": self.unknown_reason,
            "shadow_only": self.shadow_only,
            "alert_suppressed": self.alert_suppressed,
            "disclaimer": self.disclaimer,
        }


class FallRiskAssessmentV1:
    """Rolling rule-based fall-risk assessment over feature windows."""

    def __init__(
        self,
        *,
        assessment_window_seconds: float = 60.0,
        max_windows: int = 600,
        sway_velocity_threshold_mps: float = 0.10,
        standing_point_count_threshold: float = 4.0,
        presence_point_count_threshold: float = 2.0,
        presence_recent_frames: int = 5,
    ) -> None:
        if assessment_window_seconds <= 0:
            raise ValueError("assessment_window_seconds must be positive")
        if max_windows < 10:
            raise ValueError("max_windows must be at least ten")
        if presence_point_count_threshold <= 0:
            raise ValueError("presence_point_count_threshold must be positive")
        if presence_recent_frames < 1:
            raise ValueError("presence_recent_frames must be positive")
        self.assessment_window_seconds = float(assessment_window_seconds)
        self.max_windows = int(max_windows)
        self.sway_velocity_threshold_mps = sway_velocity_threshold_mps
        self.standing_point_count_threshold = standing_point_count_threshold
        self.presence_point_count_threshold = float(
            presence_point_count_threshold
        )
        self.presence_recent_frames = int(presence_recent_frames)

        self._indices = {name: i for i, name in enumerate(FEATURE_NAMES_V2)}
        # rolling buffer of (timestamp, window_values)
        self._buffer: deque[tuple[datetime, np.ndarray]] = deque()
        self._last_timestamp: datetime | None = None
        self._stream_key: tuple[str, str] | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self._last_timestamp = None
        self._stream_key = None

    def consume(self, window: TemporalFeatureWindowV2) -> FallRiskAssessmentResultV1 | None:
        if not isinstance(window, TemporalFeatureWindowV2):
            raise TypeError("risk assessment requires TemporalFeatureWindowV2")
        stream_key = (window.device_id, window.room.value)
        if self._stream_key is not None and stream_key != self._stream_key:
            self.reset()
        self._stream_key = stream_key

        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            return None
        if self._last_timestamp is not None and window.end_timestamp <= self._last_timestamp:
            raise ValueError("window timestamps must be strictly increasing")
        self._last_timestamp = window.end_timestamp

        # Occupancy is a prerequisite for assessing a person's motion.  Empty
        # frames are represented by zero-valued features and may still have
        # DEGRADED (rather than INSUFFICIENT_DATA) quality.  Without this gate,
        # the transition from a real body height to zero is misread as extreme
        # postural sway and produces the deterministic 0.4 MODERATE score.
        current_values = np.asarray(window.values, dtype=np.float32)
        point_count_index = self._indices["point_count"]
        recent_point_count = current_values[
            -self.presence_recent_frames :, point_count_index
        ]
        target_present = bool(
            np.any(recent_point_count >= self.presence_point_count_threshold)
        )
        if not target_present:
            self._buffer.clear()
            return self._unknown(window, reason="NO_TARGET")

        self._buffer.append((window.end_timestamp, np.asarray(window.values, dtype=np.float32)))
        # prune old
        while self._buffer and (
            window.end_timestamp - self._buffer[0][0]
        ).total_seconds() > self.assessment_window_seconds:
            self._buffer.popleft()
        while len(self._buffer) > self.max_windows:
            self._buffer.popleft()

        if len(self._buffer) < 5:
            return None  # not enough data to assess yet

        return self._assess(window)

    def _unknown(
        self,
        latest: TemporalFeatureWindowV2,
        *,
        reason: str,
    ) -> FallRiskAssessmentResultV1:
        return FallRiskAssessmentResultV1(
            schema_version=ASSESSMENT_SCHEMA_VERSION,
            timestamp=latest.end_timestamp.isoformat(),
            device_id=latest.device_id,
            room=latest.room.value,
            risk_level=RiskLevel.UNKNOWN,
            risk_score=None,
            sway_risk=None,
            mobility_risk=None,
            descent_risk=None,
            assessment_window_seconds=self.assessment_window_seconds,
            valid_window_count=0,
            observed_duration_seconds=0.0,
            unknown_reason=reason,
        )

    def _assess(self, latest: TemporalFeatureWindowV2) -> FallRiskAssessmentResultV1:
        timestamps = [t for t, _ in self._buffer]
        values = np.stack([v for _, v in self._buffer])  # (N, 20, 19)
        duration = (
            timestamps[-1] - timestamps[0]
        ).total_seconds() if len(timestamps) > 1 else 0.0

        idx = self._indices
        # ---- feature extraction over the window ----
        last_frame = values[:, -1, :]  # (N, 19) each window's last frame
        # postural sway: std of centroid_z and z_p90 across the assessment window
        centroid_z = last_frame[:, idx["centroid_z"]]
        z_p90 = last_frame[:, idx["z_p90"]]
        height_range = last_frame[:, idx["height_range"]]
        point_count = last_frame[:, idx["point_count"]]
        mean_velocity = last_frame[:, idx["mean_velocity"]]
        max_abs_velocity = last_frame[:, idx["max_abs_velocity"]]
        moving_range_width = last_frame[:, idx["moving_range_width"]]

        # Standing sway: consider windows that have a clear body and low motion.
        # "clear body" requires a decent point count so we don't confuse a
        # weak/distant signal with a standing person.
        body_present = point_count >= self.standing_point_count_threshold
        low_motion = np.abs(mean_velocity) < self.sway_velocity_threshold_mps
        stance_mask = body_present & low_motion
        if stance_mask.sum() >= 3:
            sway_z = float(np.std(centroid_z[stance_mask]))
            sway_z90 = float(np.std(z_p90[stance_mask]))
            # use the max of the two jitter metrics; scale z_p90 (higher magnitude)
            sway_metric = max(sway_z * 2.0, sway_z90)
        else:
            # not enough clean stance windows; fall back to raw centroid variation
            sway_metric = float(np.std(centroid_z)) * 2.0

        # Mobility: median |velocity| and median max_abs_velocity over the window
        mobility_velocity = float(np.median(np.abs(mean_velocity)))
        mobility_maxvel = float(np.median(max_abs_velocity))
        # range width: how far the person is moving laterally
        range_width = float(np.median(moving_range_width))

        # Descent risk: recent strong downward motion from latest window dynamics
        vertical_velocity = float(last_frame[-1, idx["vertical_velocity"]])
        height_delta_06 = float(last_frame[-1, idx["centroid_z_delta_0_6s"]])

        # ---- component scores (ramp thresholds, clinically informed) ----
        # sway: higher jitter -> more risk; real body sway is 0.02-0.15 m
        sway_risk = _ramp(sway_metric, 0.03, 0.12)
        # mobility: reduced activity (frailty) and erratic bursts both add risk.
        # Very low movement across the whole window (no sustained stance AND no
        # activity) suggests low mobility / prolonged inactivity -> risk.
        # We measure "how active" the person is in aggregate.
        p95_velocity = float(np.percentile(np.abs(mean_velocity), 95))
        # presence: how many windows have a clear body point cloud
        presence_ratio = float((point_count >= self.standing_point_count_threshold).mean())
        stable_stance = float(stance_mask.mean())
        # Low-activity risk only when the room is occupied (someone present) but
        # movement is almost absent AND the body is not in a stable stance.
        # A person standing still is healthy low-activity; a person who is
        # present but has no sustained stable posture and no movement is
        # concerning (e.g. lying unresponsive / severe frailty).
        low_activity = (
            p95_velocity < 0.25
            and mobility_maxvel < 0.60
            and range_width < 0.50
            and stable_stance < 0.6
            and presence_ratio > 0.3
        )
        mobility_risk = 0.0
        if low_activity:
            mobility_risk = _ramp(0.25 - p95_velocity, -0.05, 0.25)
        # erratic bursts of very high velocity (unstable movement)
        mobility_risk = max(mobility_risk, _ramp(p95_velocity, 0.7, 1.4) * 0.6)

        # descent risk from recent downward motion
        descent_risk = max(
            _ramp(-vertical_velocity, 0.25, 0.60),
            _ramp(-height_delta_06, 0.08, 0.25),
        )

        # ---- overall ----
        risk_score = float(
            np.clip(0.40 * sway_risk + 0.30 * mobility_risk + 0.30 * descent_risk, 0.0, 1.0)
        )
        level = _risk_level(risk_score)

        return FallRiskAssessmentResultV1(
            schema_version=ASSESSMENT_SCHEMA_VERSION,
            timestamp=latest.end_timestamp.isoformat(),
            device_id=latest.device_id,
            room=latest.room.value,
            risk_level=level,
            risk_score=risk_score,
            sway_risk=sway_risk,
            mobility_risk=mobility_risk,
            descent_risk=descent_risk,
            assessment_window_seconds=self.assessment_window_seconds,
            valid_window_count=len(self._buffer),
            observed_duration_seconds=duration,
            unknown_reason=None,
        )


def _ramp(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("upper ramp bound must exceed lower bound")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _risk_level(score: float) -> RiskLevel:
    if score >= 0.45:
        return RiskLevel.HIGH
    if score >= 0.20:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


class RadarRiskAssessmentLiveV1:
    """Live wrapper: build temporal windows from RadarFrame and assess risk.

    This is the real-time bridge for the radar FastAPI. It maintains a rolling
    deque of ``RadarFrame`` objects, builds 2-second 20x19 windows with
    ``RadarTemporalFeatureExtractorV2``, and feeds them to
    :class:`FallRiskAssessmentV1`. Output is shadow-only.
    """

    LIVE_SCHEMA_VERSION = "radar_risk_assessment_live_v1"

    def __init__(
        self,
        *,
        assessment_window_seconds: float = 60.0,
        max_windows: int = 600,
        evaluation_stride_seconds: float = 0.2,
        presence_point_count_threshold: int = 2,
        presence_recent_frames: int = 5,
    ) -> None:
        from collections import deque as _deque

        from radar_module.preprocess.temporal_features_v2 import (
            RadarTemporalFeatureExtractorV2,
            WINDOW_SIZE_V2,
        )

        self.assessment_window_seconds = float(assessment_window_seconds)
        self.max_windows = int(max_windows)
        self.evaluation_stride_seconds = float(evaluation_stride_seconds)
        self.presence_point_count_threshold = int(
            presence_point_count_threshold
        )
        self.presence_recent_frames = int(presence_recent_frames)
        if self.presence_point_count_threshold < 1:
            raise ValueError("presence_point_count_threshold must be positive")
        if self.presence_recent_frames < 1:
            raise ValueError("presence_recent_frames must be positive")
        self.extractor = RadarTemporalFeatureExtractorV2()
        self._frames: _deque[object] = _deque()
        self._last_frame_timestamp: object | None = None
        self._last_eval_timestamp: object | None = None
        self._stream_key: tuple[str, str, str] | None = None
        self._assessor = FallRiskAssessmentV1(
            assessment_window_seconds=self.assessment_window_seconds,
            max_windows=self.max_windows,
            presence_point_count_threshold=float(
                self.presence_point_count_threshold
            ),
            presence_recent_frames=self.presence_recent_frames,
        )

    @property
    def threshold(self) -> None:
        return None

    def reset(self) -> None:
        self._frames.clear()
        self._last_frame_timestamp = None
        self._last_eval_timestamp = None
        self._stream_key = None
        self._assessor.reset()

    def consume(self, frame: object):
        from radar_module.contracts import RadarFrame
        from radar_module.preprocess.temporal_features_v2 import (
            TemporalDataQuality,
        )

        if not isinstance(frame, RadarFrame):
            raise TypeError("risk assessment live requires RadarFrame")
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
        ).total_seconds() < self.evaluation_stride_seconds:
            return None
        self._last_eval_timestamp = frame.timestamp

        required_history = (
            self.extractor.window_size - 1
        ) / self.extractor.target_sample_rate_hz
        observed = (
            frame.timestamp - self._frames[0].timestamp
        ).total_seconds()
        if observed + 1e-9 < required_history:
            reason = "NO_TARGET" if not self._recent_target_present() else "WARMUP"
            return self._unknown_payload(frame, reason=reason)

        window = self.extractor.transform(
            tuple(self._frames), end_timestamp=frame.timestamp
        )
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            self._assessor.reset()
            reason = (
                "NO_TARGET"
                if not self._recent_target_present()
                else "INSUFFICIENT_DATA"
            )
            return self._unknown_payload(
                frame,
                reason=reason,
            )
        result = self._assessor.consume(window)
        if result is None:
            return self._unknown_payload(
                frame,
                reason="ASSESSMENT_WARMUP",
            )
        payload = result.to_dict()
        payload["schema_version"] = self.LIVE_SCHEMA_VERSION
        return payload

    def _recent_target_present(self) -> bool:
        recent = list(self._frames)[-self.presence_recent_frames :]
        return any(
            len(frame.points) >= self.presence_point_count_threshold
            for frame in recent
        )

    def _unknown_payload(self, frame: object, *, reason: str) -> dict[str, object]:
        return {
            "schema_version": self.LIVE_SCHEMA_VERSION,
            "timestamp": frame.timestamp.isoformat(),
            "device_id": frame.device_id,
            "room": frame.room.value,
            "risk_level": RiskLevel.UNKNOWN.value,
            "risk_score": None,
            "sway_risk": None,
            "mobility_risk": None,
            "descent_risk": None,
            "assessment_window_seconds": self.assessment_window_seconds,
            "valid_window_count": 0,
            "observed_duration_seconds": 0.0,
            "unknown_reason": reason,
            "shadow_only": True,
            "alert_suppressed": True,
            "disclaimer": ASSESSMENT_DISCLAIMER,
        }


def _build_parser() -> object:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Fall-risk assessment CLI smoke test (no file processing)."
    )
    parser.add_argument("--window-seconds", type=float, default=60.0)
    return parser


if __name__ == "__main__":  # pragma: no cover
    import json

    args = _build_parser().parse_args()
    a = FallRiskAssessmentV1(assessment_window_seconds=args.window_seconds)
    print(json.dumps({"ok": True, "schema": ASSESSMENT_SCHEMA_VERSION}))
