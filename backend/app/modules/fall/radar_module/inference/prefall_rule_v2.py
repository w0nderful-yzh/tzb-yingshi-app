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


RULE_BASELINE_DISCLAIMER = (
    "可解释规则基线，仅用于数据与报警逻辑验证，不代表已训练的跌倒预测模型"
)


class PreFallRuleState(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    IMMINENT = "IMMINENT"
    UNKNOWN = "UNKNOWN"


class FallRiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class PreFallRuleResult:
    timestamp: str
    room: str
    device_id: str
    state: PreFallRuleState
    pre_fall_score: float
    prediction_horizon_seconds: tuple[float, float]
    data_quality: TemporalDataQuality
    components: dict[str, float]
    target_track_id: int | None = None
    model_mode: str = "RULE_BASELINE_V2"
    disclaimer: str = RULE_BASELINE_DISCLAIMER

    @property
    def prediction_state(self) -> PreFallRuleState:
        """Explicit name for the legacy ``state`` prediction field."""

        return self.state

    @property
    def fall_risk_score(self) -> float:
        """Instant umbrella risk; it always covers pre-fall evidence."""

        motion_risk = float(self.components.get("motion_risk", 0.0))
        return float(np.clip(max(self.pre_fall_score, motion_risk), 0.0, 1.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "room": self.room,
            "device_id": self.device_id,
            "state": self.state.value,
            "prediction_state": self.prediction_state.value,
            "pre_fall_score": self.pre_fall_score,
            "fall_risk_score": self.fall_risk_score,
            "prediction_horizon_seconds": list(
                self.prediction_horizon_seconds
            ),
            "data_quality": self.data_quality.value,
            "components": dict(self.components),
            "target_track_id": self.target_track_id,
            "model_mode": self.model_mode,
            "disclaimer": self.disclaimer,
        }


class PreFallRulePredictorV2:
    """Transparent engineering baseline for pre-impact motion trends."""

    def __init__(
        self,
        *,
        watch_threshold: float = 0.45,
        imminent_threshold: float = 0.70,
    ) -> None:
        if not 0 < watch_threshold < imminent_threshold < 1:
            raise ValueError(
                "thresholds must satisfy 0 < watch < imminent < 1"
            )
        self.watch_threshold = watch_threshold
        self.imminent_threshold = imminent_threshold
        self._indices = {
            name: index for index, name in enumerate(FEATURE_NAMES_V2)
        }

    def predict(self, window: TemporalFeatureWindowV2) -> PreFallRuleResult:
        if not isinstance(window, TemporalFeatureWindowV2):
            raise TypeError("rule predictor requires TemporalFeatureWindowV2")
        if (
            window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA
            or int(window.point_present_mask[-3:].sum()) < 2
        ):
            return self._result(
                window,
                state=PreFallRuleState.UNKNOWN,
                score=0.0,
                components={"quality_gate": 0.0, "motion_risk": 0.0},
            )

        latest = window.values[-1]
        lower_z = window.values[:, self._indices["z_p10"]]
        median_z = window.values[:, self._indices["z_p50"]]
        upper_z = window.values[:, self._indices["z_p90"]]
        core_height = (
            upper_z - lower_z
        )
        lower_slope = _robust_slope(
            lower_z[-7:], window.point_present_mask[-7:]
        )
        vertical_slope = _robust_slope(
            median_z[-7:], window.point_present_mask[-7:]
        )
        upper_slope = _robust_slope(
            upper_z[-7:], window.point_present_mask[-7:]
        )
        shape_slope = _robust_slope(
            core_height[-4:], window.point_present_mask[-4:]
        )
        if (
            lower_slope is None
            or vertical_slope is None
            or upper_slope is None
            or shape_slope is None
        ):
            return self._result(
                window,
                state=PreFallRuleState.UNKNOWN,
                score=0.0,
                components={"quality_gate": 0.0, "motion_risk": 0.0},
            )
        drop_06 = -vertical_slope * 0.6
        downward_velocity = -vertical_slope
        max_abs_velocity = float(
            latest[self._indices["max_abs_velocity"]]
        )
        contraction_03 = -shape_slope * 0.3
        vertical_speed = abs(vertical_slope)
        height_change_06 = abs(vertical_slope) * 0.6
        shape_change_03 = abs(shape_slope) * 0.3
        components = {
            "height_drop_0_6s": _ramp(drop_06, 0.08, 0.25),
            "downward_velocity": _ramp(downward_velocity, 0.20, 0.70),
            "doppler_activity": _ramp(max_abs_velocity, 0.20, 0.80),
            "height_contraction_0_3s": _ramp(contraction_03, 0.05, 0.30),
            "vertical_speed": _ramp(vertical_speed, 0.15, 0.80),
            "height_change_0_6s": _ramp(height_change_06, 0.05, 0.30),
            "shape_change_0_3s": _ramp(shape_change_03, 0.04, 0.25),
        }
        score = (
            0.35 * components["height_drop_0_6s"]
            + 0.35 * components["downward_velocity"]
            + 0.15 * components["doppler_activity"]
            + 0.15 * components["height_contraction_0_3s"]
        )
        if window.data_quality is TemporalDataQuality.DEGRADED:
            score *= 0.8
        recent_presence_count = int(window.point_present_mask[-7:].sum())
        recent_present_mask = window.point_present_mask[-7:]
        recent_point_counts = window.values[-7:, self._indices["point_count"]]
        median_point_count = float(np.median(recent_point_counts[recent_present_mask]))
        median_core_height = float(np.median(core_height[-7:][recent_present_mask]))
        downward_step_count = _downward_step_count(
            median_z[-7:], window.point_present_mask[-7:]
        )
        data_continuity = recent_presence_count >= 5
        body_structure = (
            median_point_count >= 4.0 or median_core_height >= 0.25
        )
        coherent_body_descent = (
            lower_slope <= -0.12
            and vertical_slope <= -0.20
            and upper_slope <= -0.12
        )
        sustained_descent = (
            drop_06 >= 0.08
            and downward_velocity >= 0.20
            and downward_step_count >= 2
            and data_continuity
        )
        posture_collapse = contraction_03 >= 0.04 or drop_06 >= 0.18
        risk_relevant_descent = (
            sustained_descent
            and posture_collapse
            and body_structure
            and coherent_body_descent
        )
        strong_fall_evidence = (
            risk_relevant_descent
            and drop_06 >= 0.30
            and downward_velocity >= 0.55
            and downward_step_count >= 3
        )
        if strong_fall_evidence:
            score = max(score, 0.75)
        if not risk_relevant_descent:
            score *= 0.18
        score = float(np.clip(score, 0.0, 1.0))
        motion_risk = (
            0.35 * components["vertical_speed"]
            + 0.30 * components["doppler_activity"]
            + 0.20 * components["height_change_0_6s"]
            + 0.15 * components["shape_change_0_3s"]
        )
        if window.data_quality is TemporalDataQuality.DEGRADED:
            motion_risk *= 0.8
        if strong_fall_evidence:
            motion_risk = max(motion_risk, 0.75)
        if not risk_relevant_descent:
            motion_risk *= 0.18
        components["sustained_descent_gate"] = float(sustained_descent)
        components["posture_collapse_gate"] = float(posture_collapse)
        components["data_continuity_gate"] = float(data_continuity)
        components["body_structure_gate"] = float(body_structure)
        components["coherent_body_descent_gate"] = float(
            coherent_body_descent
        )
        components["recent_point_frames_0_6s"] = float(
            recent_presence_count
        )
        components["downward_steps_0_6s"] = float(downward_step_count)
        components["height_drop_m_0_6s"] = float(drop_06)
        components["lower_z_slope_mps"] = float(lower_slope)
        components["median_z_slope_mps"] = float(vertical_slope)
        components["upper_z_slope_mps"] = float(upper_slope)
        components["median_point_count_0_6s"] = median_point_count
        components["median_core_height_m_0_6s"] = median_core_height
        components["strong_fall_evidence_gate"] = float(
            strong_fall_evidence
        )
        components["motion_risk"] = float(
            np.clip(max(score, motion_risk), 0.0, 1.0)
        )

        if score >= self.imminent_threshold:
            state = PreFallRuleState.IMMINENT
        elif score >= self.watch_threshold:
            state = PreFallRuleState.WATCH
        else:
            state = PreFallRuleState.NORMAL
        return self._result(window, state=state, score=score, components=components)

    def _result(
        self,
        window: TemporalFeatureWindowV2,
        *,
        state: PreFallRuleState,
        score: float,
        components: dict[str, float],
    ) -> PreFallRuleResult:
        return PreFallRuleResult(
            timestamp=window.end_timestamp.isoformat(),
            room=window.room.value,
            device_id=window.device_id,
            state=state,
            pre_fall_score=score,
            prediction_horizon_seconds=(0.2, 1.5),
            data_quality=window.data_quality,
            components=components,
            target_track_id=window.target_track_id,
        )


@dataclass(frozen=True, slots=True)
class ShortTermRoomRiskResultV2:
    timestamp: str
    room: str
    device_id: str
    target_track_id: int | None
    state: PreFallRuleState
    pre_fall_score: float
    confirmed_score: float
    fall_risk_score: float
    fall_risk_score_5s: float
    fall_risk_level: FallRiskLevel
    data_quality: TemporalDataQuality
    confirmation_windows: int
    event_triggered: bool
    model_mode: str
    disclaimer: str = RULE_BASELINE_DISCLAIMER

    @property
    def prediction_state(self) -> PreFallRuleState:
        return self.state

    @property
    def room_risk_score_5s(self) -> float:
        """Backward-compatible alias for ``fall_risk_score_5s``."""

        return self.fall_risk_score_5s

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "room": self.room,
            "device_id": self.device_id,
            "target_track_id": self.target_track_id,
            "state": self.state.value,
            "prediction_state": self.prediction_state.value,
            "pre_fall_score": self.pre_fall_score,
            "confirmed_score": self.confirmed_score,
            "fall_risk_score": self.fall_risk_score,
            "fall_risk_score_5s": self.fall_risk_score_5s,
            "fall_risk_level": self.fall_risk_level.value,
            "room_risk_score_5s": self.room_risk_score_5s,
            "data_quality": self.data_quality.value,
            "confirmation_windows": self.confirmation_windows,
            "event_triggered": self.event_triggered,
            "model_mode": self.model_mode,
            "disclaimer": self.disclaimer,
        }


class PreFallTemporalAggregatorV2:
    """Confirm consecutive radar-only scores and retain five-second room risk.

    A single score spike cannot enter ``IMMINENT``. The confirmed score is the
    minimum of the latest consecutive windows, so every window in the run must
    satisfy the threshold. The five-second value is anonymous room/track-local
    short-term state, not a long-term or person-specific medical risk score.
    """

    def __init__(
        self,
        *,
        confirmation_windows: int = 3,
        history_seconds: float = 5.0,
        watch_enter_threshold: float = 0.45,
        watch_exit_threshold: float = 0.30,
        imminent_enter_threshold: float = 0.70,
        imminent_exit_threshold: float = 0.55,
        max_gap_seconds: float = 0.5,
    ) -> None:
        if confirmation_windows < 2:
            raise ValueError("confirmation_windows must be at least two")
        if history_seconds <= 0:
            raise ValueError("history_seconds must be positive")
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if not (
            0
            < watch_exit_threshold
            < watch_enter_threshold
            < imminent_exit_threshold
            < imminent_enter_threshold
            < 1
        ):
            raise ValueError("temporal thresholds are not ordered correctly")
        self.confirmation_windows = confirmation_windows
        self.history_seconds = history_seconds
        self.watch_enter_threshold = watch_enter_threshold
        self.watch_exit_threshold = watch_exit_threshold
        self.imminent_enter_threshold = imminent_enter_threshold
        self.imminent_exit_threshold = imminent_exit_threshold
        self.max_gap_seconds = max_gap_seconds
        self._recent_scores: deque[float] = deque(maxlen=confirmation_windows)
        self._risk_history: deque[tuple[datetime, float]] = deque()
        self._stream_key: tuple[str, str, int | None] | None = None
        self._last_timestamp: datetime | None = None
        self._state = PreFallRuleState.NORMAL
        self._event_latched = False

    def reset(self) -> None:
        self._recent_scores.clear()
        self._risk_history.clear()
        self._stream_key = None
        self._last_timestamp = None
        self._state = PreFallRuleState.NORMAL
        self._event_latched = False

    def consume(
        self, result: PreFallRuleResult
    ) -> ShortTermRoomRiskResultV2:
        if not isinstance(result, PreFallRuleResult):
            raise TypeError("aggregator requires PreFallRuleResult")
        timestamp = _parse_result_timestamp(result.timestamp)
        stream_key = (result.device_id, result.room, result.target_track_id)
        if self._stream_key is not None and stream_key != self._stream_key:
            self.reset()
        if self._last_timestamp is not None:
            gap_seconds = (timestamp - self._last_timestamp).total_seconds()
            if gap_seconds <= 0:
                raise ValueError("result timestamps must be strictly increasing")
            if gap_seconds > self.max_gap_seconds:
                self.reset()
        self._stream_key = stream_key
        self._last_timestamp = timestamp

        if result.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            self._recent_scores.clear()
            self._risk_history.clear()
            self._state = PreFallRuleState.UNKNOWN
            self._event_latched = False
            return self._build_result(
                result,
                confirmed_score=0.0,
                fall_risk_score_5s=0.0,
                event_triggered=False,
            )

        self._recent_scores.append(float(np.clip(result.pre_fall_score, 0, 1)))
        confirmed_score = (
            min(self._recent_scores)
            if len(self._recent_scores) == self.confirmation_windows
            else 0.0
        )
        self._risk_history.append((timestamp, result.fall_risk_score))
        self._prune_history(timestamp)
        previous_state = self._state
        self._state = self._next_state(confirmed_score)

        event_triggered = False
        if (
            self._state is PreFallRuleState.IMMINENT
            and previous_state is not PreFallRuleState.IMMINENT
            and not self._event_latched
        ):
            event_triggered = True
            self._event_latched = True
        elif self._state in (PreFallRuleState.NORMAL, PreFallRuleState.UNKNOWN):
            self._event_latched = False

        return self._build_result(
            result,
            confirmed_score=confirmed_score,
            fall_risk_score_5s=self._room_risk_score(),
            event_triggered=event_triggered,
        )

    def _next_state(self, confirmed_score: float) -> PreFallRuleState:
        if self._state is PreFallRuleState.IMMINENT:
            if confirmed_score >= self.imminent_exit_threshold:
                return PreFallRuleState.IMMINENT
        if confirmed_score >= self.imminent_enter_threshold:
            return PreFallRuleState.IMMINENT
        if self._state in (PreFallRuleState.WATCH, PreFallRuleState.IMMINENT):
            if confirmed_score >= self.watch_exit_threshold:
                return PreFallRuleState.WATCH
        if confirmed_score >= self.watch_enter_threshold:
            return PreFallRuleState.WATCH
        return PreFallRuleState.NORMAL

    def _prune_history(self, timestamp: datetime) -> None:
        while self._risk_history and (
            timestamp - self._risk_history[0][0]
        ).total_seconds() > self.history_seconds:
            self._risk_history.popleft()

    def _room_risk_score(self) -> float:
        if not self._risk_history:
            return 0.0
        return float(max(score for _, score in self._risk_history))

    def _build_result(
        self,
        source: PreFallRuleResult,
        *,
        confirmed_score: float,
        fall_risk_score_5s: float,
        event_triggered: bool,
    ) -> ShortTermRoomRiskResultV2:
        return ShortTermRoomRiskResultV2(
            timestamp=source.timestamp,
            room=source.room,
            device_id=source.device_id,
            target_track_id=source.target_track_id,
            state=self._state,
            pre_fall_score=source.pre_fall_score,
            confirmed_score=confirmed_score,
            fall_risk_score=(
                0.0
                if source.data_quality
                is TemporalDataQuality.INSUFFICIENT_DATA
                else source.fall_risk_score
            ),
            fall_risk_score_5s=(
                0.0
                if source.data_quality
                is TemporalDataQuality.INSUFFICIENT_DATA
                else max(fall_risk_score_5s, source.pre_fall_score)
            ),
            fall_risk_level=(
                FallRiskLevel.UNKNOWN
                if source.data_quality
                is TemporalDataQuality.INSUFFICIENT_DATA
                else _risk_level(fall_risk_score_5s)
            ),
            data_quality=source.data_quality,
            confirmation_windows=self.confirmation_windows,
            event_triggered=event_triggered,
            model_mode=source.model_mode,
        )


def _parse_result_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("result timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("result timestamp needs timezone")
    return parsed


def _ramp(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("upper ramp bound must exceed lower bound")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _robust_slope(
    values: np.ndarray,
    point_present_mask: np.ndarray,
    *,
    period_seconds: float = 0.1,
) -> float | None:
    """Return a median pairwise slope, resistant to one-frame point jumps."""

    valid_indices = np.flatnonzero(
        point_present_mask & np.isfinite(values)
    )
    if len(valid_indices) < 3:
        return None
    slopes = [
        float(values[right] - values[left])
        / float((right - left) * period_seconds)
        for offset, left in enumerate(valid_indices[:-1])
        for right in valid_indices[offset + 1 :]
    ]
    return float(np.median(slopes)) if slopes else None


def _downward_step_count(
    values: np.ndarray,
    point_present_mask: np.ndarray,
    *,
    minimum_drop_per_0_1s: float = 0.025,
) -> int:
    valid_indices = np.flatnonzero(
        point_present_mask & np.isfinite(values)
    )
    count = 0
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        gap = int(right - left)
        if gap > 2:
            continue
        required_drop = minimum_drop_per_0_1s * gap
        if float(values[left] - values[right]) >= required_drop:
            count += 1
    return count


def _risk_level(score: float) -> FallRiskLevel:
    if score >= 0.60:
        return FallRiskLevel.HIGH
    if score >= 0.30:
        return FallRiskLevel.MODERATE
    return FallRiskLevel.LOW
