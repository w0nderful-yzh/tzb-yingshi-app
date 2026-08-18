from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.contracts import RadarFrame, SourceMode
from radar_module.inference.prefall_rule_v2 import PreFallRulePredictorV2
from radar_module.model.radar_lstm import RadarLSTM
from radar_module.model.research_training_v2 import (
    RESEARCH_MODEL_MODE,
    RESEARCH_MODEL_VERSION,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
WINDOW_SIZE_V2,
)


TRACK_CONFIRMATION_SECONDS = 0.4
TRACK_CONFIRMATION_FRAMES = 3
TRACK_LOSS_SECONDS = 0.5


RESEARCH_LIVE_DISCLAIMER = (
    "雷达单模态跌倒预测仍在验证；动作风险可用于现场风险记录，"
    "尚未通过本机雷达真实跌倒验证"
)


@dataclass(frozen=True, slots=True)
class ResearchLiveResultV2:
    timestamp: str
    prediction_state: str
    pre_fall_score: float
    fall_risk_score: float
    fall_risk_score_5s: float
    fall_risk_level: str
    action_risk_event_triggered: bool
    data_quality: str
    threshold: float
    prediction_horizon_seconds: tuple[float, float]
    positive_anchor: str
    rule_components: dict[str, float] = field(default_factory=dict)
    model_mode: str = RESEARCH_MODEL_MODE
    shadow_only: bool = True
    alert_suppressed: bool = True
    disclaimer: str = RESEARCH_LIVE_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["prediction_horizon_seconds"] = list(
            self.prediction_horizon_seconds
        )
        return payload


class RadarResearchLivePredictorV2:
    """Run a research v2 checkpoint on decoded frames without alert routing."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        confirmation_windows: int = 3,
        prefer_track_filter: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        if confirmation_windows < 2:
            raise ValueError("confirmation_windows must be at least two")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = _load_checkpoint(self.checkpoint_path, device)
        self.device = torch.device(device)
        self.model = RadarLSTM(
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
        horizon = tuple(
            float(value)
            for value in checkpoint.get("prediction_horizon_seconds", (0.1, 0.6))
        )
        if len(horizon) != 2 or not 0 < horizon[0] <= horizon[1]:
            raise ValueError("research checkpoint horizon is invalid")
        self.prediction_horizon_seconds = horizon
        self.positive_anchor = str(
            checkpoint.get("positive_anchor", "descent_onset")
        )
        self.confirmation_windows = confirmation_windows
        self.prefer_track_filter = prefer_track_filter
        self.extractor = RadarTemporalFeatureExtractorV2()
        self.rule_predictor = PreFallRulePredictorV2()
        self._frames: deque[RadarFrame] = deque()
        self._stream_key: tuple[str, object, object] | None = None
        self._last_frame_timestamp: datetime | None = None
        self._last_evaluation_timestamp: datetime | None = None
        self._high_run = 0
        self._risk_history: deque[tuple[datetime, float]] = deque()
        self._action_risk_latched = False
        self._active_track_id: int | None = None
        self._candidate_track_id: int | None = None
        self._candidate_since: datetime | None = None
        self._candidate_frame_count = 0
        self._active_track_missing_since: datetime | None = None
        self._use_track_filter = False

    def reset(self) -> None:
        self._frames.clear()
        self._stream_key = None
        self._last_frame_timestamp = None
        self._last_evaluation_timestamp = None
        self._high_run = 0
        self._risk_history.clear()
        self._action_risk_latched = False
        self._reset_tracking()

    def consume(self, frame: RadarFrame) -> ResearchLiveResultV2 | None:
        if not isinstance(frame, RadarFrame):
            raise TypeError("research live predictor requires RadarFrame")
        stream_key = (frame.device_id, frame.room, frame.source_mode)
        if self._stream_key is not None and stream_key != self._stream_key:
            self.reset()
        if self._last_frame_timestamp is not None:
            gap = (frame.timestamp - self._last_frame_timestamp).total_seconds()
            if gap <= 0:
                raise ValueError("radar frame timestamps must be strictly increasing")
            if gap > 1.0:
                self.reset()
        self._stream_key = stream_key
        self._last_frame_timestamp = frame.timestamp

        target_track_id: int | None = None
        if (
            frame.source_mode is SourceMode.REAL
            and self.prefer_track_filter
        ):
            target_track_id = self._update_tracking(frame)
            if target_track_id is None and self._use_track_filter:
                return self._tracking_unknown(frame.timestamp)

        self._frames.append(frame)
        while self._frames and (
            frame.timestamp - self._frames[0].timestamp
        ).total_seconds() > 2.2:
            self._frames.popleft()

        if self._last_evaluation_timestamp is not None and (
            frame.timestamp - self._last_evaluation_timestamp
        ).total_seconds() < 0.095:
            return None
        if (
            frame.timestamp - self._frames[0].timestamp
        ).total_seconds() < (WINDOW_SIZE_V2 - 1) / 10.0:
            return None
        self._last_evaluation_timestamp = frame.timestamp
        window = self.extractor.transform(
            tuple(self._frames),
            end_timestamp=frame.timestamp,
            target_track_id=target_track_id,
        )
        rule = self.rule_predictor.predict(window)
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            self._high_run = 0
            self._risk_history.clear()
            return self._result(
                frame.timestamp,
                prediction_state="UNKNOWN",
                pre_fall_score=0.0,
                fall_risk_score=0.0,
                fall_risk_score_5s=0.0,
                action_risk_event_triggered=False,
                data_quality=window.data_quality,
                rule_components=rule.components,
            )

        values = np.asarray(window.values, dtype=np.float32)
        normalized = (values - self.mean[None, :]) / self.std[None, :]
        with torch.inference_mode():
            score = float(
                torch.sigmoid(
                    self.model(
                        torch.from_numpy(normalized)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                ).item()
            )
        if rule.components.get("body_structure_gate", 0.0) < 0.5:
            score *= 0.2
        is_high = score >= self.threshold
        self._high_run = self._high_run + 1 if is_high else 0
        prediction_state = (
            "IMMINENT"
            if self._high_run >= self.confirmation_windows
            else "WATCH"
            if is_high
            else "NORMAL"
        )
        # Prediction and action risk are separate outputs.  The learned score
        # is only the pre-fall prediction score; the interpretable rule owns
        # the current-action risk score and its event threshold.
        fall_risk_score = rule.fall_risk_score
        self._risk_history.append((frame.timestamp, fall_risk_score))
        while self._risk_history and (
            frame.timestamp - self._risk_history[0][0]
        ).total_seconds() > 5.0:
            self._risk_history.popleft()
        fall_risk_score_5s = max(value for _, value in self._risk_history)
        action_risk_event_triggered = False
        if fall_risk_score_5s >= 0.30:
            if not self._action_risk_latched:
                action_risk_event_triggered = True
                self._action_risk_latched = True
        else:
            self._action_risk_latched = False
        return self._result(
            frame.timestamp,
            prediction_state=prediction_state,
            pre_fall_score=score,
            fall_risk_score=fall_risk_score,
            fall_risk_score_5s=fall_risk_score_5s,
            action_risk_event_triggered=action_risk_event_triggered,
            data_quality=window.data_quality,
            rule_components=rule.components,
        )

    def _update_tracking(self, frame: RadarFrame) -> int | None:
        counts = Counter(
            point.track_id
            for point in frame.points
            if point.track_id is not None
        )
        present_ids = set(counts)
        if self._active_track_id is not None:
            if self._active_track_id in present_ids:
                self._active_track_missing_since = None
                return self._active_track_id
            if self._active_track_missing_since is None:
                self._active_track_missing_since = frame.timestamp
                self._clear_decision_state()
            missing_seconds = (
                frame.timestamp - self._active_track_missing_since
            ).total_seconds()
            if missing_seconds < TRACK_LOSS_SECONDS:
                return None
            self._drop_active_track()

        if not counts:
            self._clear_candidate()
            return None
        dominant_track_id = min(
            counts,
            key=lambda track_id: (-counts[track_id], track_id),
        )
        if dominant_track_id != self._candidate_track_id:
            self._candidate_track_id = dominant_track_id
            self._candidate_since = frame.timestamp
            self._candidate_frame_count = 1
            return None
        self._candidate_frame_count += 1
        assert self._candidate_since is not None
        candidate_seconds = (
            frame.timestamp - self._candidate_since
        ).total_seconds()
        if (
            candidate_seconds < TRACK_CONFIRMATION_SECONDS
            or self._candidate_frame_count < TRACK_CONFIRMATION_FRAMES
        ):
            return None

        self._active_track_id = dominant_track_id
        self._use_track_filter = True
        self._active_track_missing_since = None
        self._clear_candidate()
        # Do not reuse frames from before the target was stable.  A newly
        # assigned or recycled tracker ID otherwise looks like a height jump.
        self._clear_inference_window()
        return self._active_track_id

    def _tracking_unknown(self, timestamp: datetime) -> ResearchLiveResultV2 | None:
        if self._last_evaluation_timestamp is not None and (
            timestamp - self._last_evaluation_timestamp
        ).total_seconds() < 0.095:
            return None
        self._last_evaluation_timestamp = timestamp
        return self._result(
            timestamp,
            prediction_state="UNKNOWN",
            pre_fall_score=0.0,
            fall_risk_score=0.0,
            fall_risk_score_5s=0.0,
            action_risk_event_triggered=False,
            data_quality=TemporalDataQuality.INSUFFICIENT_DATA,
        )

    def _drop_active_track(self) -> None:
        self._active_track_id = None
        self._active_track_missing_since = None
        self._clear_candidate()
        self._clear_inference_window()

    def _clear_inference_window(self) -> None:
        self._frames.clear()
        self._last_evaluation_timestamp = None
        self._clear_decision_state()

    def _clear_decision_state(self) -> None:
        self._high_run = 0
        self._risk_history.clear()
        self._action_risk_latched = False

    def _reset_tracking(self) -> None:
        self._active_track_id = None
        self._active_track_missing_since = None
        self._use_track_filter = False
        self._clear_candidate()

    def _clear_candidate(self) -> None:
        self._candidate_track_id = None
        self._candidate_since = None
        self._candidate_frame_count = 0

    def _result(
        self,
        timestamp: datetime,
        *,
        prediction_state: str,
        pre_fall_score: float,
        fall_risk_score: float,
        fall_risk_score_5s: float,
        action_risk_event_triggered: bool,
        data_quality: TemporalDataQuality,
        rule_components: dict[str, float] | None = None,
    ) -> ResearchLiveResultV2:
        return ResearchLiveResultV2(
            timestamp=timestamp.isoformat(),
            prediction_state=prediction_state,
            pre_fall_score=float(np.clip(pre_fall_score, 0.0, 1.0)),
            fall_risk_score=float(np.clip(fall_risk_score, 0.0, 1.0)),
            fall_risk_score_5s=float(
                np.clip(fall_risk_score_5s, 0.0, 1.0)
            ),
            fall_risk_level=_risk_level(fall_risk_score_5s),
            action_risk_event_triggered=action_risk_event_triggered,
            data_quality=data_quality.value,
            threshold=self.threshold,
            prediction_horizon_seconds=self.prediction_horizon_seconds,
            positive_anchor=self.positive_anchor,
            rule_components=dict(rule_components or {}),
        )


def _load_checkpoint(
    path: Path, device: str | torch.device
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"research checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("research checkpoint root must be a mapping")
    if checkpoint.get("model_version") != RESEARCH_MODEL_VERSION:
        raise ValueError("unsupported research checkpoint")
    if checkpoint.get("model_mode") != RESEARCH_MODEL_MODE:
        raise ValueError("checkpoint is not research weak supervision")
    if bool(checkpoint.get("deployment_eligible", True)):
        raise ValueError("live shadow requires a non-deployable checkpoint")
    if checkpoint.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("research checkpoint feature version is incompatible")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError("research checkpoint feature names/order are incompatible")
    if int(checkpoint.get("window_size", -1)) != WINDOW_SIZE_V2:
        raise ValueError("research checkpoint window size is incompatible")
    return checkpoint


def _risk_level(score: float) -> str:
    if score >= 0.60:
        return "HIGH"
    if score >= 0.30:
        return "MODERATE"
    return "LOW"
