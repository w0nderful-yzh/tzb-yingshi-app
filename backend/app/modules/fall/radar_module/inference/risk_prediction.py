from __future__ import annotations

import numpy as np
import torch

from radar_module.contracts import (
    DEMO_DISCLAIMER,
    FeatureVector,
    HumanState,
    ModelMode,
    RadarRiskResult,
)
from radar_module.model.radar_lstm import LoadedRadarModel
from radar_module.preprocess.window_generation import FeatureWindowGenerator


class RadarRiskPredictor:
    def __init__(
        self,
        loaded_model: LoadedRadarModel,
        *,
        risk_threshold: float = 0.7,
        consecutive_windows: int = 3,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0 < risk_threshold < 1:
            raise ValueError("risk_threshold must be between zero and one")
        if consecutive_windows <= 0:
            raise ValueError("consecutive_windows must be positive")
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.model_mode = loaded_model.model_mode
        self.risk_threshold = risk_threshold
        self.consecutive_windows = consecutive_windows
        self.device = torch.device(device)
        self.window_generator = FeatureWindowGenerator(
            window_size=loaded_model.window_size,
            feature_version=loaded_model.feature_version,
            feature_names=loaded_model.feature_names,
        )
        self._high_risk_count = 0
        self._event_latched = False

    def reset(self) -> None:
        self.window_generator.reset()
        self._high_risk_count = 0
        self._event_latched = False

    def consume(self, feature: FeatureVector) -> RadarRiskResult:
        if not isinstance(feature, FeatureVector):
            raise TypeError("RadarRiskPredictor only accepts FeatureVector")
        if not feature.human_present:
            self.reset()
            return self._result(
                feature,
                human_state=HumanState.NO_PERSON,
                risk_score=0.0,
            )

        window = self.window_generator.consume(feature)
        if window is None:
            self._high_risk_count = 0
            self._event_latched = False
            return self._result(
                feature,
                human_state=HumanState.NORMAL,
                risk_score=0.0,
            )

        risk_score = self._infer_score(window)
        high_risk = risk_score >= self.risk_threshold
        if high_risk:
            self._high_risk_count += 1
        else:
            self._high_risk_count = 0
            self._event_latched = False

        event_triggered = False
        if (
            high_risk
            and self._high_risk_count >= self.consecutive_windows
            and not self._event_latched
        ):
            event_triggered = True
            self._event_latched = True

        return self._result(
            feature,
            human_state=(
                HumanState.FALL_RISK if high_risk else HumanState.NORMAL
            ),
            risk_score=risk_score,
            event_triggered=event_triggered,
        )

    def _infer_score(self, window: np.ndarray) -> float:
        tensor = torch.from_numpy(window).unsqueeze(0).to(
            device=self.device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            risk_logit = self.model(tensor)
            risk_score = torch.sigmoid(risk_logit)
        return float(risk_score.item())

    def _result(
        self,
        feature: FeatureVector,
        *,
        human_state: HumanState,
        risk_score: float,
        event_triggered: bool = False,
    ) -> RadarRiskResult:
        return RadarRiskResult(
            room=feature.room,
            device_id=feature.device_id,
            timestamp=feature.timestamp,
            source_mode=feature.source_mode,
            human_state=human_state,
            risk_score=risk_score,
            model_mode=self.model_mode,
            disclaimer=(
                DEMO_DISCLAIMER
                if self.model_mode is ModelMode.TEST_CHECKPOINT
                else None
            ),
            event_triggered=event_triggered,
        )
