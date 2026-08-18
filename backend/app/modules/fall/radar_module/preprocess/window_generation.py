from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from radar_module.contracts import FeatureVector


class FeatureWindowGenerator:
    def __init__(
        self,
        *,
        window_size: int = 30,
        feature_version: str = "radar_features_v1",
        feature_names: tuple[str, ...],
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        self.window_size = window_size
        self.feature_version = feature_version
        self.feature_names = tuple(feature_names)
        self._features: deque[NDArray[np.float32]] = deque(maxlen=window_size)
        self._stream_key: tuple[str, str, str] | None = None

    @property
    def current_size(self) -> int:
        return len(self._features)

    def reset(self) -> None:
        self._features.clear()
        self._stream_key = None

    def consume(
        self,
        feature: FeatureVector,
    ) -> NDArray[np.float32] | None:
        if not isinstance(feature, FeatureVector):
            raise TypeError("FeatureWindowGenerator only accepts FeatureVector")
        self._validate_feature_contract(feature)
        stream_key = (
            feature.device_id,
            feature.room.value,
            feature.source_mode.value,
        )
        if self._stream_key is not None and self._stream_key != stream_key:
            self.reset()
        self._stream_key = stream_key
        self._features.append(feature.values.copy())
        if len(self._features) < self.window_size:
            return None
        return np.stack(tuple(self._features), axis=0).astype(
            np.float32,
            copy=False,
        )

    def _validate_feature_contract(self, feature: FeatureVector) -> None:
        if feature.version != self.feature_version:
            raise ValueError(
                f"feature version mismatch: expected {self.feature_version}, "
                f"got {feature.version}"
            )
        if feature.names != self.feature_names:
            raise ValueError("feature name/order mismatch")
