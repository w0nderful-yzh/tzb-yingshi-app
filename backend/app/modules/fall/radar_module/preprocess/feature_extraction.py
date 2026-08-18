from __future__ import annotations

import numpy as np

from radar_module.contracts import FeatureVector, RadarFrame


class RadarFeatureExtractor:
    """MVP V1八维特征；仅用于风险推理链路验证。"""

    feature_version = "radar_features_v1"
    feature_names = (
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "height_range",
        "mean_velocity",
        "max_abs_velocity",
        "velocity_std",
        "point_count",
    )

    def extract(self, frame: RadarFrame) -> FeatureVector:
        if not isinstance(frame, RadarFrame):
            raise TypeError("RadarFeatureExtractor only accepts RadarFrame")

        if not frame.points:
            values = np.zeros(len(self.feature_names), dtype=np.float32)
            return self._build_feature(frame, values, human_present=False)

        coordinates = np.asarray(
            [(point.x, point.y, point.z) for point in frame.points],
            dtype=np.float32,
        )
        velocities = np.asarray(
            [point.velocity for point in frame.points],
            dtype=np.float32,
        )
        centroid = coordinates.mean(axis=0)
        values = np.asarray(
            [
                centroid[0],
                centroid[1],
                centroid[2],
                coordinates[:, 2].max() - coordinates[:, 2].min(),
                velocities.mean(),
                np.abs(velocities).max(),
                velocities.std(),
                float(len(frame.points)),
            ],
            dtype=np.float32,
        )
        return self._build_feature(frame, values, human_present=True)

    def _build_feature(
        self,
        frame: RadarFrame,
        values: np.ndarray,
        *,
        human_present: bool,
    ) -> FeatureVector:
        return FeatureVector(
            timestamp=frame.timestamp,
            device_id=frame.device_id,
            room=frame.room,
            source_mode=frame.source_mode,
            human_present=human_present,
            version=self.feature_version,
            names=self.feature_names,
            values=values,
        )
