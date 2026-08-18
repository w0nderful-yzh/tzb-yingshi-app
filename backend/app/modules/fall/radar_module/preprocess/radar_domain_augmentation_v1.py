from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from radar_module.contracts import RadarFrame, RadarPoint
from radar_module.preprocess.temporal_features_v2 import (
    RadarTemporalFeatureExtractorV2,
)


@dataclass(frozen=True, slots=True)
class RadarDomainAugmentationConfigV1:
    """Light point-domain perturbations used only while exporting train data."""

    seed: int = 20260809
    clean_recording_probability: float = 0.30
    minimum_keep_ratio: float = 0.70
    maximum_keep_ratio: float = 0.95
    minimum_points_after_dropout: int = 4
    minimum_xyz_jitter_std_m: float = 0.005
    maximum_xyz_jitter_std_m: float = 0.020
    xyz_jitter_clip_m: float = 0.050
    minimum_velocity_noise_std_mps: float = 0.010
    maximum_velocity_noise_std_mps: float = 0.050
    velocity_noise_clip_mps: float = 0.150

    def __post_init__(self) -> None:
        if not 0.0 <= self.clean_recording_probability <= 1.0:
            raise ValueError("clean_recording_probability must be in [0, 1]")
        if not 0.0 < self.minimum_keep_ratio <= self.maximum_keep_ratio <= 1.0:
            raise ValueError("point keep ratios must be in (0, 1]")
        if self.minimum_points_after_dropout <= 0:
            raise ValueError("minimum_points_after_dropout must be positive")
        if not 0.0 <= self.minimum_xyz_jitter_std_m <= self.maximum_xyz_jitter_std_m:
            raise ValueError("xyz jitter range is invalid")
        if self.xyz_jitter_clip_m <= 0.0:
            raise ValueError("xyz_jitter_clip_m must be positive")
        if not (
            0.0
            <= self.minimum_velocity_noise_std_mps
            <= self.maximum_velocity_noise_std_mps
        ):
            raise ValueError("velocity noise range is invalid")
        if self.velocity_noise_clip_mps <= 0.0:
            raise ValueError("velocity_noise_clip_mps must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecordingAugmentationPlanV1:
    clean: bool
    point_keep_ratio: float
    xyz_jitter_std_m: float
    velocity_noise_std_mps: float


class RadarDomainAugmentedFeatureExtractorV1(RadarTemporalFeatureExtractorV2):
    """Deterministically perturb raw points before calculating v2 features.

    The recording plan is keyed by ``device_id``.  A frame is keyed by its
    timestamp, so overlapping windows receive identical augmented points.
    Validation and test rows are later taken from their clean exports; this
    extractor is only used to create a parallel training candidate artifact.
    """

    augmentation_version = "radar_point_domain_augmentation_v1"

    def __init__(
        self,
        config: RadarDomainAugmentationConfigV1 | None = None,
        **extractor_kwargs: float,
    ) -> None:
        super().__init__(**extractor_kwargs)
        self.config = config or RadarDomainAugmentationConfigV1()
        self._plans: dict[str, RecordingAugmentationPlanV1] = {}
        self._cached_device_id: str | None = None
        self._frame_cache: dict[str, RadarFrame] = {}

    def recording_plan(self, device_id: str) -> RecordingAugmentationPlanV1:
        if device_id not in self._plans:
            rng = np.random.default_rng(
                _stable_seed(self.config.seed, "recording", device_id)
            )
            clean = bool(rng.random() < self.config.clean_recording_probability)
            if clean:
                plan = RecordingAugmentationPlanV1(
                    clean=True,
                    point_keep_ratio=1.0,
                    xyz_jitter_std_m=0.0,
                    velocity_noise_std_mps=0.0,
                )
            else:
                plan = RecordingAugmentationPlanV1(
                    clean=False,
                    point_keep_ratio=float(
                        rng.uniform(
                            self.config.minimum_keep_ratio,
                            self.config.maximum_keep_ratio,
                        )
                    ),
                    xyz_jitter_std_m=float(
                        rng.uniform(
                            self.config.minimum_xyz_jitter_std_m,
                            self.config.maximum_xyz_jitter_std_m,
                        )
                    ),
                    velocity_noise_std_mps=float(
                        rng.uniform(
                            self.config.minimum_velocity_noise_std_mps,
                            self.config.maximum_velocity_noise_std_mps,
                        )
                    ),
                )
            self._plans[device_id] = plan
        return self._plans[device_id]

    def transform(self, frames, **kwargs):  # type: ignore[override]
        ordered = tuple(frames)
        if not ordered:
            return super().transform(ordered, **kwargs)
        plan = self.recording_plan(ordered[0].device_id)
        if plan.clean:
            return super().transform(ordered, **kwargs)
        if self._cached_device_id != ordered[0].device_id:
            self._cached_device_id = ordered[0].device_id
            self._frame_cache.clear()
        augmented_frames: list[RadarFrame] = []
        for frame in ordered:
            frame_key = frame.timestamp.isoformat()
            augmented_frame = self._frame_cache.get(frame_key)
            if augmented_frame is None:
                augmented_frame = self._augment_frame(frame, plan)
                self._frame_cache[frame_key] = augmented_frame
            augmented_frames.append(augmented_frame)
        augmented = tuple(augmented_frames)
        return super().transform(augmented, **kwargs)

    def augmentation_spec(self) -> dict[str, object]:
        return {
            "version": self.augmentation_version,
            **self.config.to_dict(),
            "determinism": "device_id recording plan + timestamp frame seed",
            "frame_masks_modified": False,
        }

    def _augment_frame(
        self,
        frame: RadarFrame,
        plan: RecordingAugmentationPlanV1,
    ) -> RadarFrame:
        points = frame.points
        if not points:
            return frame
        rng = np.random.default_rng(
            _stable_seed(
                self.config.seed,
                "frame",
                frame.device_id,
                frame.timestamp.isoformat(),
            )
        )
        selected = np.arange(len(points), dtype=np.int64)
        if len(points) >= self.config.minimum_points_after_dropout:
            keep_count = max(
                self.config.minimum_points_after_dropout,
                int(round(len(points) * plan.point_keep_ratio)),
            )
            keep_count = min(keep_count, len(points))
            if keep_count < len(points):
                selected = np.sort(
                    rng.choice(len(points), size=keep_count, replace=False)
                )

        result: list[RadarPoint] = []
        for point_index in selected:
            point = points[int(point_index)]
            xyz_noise = np.clip(
                rng.normal(0.0, plan.xyz_jitter_std_m, size=3),
                -self.config.xyz_jitter_clip_m,
                self.config.xyz_jitter_clip_m,
            )
            velocity_noise = float(
                np.clip(
                    rng.normal(0.0, plan.velocity_noise_std_mps),
                    -self.config.velocity_noise_clip_mps,
                    self.config.velocity_noise_clip_mps,
                )
            )
            result.append(
                RadarPoint(
                    x=float(point.x + xyz_noise[0]),
                    y=float(point.y + xyz_noise[1]),
                    z=float(point.z + xyz_noise[2]),
                    velocity=float(point.velocity + velocity_noise),
                    snr=point.snr,
                    track_id=point.track_id,
                )
            )
        return RadarFrame(
            timestamp=frame.timestamp,
            device_id=frame.device_id,
            room=frame.room,
            source_mode=frame.source_mode,
            points=tuple(result),
        )


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)
