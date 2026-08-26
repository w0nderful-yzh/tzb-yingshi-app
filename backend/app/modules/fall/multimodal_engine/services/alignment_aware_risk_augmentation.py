from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import threading

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    AlignmentAwareRiskAugmentationResult,
    CameraEvidence,
)


@dataclass(frozen=True, slots=True)
class AssociatedEvidenceConfig:
    """Shadow evidence gates, not formal model or alert thresholds."""

    window_seconds: float = 1.2
    minimum_track_samples: int = 2
    minimum_point_count: int = 3
    minimum_track_stability: float = 0.60
    weak_vertical_velocity_mps: float = -0.10
    strong_vertical_velocity_mps: float = -0.35
    weak_height_drop_m: float = -0.05
    strong_height_drop_m: float = -0.20

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("associated evidence window must be positive")
        if self.minimum_track_samples < 2:
            raise ValueError("at least two track samples are required")
        if self.minimum_point_count < 1:
            raise ValueError("minimum point count must be positive")
        if not 0 <= self.minimum_track_stability <= 1:
            raise ValueError("track stability must be within [0, 1]")
        if self.strong_vertical_velocity_mps >= self.weak_vertical_velocity_mps:
            raise ValueError("strong descent velocity must be more negative than weak")
        if self.strong_height_drop_m >= self.weak_height_drop_m:
            raise ValueError("strong height drop must be more negative than weak")


@dataclass(frozen=True, slots=True)
class _TrackObservation:
    timestamp: datetime
    association_state: str
    track_id: int | None
    z_m: float | None
    vertical_velocity_mps: float | None
    horizontal_speed_mps: float | None
    point_count: int
    point_cloud_spread_m: float | None
    association_confidence: float


class AlignmentAwareRiskAugmentation:
    """Annotate the frozen BioSTGCN result with matched TI motion evidence.

    The numeric readout is always exactly ``camera_score``. Radar tracking can
    corroborate, expose conflict or request WATCH in this shadow branch, but it
    cannot create a learned score, modify Fixed Fusion or trigger an alert.
    """

    def __init__(self, config: AssociatedEvidenceConfig | None = None) -> None:
        self.config = config or AssociatedEvidenceConfig()
        self._history: deque[_TrackObservation] = deque(maxlen=256)
        self._last_signature: tuple[object, ...] | None = None
        self._stream_key: tuple[str | None, str | None] | None = None
        self._lock = threading.RLock()

    def apply(
        self,
        camera: CameraEvidence,
        alignment: AlignedPersonEvidence,
    ) -> AlignmentAwareRiskAugmentationResult:
        with self._lock:
            stream_key = (camera.device_id, alignment.radar_config_name)
            if self._stream_key is not None and stream_key != self._stream_key:
                self._reset_unlocked()
            self._stream_key = stream_key
            self._append_unique(camera, alignment)
            reference = alignment.radar_source_timestamp or camera.source_timestamp
            self._prune(reference - timedelta(seconds=self.config.window_seconds))

            samples = [
                item
                for item in self._history
                if alignment.radar_track_id is not None
                and item.track_id == alignment.radar_track_id
                and item.association_state == "MATCHED"
            ]
            stability = self._track_stability(alignment.radar_track_id)
            features = self._motion_features(samples, alignment, stability)
            strength, evidence_reasons = self._motion_strength(
                alignment,
                samples,
                features,
                stability,
            )
            risk_state, evidence_state, decision_reasons = self._camera_led_state(
                camera,
                alignment,
                strength,
            )
            return AlignmentAwareRiskAugmentationResult(
                associated_short_term_fall_score=camera.camera_score,
                associated_risk_state=risk_state,
                associated_evidence_state=evidence_state,
                base_camera_score=camera.camera_score,
                base_camera_state=camera.camera_risk_state,
                radar_motion_evidence_strength=strength,
                association_state=alignment.association_state,
                sync_delta_ms=alignment.sync_delta_ms,
                radar_track_id=alignment.radar_track_id,
                radar_evidence_count=len(samples),
                track_stability=stability,
                radar_motion_features=features,
                reason_codes=list(
                    dict.fromkeys(
                        [
                            "SHADOW_ONLY",
                            "BIOSTGCN_CAMERA_PRIMARY",
                            "RADAR_TRACKING_EVIDENCE_ONLY",
                            *alignment.reason_codes,
                            *evidence_reasons,
                            *decision_reasons,
                        ]
                    )
                ),
            )

    def _append_unique(
        self,
        camera: CameraEvidence,
        alignment: AlignedPersonEvidence,
    ) -> None:
        signature = (
            alignment.radar_frame_number,
            alignment.radar_source_timestamp,
            alignment.association_state,
            alignment.radar_track_id,
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature
        vx, vy, vz = alignment.radar_velocity_xyz_mps
        horizontal_speed = (
            math.hypot(float(vx), float(vy))
            if isinstance(vx, (int, float)) and isinstance(vy, (int, float))
            else None
        )
        z_m = alignment.radar_position_xyz_m[2]
        self._history.append(
            _TrackObservation(
                timestamp=alignment.radar_source_timestamp or camera.source_timestamp,
                association_state=alignment.association_state,
                track_id=alignment.radar_track_id,
                z_m=float(z_m) if isinstance(z_m, (int, float)) else None,
                vertical_velocity_mps=(
                    float(vz) if isinstance(vz, (int, float)) else None
                ),
                horizontal_speed_mps=horizontal_speed,
                point_count=alignment.radar_point_count,
                point_cloud_spread_m=alignment.radar_point_cloud_spread_m,
                association_confidence=alignment.association_confidence,
            )
        )

    def _prune(self, cutoff: datetime) -> None:
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

    def _track_stability(self, track_id: int | None) -> float | None:
        if not self._history or track_id is None:
            return None
        matched = sum(
            item.association_state == "MATCHED" and item.track_id == track_id
            for item in self._history
        )
        return matched / len(self._history)

    @staticmethod
    def _motion_features(
        samples: list[_TrackObservation],
        alignment: AlignedPersonEvidence,
        stability: float | None,
    ) -> dict[str, float | int | bool | str | None]:
        latest = samples[-1] if samples else None
        z_samples = [item for item in samples if item.z_m is not None]
        height_delta = (
            z_samples[-1].z_m - z_samples[0].z_m
            if len(z_samples) >= 2
            else None
        )
        vertical_velocity = latest.vertical_velocity_mps if latest is not None else None
        if vertical_velocity is None and len(z_samples) >= 2:
            elapsed = (z_samples[-1].timestamp - z_samples[0].timestamp).total_seconds()
            if elapsed > 0:
                vertical_velocity = height_delta / elapsed
        spread_samples = [
            item.point_cloud_spread_m
            for item in samples
            if item.point_cloud_spread_m is not None
        ]
        spread_delta = (
            spread_samples[-1] - spread_samples[0]
            if len(spread_samples) >= 2
            else None
        )
        return {
            "height_delta_m": height_delta,
            "vertical_velocity_mps": vertical_velocity,
            "horizontal_speed_mps": (
                latest.horizontal_speed_mps if latest is not None else None
            ),
            "point_cloud_spread_m": (
                latest.point_cloud_spread_m if latest is not None else None
            ),
            "point_cloud_spread_delta_m": spread_delta,
            "point_count": alignment.radar_point_count,
            "track_stability": stability,
            "association_confidence": alignment.association_confidence,
            "radar_config_name": alignment.radar_config_name,
            "sample_count": len(samples),
            "point_cloud_spread_affects_state": False,
        }

    def _motion_strength(
        self,
        alignment: AlignedPersonEvidence,
        samples: list[_TrackObservation],
        features: dict[str, float | int | bool | str | None],
        stability: float | None,
    ) -> tuple[str, list[str]]:
        if (
            alignment.association_state != "MATCHED"
            or not alignment.eligible_for_temporal_association
        ):
            return "UNKNOWN", ["ASSOCIATED_RADAR_EVIDENCE_UNAVAILABLE"]
        if len(samples) < self.config.minimum_track_samples:
            return "UNKNOWN", ["RADAR_TRACK_HISTORY_INSUFFICIENT"]
        if alignment.radar_point_count < self.config.minimum_point_count:
            return "UNKNOWN", ["RADAR_POINT_COUNT_INSUFFICIENT"]
        if stability is None or stability < self.config.minimum_track_stability:
            return "UNKNOWN", ["RADAR_TRACK_UNSTABLE"]

        velocity = features.get("vertical_velocity_mps")
        height_delta = features.get("height_delta_m")
        if not isinstance(velocity, (int, float)) or not isinstance(
            height_delta, (int, float)
        ):
            return "UNKNOWN", ["RADAR_VERTICAL_MOTION_INCOMPLETE"]
        if (
            velocity <= self.config.strong_vertical_velocity_mps
            and height_delta <= self.config.strong_height_drop_m
        ):
            return "STRONG", ["ASSOCIATED_STRONG_DESCENT_EVIDENCE"]
        if (
            velocity <= self.config.weak_vertical_velocity_mps
            and height_delta < 0
        ) or (
            height_delta <= self.config.weak_height_drop_m and velocity < 0
        ):
            return "WEAK", ["ASSOCIATED_WEAK_DESCENT_EVIDENCE"]
        return "NONE", ["NO_ASSOCIATED_DESCENT_EVIDENCE"]

    @staticmethod
    def _camera_led_state(
        camera: CameraEvidence,
        alignment: AlignedPersonEvidence,
        strength: str,
    ) -> tuple[str, str, list[str]]:
        if not camera.available or camera.camera_score is None:
            return "UNKNOWN", "UNKNOWN", ["CAMERA_EVIDENCE_UNAVAILABLE"]

        camera_state = camera.camera_risk_state
        associated = alignment.association_state == "MATCHED"
        if not associated:
            if camera_state == "HIGH":
                return "HIGH", "CAMERA_ONLY_HIGH", ["RADAR_CANNOT_VETO_CAMERA_HIGH"]
            if camera_state == "MEDIUM":
                return "WATCH", "CAMERA_ONLY_WATCH", ["ASSOCIATION_NOT_AVAILABLE"]
            return "UNKNOWN", "NOT_ASSOCIATED", ["ASSOCIATION_NOT_AVAILABLE"]

        if strength == "UNKNOWN":
            if camera_state == "HIGH":
                return "HIGH", "CAMERA_ONLY_HIGH", ["RADAR_CANNOT_VETO_CAMERA_HIGH"]
            if camera_state == "MEDIUM":
                return "WATCH", "CAMERA_ONLY_WATCH", ["RADAR_EVIDENCE_INSUFFICIENT"]
            return "NORMAL", "CAMERA_ONLY_NORMAL", ["RADAR_EVIDENCE_INSUFFICIENT"]

        if camera_state == "HIGH":
            if strength in {"WEAK", "STRONG"}:
                return "HIGH", "CORROBORATED_HIGH", ["CAMERA_RISK_RADAR_DESCENT_CONSISTENT"]
            return "WATCH", "MODALITY_CONFLICT", ["CAMERA_HIGH_WITHOUT_RADAR_DESCENT"]
        if camera_state == "MEDIUM":
            if strength in {"WEAK", "STRONG"}:
                return "WATCH", "CORROBORATED_WATCH", ["CAMERA_WATCH_RADAR_DESCENT_CONSISTENT"]
            return "WATCH", "CAMERA_ONLY_WATCH", ["CAMERA_WATCH_NOT_RADAR_CONFIRMED"]
        if camera_state == "LOW":
            if strength == "STRONG":
                return "WATCH", "RADAR_MOTION_ANOMALY", ["RADAR_CANNOT_ESCALATE_CAMERA_LOW_TO_HIGH"]
            if strength == "WEAK":
                return "NORMAL", "NORMAL_CORROBORATED", ["WEAK_RADAR_MOTION_WITHOUT_CAMERA_RISK"]
            return "NORMAL", "NORMAL_CORROBORATED", ["BOTH_MODALITIES_NO_CURRENT_RISK"]
        return "UNKNOWN", "UNKNOWN", ["CAMERA_STATE_UNKNOWN"]

    def _reset_unlocked(self) -> None:
        self._history.clear()
        self._last_signature = None

    def reset(self) -> None:
        with self._lock:
            self._stream_key = None
            self._reset_unlocked()
