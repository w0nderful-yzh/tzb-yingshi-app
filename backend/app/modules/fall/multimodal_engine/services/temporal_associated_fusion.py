from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    CameraEvidence,
    FusionResult,
    RadarEvidence,
    TemporalAssociatedFusionResult,
)


@dataclass(frozen=True, slots=True)
class TemporalAssociationConfig:
    """Experimental-only continuity controls; these are not model thresholds."""

    window_seconds: float = 2.0
    confirmation_windows: int = 2
    minimum_quality: float = 0.25

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("temporal association window must be positive")
        if self.confirmation_windows < 1:
            raise ValueError("temporal confirmations must be positive")
        if not 0 <= self.minimum_quality <= 1:
            raise ValueError("minimum quality must be within [0, 1]")


class TemporalAssociatedFusion:
    """Conservative shadow fusion over deduplicated, short-lived evidence.

    It deliberately does not retain a maximum score. The buffer stores each
    unique source window and checks continuity, temporal order, observable
    motion direction and the limited association metadata currently available.
    """

    def __init__(self, config: TemporalAssociationConfig | None = None) -> None:
        self.config = config or TemporalAssociationConfig()
        self._camera: deque[CameraEvidence] = deque(maxlen=128)
        self._radar: deque[RadarEvidence] = deque(maxlen=128)
        self._last_camera_timestamp: datetime | None = None
        self._last_radar_timestamp: datetime | None = None
        self._stream_key: tuple[str | None, str | None] | None = None
        self._lock = threading.RLock()

    def apply(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        fixed: FusionResult,
        *,
        alignment: AlignedPersonEvidence | None = None,
    ) -> TemporalAssociatedFusionResult:
        with self._lock:
            stream_key = (radar.room, radar.device_id or camera.device_id)
            if self._stream_key is not None and stream_key != self._stream_key:
                self._reset_unlocked()
            self._stream_key = stream_key
            self._append_unique(camera, radar)
            reference = max(camera.source_timestamp, radar.source_timestamp)
            cutoff = reference - timedelta(seconds=self.config.window_seconds)
            self._prune(cutoff)

            camera_recent = list(self._camera)
            radar_recent = list(self._radar)
            association, association_reasons = self._association(
                camera,
                radar,
                alignment,
            )
            camera_continuous = self._tail_is_continuous_camera_risk(camera_recent)
            radar_continuous = self._tail_is_continuous_radar_risk(radar_recent)
            relation = self._temporal_relation(camera_recent, radar_recent)
            causal = relation in {"ALIGNED", "RADAR_THEN_CAMERA"}

            reasons = list(association_reasons)
            degraded_mode = "NONE"
            if not camera.available and not radar.available:
                state = "UNKNOWN"
                degraded_mode = "BOTH_UNAVAILABLE"
                reasons.extend(["CAMERA_UNAVAILABLE", "RADAR_UNAVAILABLE"])
            elif not camera.available:
                state = "WATCH"
                degraded_mode = "RADAR_ONLY"
                reasons.append("CAMERA_UNAVAILABLE")
            elif not radar.available:
                state = "WATCH"
                degraded_mode = "CAMERA_ONLY"
                reasons.append("RADAR_UNAVAILABLE")
            elif fixed.sync_delta_seconds is None or (
                fixed.sync_delta_seconds > self.config.window_seconds
            ):
                state = "WATCH"
                degraded_mode = "OUT_OF_SYNC"
                reasons.append("TEMPORAL_WINDOW_EXCEEDED")
            elif min(camera.camera_quality, radar.radar_quality) < self.config.minimum_quality:
                state = "WATCH"
                degraded_mode = "LOW_QUALITY"
                reasons.append("MODALITY_QUALITY_BELOW_TEMPORAL_MINIMUM")
            elif association == "CONFLICT" and (
                camera_continuous
                or radar_continuous
                or self._has_current_weak_risk(camera_recent, radar_recent)
            ):
                state = "WATCH"
                degraded_mode = "MODALITY_CONFLICT"
                reasons.append("TARGET_ASSOCIATION_CONFLICT")
            elif association == "CONFLICT":
                # An identity/geometry conflict is uncertainty, not positive
                # fall evidence. Preserve UNKNOWN instead of inventing WATCH.
                state = "UNKNOWN"
                degraded_mode = "MODALITY_CONFLICT"
                reasons.extend(
                    ["TARGET_ASSOCIATION_CONFLICT", "NO_RISK_EVIDENCE_TO_ESCALATE"]
                )
            elif camera_continuous and radar_continuous and causal and association == "MATCHED":
                state = "HIGH"
                reasons.append("CONTINUOUS_CAUSAL_RISK_EVIDENCE")
            elif camera_continuous or radar_continuous or self._has_current_weak_risk(
                camera_recent, radar_recent
            ):
                state = "WATCH"
                reasons.append("RISK_EVIDENCE_NOT_YET_CAUSALLY_CONFIRMED")
            else:
                state = "NORMAL"
                reasons.append("NO_CONTINUOUS_RISK_SEQUENCE")

            if association != "MATCHED":
                reasons.append("NO_VERIFIED_CROSS_MODAL_TARGET_ID")
            snapshot = self._radar_snapshot(radar)
            return TemporalAssociatedFusionResult(
                # Keep the latest fixed 0.6/0.4 score as a comparable readout;
                # temporal logic only determines whether that evidence is valid.
                fusion_score=fixed.raw_fusion_score,
                fusion_state=state,
                window_seconds=self.config.window_seconds,
                camera_evidence_count=len(camera_recent),
                radar_evidence_count=len(radar_recent),
                continuous_camera_risk=camera_continuous,
                continuous_radar_risk=radar_continuous,
                target_association=association,
                alignment_state=(
                    alignment.association_state
                    if alignment is not None
                    else "CALIBRATION_INVALID"
                ),
                camera_target_id=(
                    str(alignment.camera_person_id)
                    if alignment is not None and alignment.camera_person_id is not None
                    else None
                ),
                radar_target_id=(
                    str(alignment.radar_track_id)
                    if alignment is not None and alignment.radar_track_id is not None
                    else None
                ),
                temporal_relation=relation,
                causal_consistency=causal,
                sync_delta_ms=fixed.sync_delta_ms,
                degraded_mode=degraded_mode,
                reason_codes=sorted(set(reasons)),
                radar_evidence_snapshot=snapshot,
            )

    def _append_unique(self, camera: CameraEvidence, radar: RadarEvidence) -> None:
        if camera.source_timestamp != self._last_camera_timestamp:
            self._camera.append(camera.model_copy(deep=True))
            self._last_camera_timestamp = camera.source_timestamp
        if radar.source_timestamp != self._last_radar_timestamp:
            self._radar.append(radar.model_copy(deep=True))
            self._last_radar_timestamp = radar.source_timestamp

    def _prune(self, cutoff: datetime) -> None:
        while self._camera and self._camera[0].source_timestamp < cutoff:
            self._camera.popleft()
        while self._radar and self._radar[0].source_timestamp < cutoff:
            self._radar.popleft()

    def _tail_is_continuous_camera_risk(self, samples: list[CameraEvidence]) -> bool:
        return self._tail_count(
            [item.available and item.camera_risk_state == "HIGH" for item in samples]
        ) >= self.config.confirmation_windows

    def _tail_is_continuous_radar_risk(self, samples: list[RadarEvidence]) -> bool:
        return self._tail_count(
            [
                item.available
                and item.radar_risk_state in {"IMMINENT", "CONFIRMED"}
                and self._descending(item)
                for item in samples
            ]
        ) >= self.config.confirmation_windows

    @staticmethod
    def _tail_count(values: list[bool]) -> int:
        count = 0
        for value in reversed(values):
            if not value:
                break
            count += 1
        return count

    @staticmethod
    def _descending(radar: RadarEvidence) -> bool:
        feature = radar.radar_feature if isinstance(radar.radar_feature, dict) else {}
        velocity = feature.get("vertical_velocity")
        height_delta = feature.get("height_delta")
        return bool(
            isinstance(velocity, (int, float))
            and isinstance(height_delta, (int, float))
            and velocity < 0
            and height_delta < 0
        )

    @staticmethod
    def _has_current_weak_risk(
        camera: list[CameraEvidence], radar: list[RadarEvidence]
    ) -> bool:
        # Weak evidence is not held across the whole temporal buffer. Retaining
        # any recent weak state would be a disguised peak/state hold and would
        # inflate WATCH after the motion has ended.
        latest_camera = camera[-1] if camera else None
        latest_radar = radar[-1] if radar else None
        camera_risk = bool(
            latest_camera
            and latest_camera.available
            and latest_camera.camera_risk_state in {"MEDIUM", "HIGH"}
        )
        radar_risk = bool(
            latest_radar
            and latest_radar.available
            and latest_radar.radar_risk_state in {"WATCH", "IMMINENT", "CONFIRMED"}
        )
        return camera_risk or radar_risk

    @staticmethod
    def _association(
        camera: CameraEvidence,
        radar: RadarEvidence,
        alignment: AlignedPersonEvidence | None = None,
    ) -> tuple[str, list[str]]:
        if alignment is not None:
            reasons = list(alignment.reason_codes)
            if alignment.association_state == "MATCHED":
                return "MATCHED", reasons
            if alignment.association_state in {"TRACK_CONFLICT", "MULTIPLE_CANDIDATES"}:
                return "CONFLICT", reasons
            return "UNKNOWN", reasons
        camera_feature = camera.camera_feature if isinstance(camera.camera_feature, dict) else {}
        radar_feature = radar.radar_feature if isinstance(radar.radar_feature, dict) else {}
        camera_present = camera_feature.get("target_present") is True
        point_count = radar_feature.get("point_count")
        radar_present = isinstance(point_count, (int, float)) and point_count > 0
        if camera_present and radar_present:
            return "SINGLE_TARGET_ASSUMED", ["SINGLE_TARGET_SCENE_ASSUMPTION_ONLY"]
        if camera.available and radar.available and camera_present != radar_present:
            return "CONFLICT", ["CROSS_MODAL_TARGET_PRESENCE_CONFLICT"]
        return "UNKNOWN", ["TARGET_ASSOCIATION_METADATA_UNAVAILABLE"]

    @classmethod
    def _temporal_relation(
        cls, camera: list[CameraEvidence], radar: list[RadarEvidence]
    ) -> str:
        camera_risk = [
            item.source_timestamp
            for item in camera
            if item.available and item.camera_risk_state == "HIGH"
        ]
        radar_risk = [
            item.source_timestamp
            for item in radar
            if item.available
            and item.radar_risk_state in {"IMMINENT", "CONFIRMED"}
            and cls._descending(item)
        ]
        if not camera_risk and not radar_risk:
            return "NO_RISK_SEQUENCE"
        if not camera_risk or not radar_risk:
            return "INSUFFICIENT_EVIDENCE"
        delta = (camera_risk[-1] - radar_risk[-1]).total_seconds()
        if abs(delta) <= 0.25:
            return "ALIGNED"
        return "RADAR_THEN_CAMERA" if delta > 0 else "CAMERA_THEN_RADAR"

    @staticmethod
    def _radar_snapshot(radar: RadarEvidence) -> dict[str, float | int | bool | str | None]:
        feature = radar.radar_feature if isinstance(radar.radar_feature, dict) else {}
        return {
            "score": radar.radar_score,
            "vertical_velocity": feature.get("vertical_velocity"),
            "height_delta": feature.get("height_delta"),
            "motion_direction": feature.get("motion_direction"),
            "point_count": feature.get("point_count"),
            "missing_frame_ratio": feature.get("missing_frame_ratio"),
            "quality": radar.radar_quality,
            "timestamp": radar.source_timestamp.isoformat(),
            "target_id": feature.get("target_id"),
            "track_count": feature.get("track_count"),
        }

    def _reset_unlocked(self) -> None:
        self._camera.clear()
        self._radar.clear()
        self._last_camera_timestamp = None
        self._last_radar_timestamp = None

    def reset(self) -> None:
        with self._lock:
            self._stream_key = None
            self._reset_unlocked()
