from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import threading

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    AlignedPersonEvidence,
    CameraEvidence,
    RadarEligibilityDecision,
    RadarEvidence,
)


@dataclass(frozen=True)
class RadarEligibilityConfig:
    """Configurable engineering gates; none of these values alter either model."""

    enabled: bool = True
    history_window_seconds: float = 1.2
    minimum_track_samples: int = 2
    minimum_point_count: int = 3
    reference_point_count: int = 20
    minimum_track_stability: float = 0.60
    minimum_radar_quality: float = 0.25
    maximum_velocity_jump_mps: float = 1.5
    height_consistency_tolerance_m: float = 0.50

    def __post_init__(self) -> None:
        if self.history_window_seconds <= 0:
            raise ValueError("eligibility history window must be positive")
        if self.minimum_track_samples < 2:
            raise ValueError("eligibility needs at least two track samples")
        if self.minimum_point_count < 1:
            raise ValueError("minimum point count must be positive")
        if self.reference_point_count < self.minimum_point_count:
            raise ValueError("reference point count must not be below minimum")
        if not 0 <= self.minimum_track_stability <= 1:
            raise ValueError("minimum track stability must be within [0, 1]")
        if not 0 <= self.minimum_radar_quality <= 1:
            raise ValueError("minimum Radar quality must be within [0, 1]")
        if self.maximum_velocity_jump_mps <= 0:
            raise ValueError("maximum velocity jump must be positive")
        if self.height_consistency_tolerance_m <= 0:
            raise ValueError("height consistency tolerance must be positive")


@dataclass(frozen=True)
class _TrackSample:
    timestamp: datetime
    track_id: int
    z_m: float | None
    vz_mps: float | None


class RadarEligibilityGate:
    """Allow Radar into Fusion only after association, timing and quality checks."""

    def __init__(self, config: RadarEligibilityConfig | None = None) -> None:
        self.config = config or RadarEligibilityConfig()
        self._samples: deque[_TrackSample] = deque(maxlen=64)
        self._last_signature: tuple[int, datetime] | None = None
        self._lock = threading.RLock()

    def evaluate(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        alignment: AlignedPersonEvidence | None,
        *,
        sync_tolerance_seconds: float,
    ) -> RadarEligibilityDecision:
        if not self.config.enabled:
            return RadarEligibilityDecision(reason_codes=["ELIGIBILITY_GATE_DISABLED"])
        if not radar.available or radar.radar_score is None:
            return self._ineligible("RADAR_MISSING")
        if alignment is None:
            return self._ineligible("TRACK_MISMATCH", target_detected=True)

        state = alignment.association_state
        target_detected = alignment.radar_track_id is not None
        if state == "RADAR_TRACK_MISSING" or not target_detected:
            return self._ineligible("RADAR_MISSING")
        if state in {
            "TRACK_CONFLICT",
            "MULTIPLE_CANDIDATES",
            "CAMERA_PERSON_MISSING",
            "CALIBRATION_INVALID",
        }:
            return self._ineligible(
                "TRACK_MISMATCH",
                target_detected=target_detected,
                extra_reasons=[state],
            )
        if state != "MATCHED" or not alignment.eligible_for_temporal_association:
            return self._ineligible(
                "TRACK_MISMATCH",
                target_detected=target_detected,
                extra_reasons=[state],
            )

        evidence_delta_ms = abs(
            (camera.source_timestamp - radar.source_timestamp).total_seconds()
        ) * 1000.0
        alignment_delta_ms = alignment.sync_delta_ms
        tolerance_ms = sync_tolerance_seconds * 1000.0
        synchronized = (
            evidence_delta_ms <= tolerance_ms
            and alignment_delta_ms is not None
            and alignment_delta_ms <= tolerance_ms
        )
        if not synchronized:
            return self._ineligible(
                "TRACK_MISMATCH",
                target_detected=True,
                target_matched=True,
                extra_reasons=["OUT_OF_SYNC"],
            )

        assert alignment.radar_track_id is not None
        timestamp = alignment.radar_source_timestamp or radar.source_timestamp
        z_value = alignment.radar_position_xyz_m[2]
        vz_value = alignment.radar_velocity_xyz_mps[2]
        with self._lock:
            signature = (alignment.radar_track_id, timestamp)
            if signature != self._last_signature:
                self._samples.append(
                    _TrackSample(
                        timestamp=timestamp,
                        track_id=alignment.radar_track_id,
                        z_m=float(z_value) if z_value is not None else None,
                        vz_mps=float(vz_value) if vz_value is not None else None,
                    )
                )
                self._last_signature = signature
            self._prune(timestamp)
            recent = list(self._samples)

        same_track = [
            sample for sample in recent if sample.track_id == alignment.radar_track_id
        ]
        track_stability = len(same_track) / len(recent) if recent else 0.0
        track_continuous = (
            len(same_track) >= self.config.minimum_track_samples
            and track_stability >= self.config.minimum_track_stability
        )

        point_count = max(0, int(alignment.radar_point_count))
        point_quality = self._point_count_quality(point_count)
        point_cloud_passed = point_count >= self.config.minimum_point_count
        velocity_continuity = self._velocity_continuity(same_track)
        height_credibility = self._height_change_credibility(same_track)
        source_quality = max(0.0, min(1.0, radar.radar_quality))
        derived_quality = source_quality * (
            0.30 * point_quality
            + 0.35 * track_stability
            + 0.20 * velocity_continuity
            + 0.15 * height_credibility
        )
        derived_quality = self._clamp(derived_quality)

        reasons: list[str] = []
        if not track_continuous:
            reasons.extend(["LOW_QUALITY", "TRACK_DISCONTINUOUS"])
        if not point_cloud_passed:
            reasons.extend(["LOW_QUALITY", "POINT_COUNT_BELOW_MINIMUM"])
        if derived_quality < self.config.minimum_radar_quality:
            reasons.extend(["LOW_QUALITY", "RADAR_QUALITY_BELOW_MINIMUM"])
        eligible = not reasons
        if eligible:
            reasons.append("RADAR_ELIGIBLE")

        return RadarEligibilityDecision(
            assessed=True,
            eligible=eligible,
            target_detected=True,
            target_matched=True,
            synchronized=True,
            track_continuous=track_continuous,
            point_cloud_quality_passed=point_cloud_passed,
            radar_quality=derived_quality,
            point_count_quality=point_quality,
            track_stability=track_stability,
            velocity_continuity=velocity_continuity,
            height_change_credibility=height_credibility,
            reason_codes=list(dict.fromkeys(reasons)),
        )

    def _prune(self, now: datetime) -> None:
        while self._samples and (
            now - self._samples[0].timestamp
        ).total_seconds() > self.config.history_window_seconds:
            self._samples.popleft()

    def _point_count_quality(self, point_count: int) -> float:
        if point_count < self.config.minimum_point_count:
            return self._clamp(point_count / self.config.minimum_point_count)
        span = self.config.reference_point_count - self.config.minimum_point_count
        if span <= 0:
            return 1.0
        return self._clamp(
            (point_count - self.config.minimum_point_count) / span
        )

    def _velocity_continuity(self, samples: list[_TrackSample]) -> float:
        values = [sample.vz_mps for sample in samples if sample.vz_mps is not None]
        if len(values) < 2:
            return 0.5
        mean_jump = sum(
            abs(current - previous)
            for previous, current in zip(values, values[1:])
        ) / (len(values) - 1)
        return self._clamp(1.0 - mean_jump / self.config.maximum_velocity_jump_mps)

    def _height_change_credibility(self, samples: list[_TrackSample]) -> float:
        usable = [
            sample
            for sample in samples
            if sample.z_m is not None and sample.vz_mps is not None
        ]
        if len(usable) < 2:
            return 0.5
        errors: list[float] = []
        for previous, current in zip(usable, usable[1:]):
            elapsed = (current.timestamp - previous.timestamp).total_seconds()
            if elapsed <= 0:
                continue
            observed_delta = current.z_m - previous.z_m
            predicted_delta = 0.5 * (previous.vz_mps + current.vz_mps) * elapsed
            errors.append(abs(observed_delta - predicted_delta))
        if not errors:
            return 0.5
        mean_error = sum(errors) / len(errors)
        return self._clamp(
            1.0 - mean_error / self.config.height_consistency_tolerance_m
        )

    @staticmethod
    def _ineligible(
        primary_reason: str,
        *,
        target_detected: bool = False,
        target_matched: bool = False,
        extra_reasons: list[str] | None = None,
    ) -> RadarEligibilityDecision:
        return RadarEligibilityDecision(
            assessed=True,
            eligible=False,
            target_detected=target_detected,
            target_matched=target_matched,
            synchronized=False,
            reason_codes=list(
                dict.fromkeys([primary_reason] + list(extra_reasons or []))
            ),
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
