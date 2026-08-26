from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
from typing import Any

from app.modules.fall.multimodal_engine.schemas.fall_live import FallLiveStatusResponse
from app.modules.fall.multimodal_engine.schemas.multimodal import AlignedPersonEvidence
from app.modules.fall.multimodal_engine.schemas.radar import RadarAlignmentEvidencePayload, RadarStatusResponse


@dataclass(frozen=True, slots=True)
class RadarTrackEvidenceFrame:
    """One Radar tracking frame retained only for timestamp association."""

    room: str
    device_id: str
    source_mode: str
    source_timestamp: datetime
    frame_number: int | None
    radar_config_name: str | None
    targets: tuple[RadarAlignmentEvidencePayload, ...]


@dataclass(frozen=True, slots=True)
class RadarTrackEvidenceMatch:
    frame: RadarTrackEvidenceFrame
    sync_delta_ms: float
    radar_age_ms: float
    camera_age_ms: float
    fresh: bool


class RadarTrackEvidenceBuffer:
    """Short, room/device-scoped history of real Radar tracking frames.

    This buffer contains geometry only. It does not retain or alter Radar risk
    scores, model outputs, thresholds, or Fusion decisions.
    """

    def __init__(
        self,
        *,
        retention_seconds: float = 3.0,
        freshness_seconds: float = 1.0,
        max_frames_per_stream: int = 128,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("Radar track retention must be positive")
        if freshness_seconds <= 0 or freshness_seconds > retention_seconds:
            raise ValueError("Radar track freshness must be within retention")
        if max_frames_per_stream < 2:
            raise ValueError("Radar track buffer needs at least two frames")
        self.retention_seconds = float(retention_seconds)
        self.freshness_seconds = float(freshness_seconds)
        self.max_frames_per_stream = int(max_frames_per_stream)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._frames: dict[tuple[str, str], deque[RadarTrackEvidenceFrame]] = {}
        self._stream_identity: dict[tuple[str, str], tuple[str, str | None]] = {}
        self._last_timestamp: dict[tuple[str, str], datetime] = {}
        self._last_frame_number: dict[tuple[str, str], int | None] = {}
        self._last_signature: dict[
            tuple[str, str], tuple[datetime, int | None, tuple[int, ...]]
        ] = {}
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def frame_count(self, *, room: str, device_id: str) -> int:
        with self._lock:
            return len(self._frames.get((room, device_id), ()))

    def clear(self) -> None:
        """Drop all pre-reconnect frames so they cannot bind to a new stream."""

        with self._lock:
            self._frames.clear()
            self._stream_identity.clear()
            self._last_timestamp.clear()
            self._last_frame_number.clear()
            self._last_signature.clear()
            self._generation += 1

    def observe(self, status: RadarStatusResponse) -> None:
        """Store each unique tracking frame emitted by the Radar poller."""

        if not status.online:
            self.clear()
            return
        if status.room is None or status.device_id is None or status.source_mode is None:
            return

        valid_targets = [
            target
            for target in status.alignment_evidence
            if target.track_id is not None and target.x is not None and target.y is not None
        ]
        if not valid_targets:
            return

        grouped: dict[datetime, list[RadarAlignmentEvidencePayload]] = defaultdict(list)
        for target in valid_targets:
            grouped[target.source_timestamp].append(target)

        key = (str(status.room), status.device_id)
        with self._lock:
            for source_timestamp in sorted(grouped):
                targets = grouped[source_timestamp]
                frame_numbers = [
                    target.frame_number
                    for target in targets
                    if target.frame_number is not None
                ]
                frame_number = max(frame_numbers) if frame_numbers else None
                config_names = sorted(
                    {
                        target.radar_config_name
                        for target in targets
                        if target.radar_config_name is not None
                    }
                )
                config_name = "+".join(config_names) if config_names else None
                identity = (str(status.source_mode), config_name)
                previous_identity = self._stream_identity.get(key)
                if previous_identity is not None and previous_identity != identity:
                    self._clear_key_unlocked(key)
                self._stream_identity[key] = identity

                previous_timestamp = self._last_timestamp.get(key)
                previous_frame = self._last_frame_number.get(key)
                stream_reset = bool(
                    previous_timestamp is not None
                    and source_timestamp < previous_timestamp
                ) or bool(
                    frame_number is not None
                    and previous_frame is not None
                    and frame_number < previous_frame
                )
                if stream_reset:
                    self._clear_key_unlocked(key)
                    self._stream_identity[key] = identity

                signature = (
                    source_timestamp,
                    frame_number,
                    tuple(sorted(int(target.track_id) for target in targets if target.track_id is not None)),
                )
                if signature == self._last_signature.get(key):
                    continue

                frame = RadarTrackEvidenceFrame(
                    room=key[0],
                    device_id=key[1],
                    source_mode=str(status.source_mode),
                    source_timestamp=source_timestamp,
                    frame_number=frame_number,
                    radar_config_name=config_name,
                    targets=tuple(target.model_copy(deep=True) for target in targets),
                )
                frames = self._frames.setdefault(
                    key,
                    deque(maxlen=self.max_frames_per_stream),
                )
                frames.append(frame)
                self._last_timestamp[key] = source_timestamp
                self._last_frame_number[key] = frame_number
                self._last_signature[key] = signature
            self._prune_unlocked(self._clock())

    def nearest(
        self,
        camera_timestamp: datetime,
        *,
        room: str | None,
        device_id: str | None,
    ) -> RadarTrackEvidenceMatch | None:
        """Return the closest retained Radar frame without relaxing any gate."""

        if room is None or device_id is None:
            return None
        now = self._clock()
        with self._lock:
            self._prune_unlocked(now)
            frames = tuple(self._frames.get((str(room), device_id), ()))
        if not frames:
            return None
        frame = min(
            frames,
            key=lambda item: abs(
                (camera_timestamp - item.source_timestamp).total_seconds()
            ),
        )
        sync_delta_ms = abs(
            (camera_timestamp - frame.source_timestamp).total_seconds() * 1000.0
        )
        radar_age_ms = max(0.0, (now - frame.source_timestamp).total_seconds() * 1000.0)
        camera_age_ms = max(0.0, (now - camera_timestamp).total_seconds() * 1000.0)
        freshness_ms = self.freshness_seconds * 1000.0
        return RadarTrackEvidenceMatch(
            frame=frame,
            sync_delta_ms=sync_delta_ms,
            radar_age_ms=radar_age_ms,
            camera_age_ms=camera_age_ms,
            fresh=radar_age_ms <= freshness_ms and camera_age_ms <= freshness_ms,
        )

    def _clear_key_unlocked(self, key: tuple[str, str]) -> None:
        self._frames.pop(key, None)
        self._stream_identity.pop(key, None)
        self._last_timestamp.pop(key, None)
        self._last_frame_number.pop(key, None)
        self._last_signature.pop(key, None)
        self._generation += 1

    def _prune_unlocked(self, now: datetime) -> None:
        for key, frames in list(self._frames.items()):
            while frames and (
                now - frames[0].source_timestamp
            ).total_seconds() > self.retention_seconds:
                frames.popleft()
            if not frames:
                self._frames.pop(key, None)


class CameraRadarAlignmentAdapter:
    """Coarse geometry gate around frozen calibration metadata."""

    def __init__(
        self,
        calibration_path: Path | None,
        *,
        enabled: bool = True,
        realtime_active: bool = False,
        radar_track_buffer: RadarTrackEvidenceBuffer | None = None,
    ) -> None:
        self.calibration_path = calibration_path
        self.enabled = enabled
        self.realtime_active = realtime_active
        self.radar_track_buffer = radar_track_buffer
        self.calibration: dict[str, Any] | None = None
        self.load_error: str | None = None
        if not enabled:
            self.load_error = "ALIGNMENT_SHADOW_DISABLED"
            return
        if calibration_path is None:
            self.load_error = "CALIBRATION_PATH_NOT_CONFIGURED"
            return
        try:
            candidate = json.loads(calibration_path.read_text(encoding="utf-8"))
            self._validate_calibration(candidate)
            self.calibration = candidate
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.load_error = f"CALIBRATION_LOAD_FAILED:{type(exc).__name__}"

    def apply(
        self,
        camera_status: FallLiveStatusResponse,
        radar_status: RadarStatusResponse,
    ) -> AlignedPersonEvidence:
        calibration = self.calibration
        base_reasons = [
            "REALTIME_ALIGNMENT_GATE" if self.realtime_active else "SHADOW_ONLY",
            "COARSE_POSITION_ASSOCIATION",
            "COARSE_COMPATIBILITY_NOT_IDENTITY",
        ]
        if calibration is None:
            return self._result(
                "CALIBRATION_INVALID",
                reasons=base_reasons + [self.load_error or "CALIBRATION_UNAVAILABLE"],
            )

        camera = camera_status.alignment_snapshot
        if camera is None or not camera.detected or camera.camera_person_id is None:
            return self._result(
                "CAMERA_PERSON_MISSING",
                calibration=calibration,
                camera_frame_id=camera.frame_id if camera is not None else None,
                reasons=base_reasons + ["CAMERA_ALIGNMENT_SNAPSHOT_UNAVAILABLE"],
            )
        if camera.bbox_xyxy is None or camera.footpoint_uv is None:
            return self._result(
                "CAMERA_PERSON_MISSING",
                calibration=calibration,
                camera_person_id=camera.camera_person_id,
                camera_frame_id=camera.frame_id,
                camera_footpoint_uv=camera.footpoint_uv,
                reasons=base_reasons + ["CAMERA_FOOTPOINT_OR_BBOX_MISSING"],
            )

        buffered_match = None
        if self.radar_track_buffer is not None:
            if not radar_status.online:
                return self._result(
                    "RADAR_TRACK_MISSING",
                    calibration=calibration,
                    camera_person_id=camera.camera_person_id,
                    camera_frame_id=camera.frame_id,
                    camera_footpoint_uv=camera.footpoint_uv,
                    reasons=base_reasons + ["RADAR_STREAM_OFFLINE"],
                )
            buffered_match = self.radar_track_buffer.nearest(
                camera.source_timestamp,
                room=radar_status.room,
                device_id=radar_status.device_id,
            )
            targets = (
                list(buffered_match.frame.targets)
                if buffered_match is not None
                else []
            )
            base_reasons.append("RADAR_TRACK_BUFFER_NEAREST_TIMESTAMP")
        else:
            targets = [
                target
                for target in radar_status.alignment_evidence
                if target.track_id is not None
                and target.x is not None
                and target.y is not None
            ]
        if not targets:
            return self._result(
                "RADAR_TRACK_MISSING",
                calibration=calibration,
                camera_person_id=camera.camera_person_id,
                camera_frame_id=camera.frame_id,
                camera_footpoint_uv=camera.footpoint_uv,
                reasons=base_reasons
                + [
                    "RADAR_TRACK_BUFFER_EMPTY"
                    if self.radar_track_buffer is not None
                    else "RADAR_TRACK_UNAVAILABLE"
                ],
            )

        radar_timestamp = (
            buffered_match.frame.source_timestamp
            if buffered_match is not None
            else max(target.source_timestamp for target in targets)
        )
        sync_delta_ms = (
            buffered_match.sync_delta_ms
            if buffered_match is not None
            else abs(
                (camera.source_timestamp - radar_timestamp).total_seconds() * 1000.0
            )
        )
        if buffered_match is not None and not buffered_match.fresh:
            return self._result(
                "OUT_OF_SYNC",
                calibration=calibration,
                camera_person_id=camera.camera_person_id,
                camera_frame_id=camera.frame_id,
                radar_frame_number=buffered_match.frame.frame_number,
                radar_source_timestamp=radar_timestamp,
                sync_delta_ms=sync_delta_ms,
                camera_footpoint_uv=camera.footpoint_uv,
                reasons=base_reasons + ["RADAR_TRACK_BUFFER_EVIDENCE_STALE"],
            )
        max_sync_ms = float(calibration["gates"]["max_sync_delta_ms"])
        if sync_delta_ms > max_sync_ms:
            return self._result(
                "OUT_OF_SYNC",
                calibration=calibration,
                camera_person_id=camera.camera_person_id,
                camera_frame_id=camera.frame_id,
                radar_frame_number=self._frame_number(targets),
                radar_source_timestamp=radar_timestamp,
                sync_delta_ms=sync_delta_ms,
                camera_footpoint_uv=camera.footpoint_uv,
                reasons=base_reasons + ["ALIGNMENT_FRAME_SYNC_GATE_EXCEEDED"],
            )

        projected: list[
            tuple[RadarAlignmentEvidencePayload, float, float, float, float]
        ] = []
        outside_count = 0
        for target in targets:
            if not self._in_valid_region(calibration, float(target.x), float(target.y)):
                outside_count += 1
                continue
            u, v = self._project(calibration, float(target.x), float(target.y))
            foot_error = math.hypot(
                u - camera.footpoint_uv[0],
                v - camera.footpoint_uv[1],
            )
            bbox_distance = self._distance_to_bbox(u, v, camera.bbox_xyxy)
            projected.append((target, u, v, foot_error, bbox_distance))
        if not projected:
            reasons = base_reasons + ["OUTSIDE_CALIBRATED_RADAR_REGION"]
            if outside_count:
                reasons.append("SOME_TARGETS_OUTSIDE_CALIBRATED_REGION")
            return self._result(
                "CALIBRATION_INVALID",
                calibration=calibration,
                camera_person_id=camera.camera_person_id,
                camera_frame_id=camera.frame_id,
                radar_frame_number=self._frame_number(targets),
                radar_source_timestamp=radar_timestamp,
                sync_delta_ms=sync_delta_ms,
                camera_footpoint_uv=camera.footpoint_uv,
                reasons=reasons,
            )

        radius = float(calibration["gates"]["uncertainty_radius_px"])
        within = [candidate for candidate in projected if candidate[4] <= radius]
        if len(within) > 1:
            return self._result(
                "MULTIPLE_CANDIDATES",
                calibration=calibration,
                camera_person_id=camera.camera_person_id,
                camera_frame_id=camera.frame_id,
                radar_frame_number=self._frame_number(targets),
                sync_delta_ms=sync_delta_ms,
                camera_footpoint_uv=camera.footpoint_uv,
                reasons=base_reasons + ["MULTIPLE_RADAR_TRACKS_INSIDE_SPATIAL_GATE"],
            )

        selected = within[0] if within else min(projected, key=lambda item: item[4])
        target, u, v, foot_error, bbox_distance = selected
        common = dict(
            calibration=calibration,
            camera_person_id=camera.camera_person_id,
            camera_frame_id=camera.frame_id,
            radar_track_id=target.track_id,
            radar_frame_number=target.frame_number,
            radar_source_timestamp=target.source_timestamp,
            sync_delta_ms=sync_delta_ms,
            radar_position_xyz_m=(target.x, target.y, target.z),
            radar_velocity_xyz_mps=(target.vx, target.vy, target.vz),
            radar_point_count=target.point_count,
            radar_point_cloud_spread_m=target.point_cloud_spread_m,
            radar_target_confidence=target.target_confidence,
            radar_config_name=target.radar_config_name,
            camera_footpoint_uv=camera.footpoint_uv,
            projected_radar_uv=(u, v),
            bbox_gate_distance_px=bbox_distance,
        )
        if not within:
            return self._result(
                "TRACK_CONFLICT",
                reasons=base_reasons + ["SPATIAL_GATE_FAILED"],
                **common,
            )

        spatial_score = self._clamp(1.0 - bbox_distance / radius)
        sync_score = self._clamp(1.0 - sync_delta_ms / max_sync_ms)
        radar_quality = target.target_confidence
        if radar_quality is None:
            radar_quality = target.radar_quality
        confidence = min(
            0.70,
            self._clamp(
                0.50 * spatial_score
                + 0.20 * sync_score
                + 0.15 * camera.footpoint_confidence
                + 0.15 * radar_quality
            ),
        )
        reasons = base_reasons + ["TARGET_ASSOCIATION_MATCHED"]
        if len(targets) > 1:
            reasons.append("MULTI_RADAR_DISAMBIGUATED_BY_SPATIAL_GATE")
        if outside_count:
            reasons.append("SOME_TARGETS_OUTSIDE_CALIBRATED_REGION")
        loo_p95 = float(
            calibration.get("training_metrics", {}).get(
                "leave_one_out_p95_px",
                radius * 0.75,
            )
        )
        if foot_error > loo_p95:
            reasons.append("CALIBRATION_LOW_CONFIDENCE")
        return self._result(
            "MATCHED",
            association_confidence=confidence,
            eligible=True,
            reasons=reasons,
            **common,
        )

    @staticmethod
    def _validate_calibration(calibration: dict[str, Any]) -> None:
        matrix = calibration["mapping"]["matrix_2x3"]
        if (
            not isinstance(matrix, list)
            or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
        ):
            raise ValueError("mapping.matrix_2x3 must be 2x3")
        float(calibration["gates"]["max_sync_delta_ms"])
        float(calibration["gates"]["uncertainty_radius_px"])
        calibration["valid_radar_region_xy_m"]
        str(calibration["calibration_version"])

    @staticmethod
    def _project(calibration: dict[str, Any], x: float, y: float) -> tuple[float, float]:
        matrix = calibration["mapping"]["matrix_2x3"]
        return (
            float(matrix[0][0]) * x + float(matrix[0][1]) * y + float(matrix[0][2]),
            float(matrix[1][0]) * x + float(matrix[1][1]) * y + float(matrix[1][2]),
        )

    @staticmethod
    def _in_valid_region(calibration: dict[str, Any], x: float, y: float) -> bool:
        region = calibration["valid_radar_region_xy_m"]
        return (
            float(region["x_min"]) <= x <= float(region["x_max"])
            and float(region["y_min"]) <= y <= float(region["y_max"])
        )

    @staticmethod
    def _distance_to_bbox(
        u: float,
        v: float,
        bbox: tuple[float, float, float, float],
    ) -> float:
        dx = max(bbox[0] - u, 0.0, u - bbox[2])
        dy = max(bbox[1] - v, 0.0, v - bbox[3])
        return math.hypot(dx, dy)

    @staticmethod
    def _frame_number(targets: list[RadarAlignmentEvidencePayload]) -> int | None:
        values = [target.frame_number for target in targets if target.frame_number is not None]
        return max(values) if values else None

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _result(
        self,
        state: str,
        *,
        calibration: dict[str, Any] | None = None,
        camera_person_id: int | None = None,
        radar_track_id: int | None = None,
        camera_frame_id: str | None = None,
        radar_frame_number: int | None = None,
        radar_source_timestamp: datetime | None = None,
        sync_delta_ms: float | None = None,
        radar_position_xyz_m: tuple[float | None, float | None, float | None] = (
            None,
            None,
            None,
        ),
        radar_velocity_xyz_mps: tuple[float | None, float | None, float | None] = (
            None,
            None,
            None,
        ),
        radar_point_count: int = 0,
        radar_point_cloud_spread_m: float | None = None,
        radar_target_confidence: float | None = None,
        radar_config_name: str | None = None,
        camera_footpoint_uv: tuple[float, float] | None = None,
        projected_radar_uv: tuple[float, float] | None = None,
        bbox_gate_distance_px: float | None = None,
        association_confidence: float = 0.0,
        eligible: bool = False,
        reasons: list[str],
    ) -> AlignedPersonEvidence:
        return AlignedPersonEvidence(
            association_state=state,
            camera_person_id=camera_person_id,
            radar_track_id=radar_track_id,
            camera_frame_id=camera_frame_id,
            radar_frame_number=radar_frame_number,
            radar_source_timestamp=radar_source_timestamp,
            sync_delta_ms=sync_delta_ms,
            radar_position_xyz_m=radar_position_xyz_m,
            radar_velocity_xyz_mps=radar_velocity_xyz_mps,
            radar_point_count=radar_point_count,
            radar_point_cloud_spread_m=radar_point_cloud_spread_m,
            radar_target_confidence=radar_target_confidence,
            radar_config_name=radar_config_name,
            camera_footpoint_uv=camera_footpoint_uv,
            projected_radar_uv=projected_radar_uv,
            bbox_gate_distance_px=bbox_gate_distance_px,
            association_confidence=association_confidence,
            calibration_version=(
                str(calibration["calibration_version"])
                if calibration is not None
                else None
            ),
            eligible_for_temporal_association=eligible,
            reason_codes=list(dict.fromkeys(reasons)),
            shadow_only=not self.realtime_active,
            realtime_active=self.realtime_active,
            affects_realtime_fusion_v2=self.realtime_active,
        )
