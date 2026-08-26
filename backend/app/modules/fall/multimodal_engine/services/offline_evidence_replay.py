from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    CameraEvidence,
    MultimodalLatestResponse,
    MultimodalQualitySummary,
    OfflineReplayLatestResponse,
    RadarEvidence,
)
from app.modules.fall.multimodal_engine.services.fusion_runtime import FusionStateConfig, FusionStateMachine
from app.modules.fall.multimodal_engine.services.multimodal_fusion import MultimodalFusionService


class OfflineEvidenceReplayService:
    """Replay a bounded evidence preview without touching live fusion state."""

    def __init__(
        self,
        preview_path: Path,
        fusion_service: MultimodalFusionService,
        *,
        state_config: FusionStateConfig,
    ) -> None:
        self.preview_path = preview_path
        self.fusion_service = fusion_service
        self.state_machines = {
            method: FusionStateMachine(state_config)
            for method in (
                "fixed_weighted",
                "quality_weighted",
                "radar_quality_adaptive",
            )
        }
        self._rows: list[dict[str, Any]] | None = None
        self._last_cursor: dict[str, int | None] = {
            "fixed_weighted": None,
            "quality_weighted": None,
            "radar_quality_adaptive": None,
        }
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.preview_path.is_file()

    def get(self, cursor: int, *, method: str = "fixed_weighted") -> OfflineReplayLatestResponse:
        with self._lock:
            rows = self._load()
            index = cursor % len(rows)
            state_machine = self.state_machines[method]
            if self._last_cursor[method] is not None and index <= self._last_cursor[method]:
                state_machine.reset()
            self._last_cursor[method] = index
            row = rows[index]
            now = datetime.now(timezone.utc)
            camera = self._camera(row, now)
            radar = self._radar(row, now)
            raw = self.fusion_service.fuse(camera, radar, method=method)
            fusion = state_machine.apply(camera, radar, raw)
            available_qualities = [
                value
                for value, available in (
                    (camera.camera_quality, camera.available),
                    (radar.radar_quality, radar.available),
                )
                if available
            ]
            if not available_qualities:
                overall, level = 0.0, "INSUFFICIENT_DATA"
            elif len(available_qualities) == 1:
                overall, level = available_qualities[0] * 0.75, "DEGRADED"
            else:
                overall = sum(available_qualities) / 2.0
                level = "GOOD" if overall >= 0.75 else "DEGRADED"
            response = MultimodalLatestResponse(
                camera=camera,
                radar=radar,
                fusion=fusion,
                dynamic_risk=self.fusion_service.dynamic_risk_index.build_dynamic_risk(
                    camera,
                    radar,
                    None,
                ),
                short_term_warning=(
                    self.fusion_service.dynamic_risk_index.build_short_term_warning(
                        camera,
                        radar,
                        fusion,
                    )
                ),
                fall_event=self.fusion_service.dynamic_risk_index.build_fall_event(
                    camera,
                    radar,
                ),
                operating_mode="OFFLINE_EVIDENCE_REPLAY",
                data_source="PUBLIC_EVIDENCE_REPLAY",
                timestamp=now,
                quality=MultimodalQualitySummary(
                    camera=camera.camera_quality,
                    radar=radar.radar_quality,
                    synchronization=(1.0 if fusion.synchronized else 0.0),
                    overall=overall,
                    level=level,
                ),
            )
            return OfflineReplayLatestResponse(
                dataset=str(row.get("dataset", "UNKNOWN")),
                subject_id=str(row.get("subject_id", "UNKNOWN")),
                recording_id=str(row.get("recording_id", "UNKNOWN")),
                cursor=index,
                next_cursor=(index + 1) % len(rows),
                sample_count=len(rows),
                multimodal=response,
            )

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            if not self.preview_path.is_file():
                raise RuntimeError("offline Evidence Replay preview is unavailable")
            payload = json.loads(self.preview_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or not payload:
                raise RuntimeError("offline Evidence Replay preview is empty")
            self._rows = [row for row in payload if isinstance(row, dict)]
            if not self._rows:
                raise RuntimeError("offline Evidence Replay preview has no valid rows")
        return self._rows

    @staticmethod
    def _camera(row: dict[str, Any], now: datetime) -> CameraEvidence:
        source = row.get("camera")
        timestamp = _timestamp(row.get("timestamp"))
        if not isinstance(source, dict):
            return CameraEvidence(
                camera_score=None,
                camera_quality=0.0,
                quality_level="INSUFFICIENT_DATA",
                timestamp=timestamp,
                source_timestamp=timestamp,
                window_start=timestamp,
                window_end=timestamp,
                received_at=now,
                evidence_age_ms=0.0,
                available=False,
                device_id=f"replay-{row.get('subject_id', 'unknown')}",
                quality_reason="camera evidence absent in offline dataset",
            )
        available = bool(source.get("available", False))
        return CameraEvidence(
            camera_score=float(source["score"]) if available and source.get("score") is not None else None,
            camera_feature=source.get("feature"),
            camera_quality=float(source.get("quality", 0.0)) if available else 0.0,
            quality_level=source.get("quality_level", "INSUFFICIENT_DATA"),
            timestamp=_timestamp(source.get("timestamp", row.get("timestamp"))),
            source_timestamp=_timestamp(source.get("timestamp", row.get("timestamp"))),
            window_start=_timestamp(source.get("timestamp", row.get("timestamp"))),
            window_end=_timestamp(source.get("timestamp", row.get("timestamp"))),
            received_at=now,
            evidence_age_ms=0.0,
            available=available,
            device_id=f"replay-{row.get('subject_id', 'unknown')}",
            model_version=str(source.get("source", "offline-camera-proxy")),
            quality_reason="offline Evidence Replay camera proxy",
        )

    @staticmethod
    def _radar(row: dict[str, Any], now: datetime) -> RadarEvidence:
        source = row.get("radar")
        timestamp = _timestamp(row.get("timestamp"))
        if not isinstance(source, dict):
            return RadarEvidence(
                radar_score=None,
                radar_quality=0.0,
                quality_level="INSUFFICIENT_DATA",
                timestamp=timestamp,
                source_timestamp=timestamp,
                window_start=timestamp,
                window_end=timestamp,
                received_at=now,
                evidence_age_ms=0.0,
                available=False,
                room="living_room",
                device_id=f"replay-{row.get('subject_id', 'unknown')}",
                quality_reason="radar evidence absent in offline dataset",
            )
        available = bool(source.get("available", False))
        source_timestamp = _timestamp(source.get("timestamp", row.get("timestamp")))
        return RadarEvidence(
            radar_score=float(source["score"]) if available and source.get("score") is not None else None,
            radar_feature=source.get("feature"),
            radar_quality=float(source.get("quality", 0.0)) if available else 0.0,
            quality_level=source.get("quality_level", "INSUFFICIENT_DATA"),
            timestamp=source_timestamp,
            source_timestamp=source_timestamp,
            window_start=source_timestamp,
            window_end=source_timestamp,
            received_at=now,
            evidence_age_ms=0.0,
            available=available,
            room="living_room",
            device_id=f"replay-{row.get('subject_id', 'unknown')}",
            model_version="frozen-tcn-b0-offline-replay",
            quality_reason="offline Evidence Replay radar evidence",
        )


def _timestamp(value: object) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
