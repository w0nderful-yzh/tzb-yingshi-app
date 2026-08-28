from __future__ import annotations

import json
import logging
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    CameraEvidence,
    MultimodalLatestResponse,
    MultimodalRuntimeStatistics,
    MultimodalTimingAudit,
    RadarEvidence,
    RiskTrendPoint,
    RuntimeAverageQuality,
    RuntimeUnknownRatio,
)


class FusionTimingTracker:
    def __init__(self, *, tolerance_ms: float, capacity: int = 2000) -> None:
        self.tolerance_ms = tolerance_ms
        self._samples: deque[float] = deque(maxlen=capacity)
        self._last_signature: tuple[datetime, datetime] | None = None
        self._lock = threading.Lock()

    def observe(self, camera: CameraEvidence, radar: RadarEvidence) -> MultimodalTimingAudit:
        with self._lock:
            if camera.available and radar.available:
                signature = (camera.source_timestamp, radar.source_timestamp)
                if signature != self._last_signature:
                    delta = (
                        abs((camera.source_timestamp - radar.source_timestamp).total_seconds())
                        * 1000.0
                    )
                    self._samples.append(delta)
                    self._last_signature = signature
            values = sorted(self._samples)
            current = (
                abs((camera.source_timestamp - radar.source_timestamp).total_seconds()) * 1000.0
                if camera.available and radar.available
                else None
            )
            return MultimodalTimingAudit(
                sync_delta_ms=current,
                sync_p50_ms=_quantile(values, 0.50) if values else None,
                sync_p95_ms=_quantile(values, 0.95) if values else None,
                sync_sample_count=len(values),
                tolerance_ms=self.tolerance_ms,
            )


class FusionRuntimeTracker:
    """Keep a bounded deduplicated observability window for the dashboard."""

    def __init__(self, *, capacity: int = 120) -> None:
        if capacity < 2:
            raise ValueError("fusion runtime statistics capacity must be at least two")
        self._samples: deque[dict[str, object]] = deque(maxlen=capacity)
        self._last_signature: tuple[object, ...] | None = None
        self._lock = threading.Lock()

    def observe(self, response: MultimodalLatestResponse) -> MultimodalRuntimeStatistics:
        camera = response.camera
        radar = response.radar
        fusion = response.camera_led_evidence_fusion_v2
        signature = (
            camera.source_timestamp,
            radar.source_timestamp,
            fusion.camera_led_state,
            fusion.fusion_mode,
            response.dynamic_risk.dynamic_risk_score,
            response.fall_event.fall_event_status,
        )
        with self._lock:
            if signature != self._last_signature:
                physio = response.physiological_evidence
                self._samples.append(
                    {
                        "timestamp": response.timestamp,
                        "camera_unknown": camera.camera_risk_state == "UNKNOWN",
                        "radar_unknown": radar.radar_risk_state == "UNKNOWN",
                        "dynamic_unknown": not response.dynamic_risk.available,
                        "short_unknown": (
                            response.short_term_warning is None
                            or response.short_term_warning.state == "UNKNOWN"
                        ),
                        "event_unknown": response.fall_event.fall_event_status == "UNKNOWN",
                        "camera_quality": camera.camera_quality,
                        "radar_quality": radar.radar_quality,
                        "overall_quality": response.quality.overall,
                        "physio_quality": (physio.quality_coverage if physio.enabled else None),
                        "dynamic_score": response.dynamic_risk.dynamic_risk_score,
                        "short_score": (
                            response.short_term_warning.short_term_fall_score
                            if response.short_term_warning is not None
                            else None
                        ),
                        "event_state": response.fall_event.fall_event_status,
                    }
                )
                self._last_signature = signature
            return self._summary_unlocked()

    def _summary_unlocked(self) -> MultimodalRuntimeStatistics:
        samples = list(self._samples)
        if not samples:
            return MultimodalRuntimeStatistics()
        count = len(samples)
        physio_values = [
            float(sample["physio_quality"])
            for sample in samples
            if sample["physio_quality"] is not None
        ]
        risk_samples = samples[-30:]
        return MultimodalRuntimeStatistics(
            sample_count=count,
            window_start=samples[0]["timestamp"],
            window_end=samples[-1]["timestamp"],
            risk_trend=[
                RiskTrendPoint(
                    timestamp=sample["timestamp"],
                    dynamic_risk_score=sample["dynamic_score"],
                    short_term_fall_score=sample["short_score"],
                    fall_event_status=sample["event_state"],
                )
                for sample in risk_samples
            ],
            unknown_ratio=RuntimeUnknownRatio(
                camera=sum(bool(sample["camera_unknown"]) for sample in samples) / count,
                radar=sum(bool(sample["radar_unknown"]) for sample in samples) / count,
                dynamic_risk=sum(bool(sample["dynamic_unknown"]) for sample in samples) / count,
                short_term_fall=sum(bool(sample["short_unknown"]) for sample in samples) / count,
                fall_event=sum(bool(sample["event_unknown"]) for sample in samples) / count,
            ),
            average_quality=RuntimeAverageQuality(
                camera=sum(float(sample["camera_quality"]) for sample in samples) / count,
                radar=sum(float(sample["radar_quality"]) for sample in samples) / count,
                overall=sum(float(sample["overall_quality"]) for sample in samples) / count,
                physiological=(sum(physio_values) / len(physio_values) if physio_values else None),
            ),
        )


class FusionShadowLogger:
    """Write deduplicated shadow evidence; failures never affect risk decisions."""

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool,
        max_bytes: int = 20 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self.enabled = enabled
        self._last_signature: tuple[object, ...] | None = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"fusion-shadow-{id(self)}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def write(self, response: MultimodalLatestResponse) -> None:
        if not self.enabled:
            return
        camera = response.camera
        radar = response.radar
        associated = response.associated_risk_augmentation
        camera_led_v2 = response.camera_led_evidence_fusion_v2
        signature = (
            camera.source_timestamp,
            radar.source_timestamp,
            camera_led_v2.camera_led_state,
            camera_led_v2.fusion_mode,
        )
        with self._lock:
            if signature == self._last_signature:
                return
            self._last_signature = signature
            payload = {
                "schema_version": "camera_led_radar_evidence_runtime_v1",
                "logged_at": datetime.now(UTC).isoformat(),
                "operating_mode": response.operating_mode,
                "data_source": response.data_source,
                "timestamp": response.timestamp.isoformat(),
                "camera_score": camera.camera_score,
                "radar_score": radar.radar_score,
                "camera_state": (
                    camera.camera_feature.get("risk_state")
                    if isinstance(camera.camera_feature, dict)
                    else None
                ),
                "camera_positive_votes": (
                    camera.camera_feature.get("positive_votes")
                    if isinstance(camera.camera_feature, dict)
                    else None
                ),
                "radar_state": (
                    radar.radar_feature.get("risk_state")
                    if isinstance(radar.radar_feature, dict)
                    else None
                ),
                "camera_quality": camera.camera_quality,
                "radar_quality": radar.radar_quality,
                "sync_delta_ms": camera_led_v2.sync_delta_ms,
                "fusion_mode": camera_led_v2.fusion_mode,
                "reason_codes": camera_led_v2.reason_codes,
                "radar_eligibility": {
                    "eligible": camera_led_v2.radar_eligible,
                    "association_state": camera_led_v2.association_state,
                },
                "risk_state": {
                    "camera": camera.camera_risk_state,
                    "radar": radar.radar_risk_state,
                    "multimodal": camera_led_v2.camera_led_state,
                },
                "dynamic_risk_score": response.dynamic_risk.dynamic_risk_score,
                "dynamic_risk_level": response.dynamic_risk.risk_level,
                "dynamic_risk_available": response.dynamic_risk.available,
                "dynamic_risk_reasons": [
                    reason.model_dump(mode="json") for reason in response.dynamic_risk.reasons
                ],
                "short_term_fall_score": (
                    response.short_term_warning.short_term_fall_score
                    if response.short_term_warning is not None
                    else None
                ),
                "fall_event_status": response.fall_event.fall_event_status,
                "physiological_evidence": response.physiological_evidence.model_dump(mode="json"),
                "final_decision_context": (
                    response.final_decision_context.model_dump(mode="json")
                    if response.final_decision_context is not None
                    else None
                ),
                "runtime_statistics": {
                    "sample_count": response.runtime_statistics.sample_count,
                    "unknown_ratio": response.runtime_statistics.unknown_ratio.model_dump(
                        mode="json"
                    ),
                    "average_quality": response.runtime_statistics.average_quality.model_dump(
                        mode="json"
                    ),
                },
                "room": radar.room,
                "device": radar.device_id or camera.device_id,
                "model_version": {
                    "camera": camera.model_version,
                    "radar": radar.model_version,
                    "multimodal": "camera-led-evidence-fusion-v2-realtime-v1",
                },
                "camera_source_timestamp": camera.source_timestamp.isoformat(),
                "radar_source_timestamp": radar.source_timestamp.isoformat(),
                "camera_window_start": camera.window_start.isoformat(),
                "camera_window_end": camera.window_end.isoformat(),
                "radar_window_start": radar.window_start.isoformat(),
                "radar_window_end": radar.window_end.isoformat(),
                "camera_processing_latency_ms": camera.processing_latency_ms,
                "radar_processing_latency_ms": radar.processing_latency_ms,
                "camera_evidence_age_ms": camera.evidence_age_ms,
                "radar_evidence_age_ms": radar.evidence_age_ms,
                "radar_evidence_snapshot": radar.radar_feature,
                "associated_risk_augmentation": (
                    associated.model_dump(mode="json") if associated is not None else None
                ),
                "camera_led_evidence_fusion_v2": (
                    {
                        "camera_score": camera_led_v2.camera_score,
                        "radar_score": camera_led_v2.radar_score,
                        "radar_quality": camera_led_v2.radar_quality,
                        "fusion_mode": camera_led_v2.fusion_mode,
                        "camera_led_state": camera_led_v2.camera_led_state,
                        "reason_codes": camera_led_v2.reason_codes,
                        "radar_eligible": camera_led_v2.radar_eligible,
                        "radar_motion_evidence_strength": (
                            camera_led_v2.radar_motion_evidence_strength
                        ),
                        "shadow_only": camera_led_v2.shadow_only,
                        "affects_alerts": camera_led_v2.affects_alerts,
                    }
                ),
                "alignment": response.alignment.model_dump(mode="json"),
                "shadow_extensions_schema": ("camera_led_evidence_fusion_v2_log_v1"),
            }
            try:
                self._logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                logging.getLogger(__name__).exception("fusion shadow log write failed")


class FusionShadowSampler:
    """Sample the active Camera-led path even when no dashboard is open."""

    def __init__(
        self,
        sample: Callable[[], object],
        *,
        enabled: bool,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("fusion shadow sample interval must be positive")
        self.sample = sample
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="fusion-shadow-sampler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval_seconds * 2.0))
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample()
            except Exception:
                logging.getLogger(__name__).exception(
                    "fusion shadow sampling failed; single-modality services remain active"
                )
            self._stop_event.wait(self.interval_seconds)


FusionResponseCallback = Callable[[MultimodalLatestResponse], None]


def _quantile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(values) - 1)
    weight = position - left
    return values[left] * (1.0 - weight) + values[right] * weight
