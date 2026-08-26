from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
from typing import Callable

from app.modules.fall.multimodal_engine.schemas.multimodal import (
    CameraEvidence,
    ContributionTrendPoint,
    FusionResult,
    MultimodalLatestResponse,
    MultimodalRuntimeStatistics,
    MultimodalTimingAudit,
    RadarEvidence,
    RiskTrendPoint,
    RuntimeAverageQuality,
    RuntimeUnknownRatio,
)


@dataclass(frozen=True, slots=True)
class FusionStateConfig:
    ema_alpha: float = 0.35
    watch_enter: float = 0.35
    watch_exit: float = 0.25
    high_enter: float = 0.65
    high_exit: float = 0.50
    imminent_enter: float = 0.80
    watch_confirmation_windows: int = 2
    high_confirmation_windows: int = 3
    normal_confirmation_windows: int = 2
    conflict_score_gap: float = 0.45
    minimum_modality_quality: float = 0.25

    def __post_init__(self) -> None:
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("fusion EMA alpha must be in (0, 1]")
        if not 0 <= self.watch_exit < self.watch_enter < self.high_exit < self.high_enter:
            raise ValueError("fusion hysteresis thresholds are not ordered")
        if not self.high_enter <= self.imminent_enter <= 1:
            raise ValueError("fusion imminent threshold is not ordered")
        if min(
            self.watch_confirmation_windows,
            self.high_confirmation_windows,
            self.normal_confirmation_windows,
        ) < 1:
            raise ValueError("fusion confirmations must be positive")


class FusionStateMachine:
    """Stabilize a decision score without changing either modality model."""

    def __init__(self, config: FusionStateConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._stream_key: tuple[str | None, str | None] | None = None
        self._last_signature: tuple[object, ...] | None = None
        self._ema: float | None = None
        self._stable_state = "UNKNOWN"
        self._candidate_state = "UNKNOWN"
        self._candidate_count = 0

    def apply(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        raw: FusionResult,
    ) -> FusionResult:
        with self._lock:
            stream_key = (radar.room, radar.device_id or camera.device_id)
            if self._stream_key is not None and stream_key != self._stream_key:
                self._reset_unlocked()
            self._stream_key = stream_key
            signature = (
                camera.source_timestamp,
                radar.source_timestamp,
                raw.method,
                raw.raw_fusion_score,
                raw.degraded_mode,
                raw.fusion_mode,
                raw.radar_eligibility.eligible,
                tuple(raw.radar_eligibility.reason_codes),
            )
            reason_codes = list(raw.reason_codes)
            degraded_mode = raw.degraded_mode
            fusion_mode = raw.fusion_mode
            conflict = self._has_conflict(camera, radar, raw)
            if conflict:
                degraded_mode = "MODALITY_CONFLICT"
                fusion_mode = "RADAR_CONFLICT"
                reason_codes.append("MODALITY_SCORE_CONFLICT")

            if signature == self._last_signature:
                return raw.model_copy(
                    update={
                        "stable_fusion_score": self._ema,
                        "fusion_state": self._candidate_state,
                        "stable_fusion_state": self._stable_state,
                        "fusion_risk_state": self._stable_state,
                        "degraded_mode": degraded_mode,
                        "fusion_mode": fusion_mode,
                        "reason_codes": sorted(set(reason_codes)),
                    }
                )
            self._last_signature = signature

            score = raw.raw_fusion_score
            if score is None:
                self._ema = None
                self._stable_state = "UNKNOWN"
                self._candidate_state = "UNKNOWN"
                self._candidate_count = 0
                return raw.model_copy(
                    update={
                        "stable_fusion_score": None,
                        "fusion_state": "UNKNOWN",
                        "stable_fusion_state": "UNKNOWN",
                        "fusion_risk_state": "UNKNOWN",
                        "degraded_mode": "BOTH_UNAVAILABLE",
                        "fusion_mode": "NO_EVIDENCE",
                        "reason_codes": sorted(set(reason_codes + ["NO_VALID_MODALITY"])),
                    }
                )

            self._ema = score if self._ema is None else (
                self.config.ema_alpha * score + (1.0 - self.config.ema_alpha) * self._ema
            )
            constrained_watch = degraded_mode != "NONE" or conflict
            candidate = self._candidate_for_score(
                self._ema,
                camera,
                radar,
                constrained_watch=constrained_watch,
            )
            self._advance(candidate)
            return raw.model_copy(
                update={
                    "stable_fusion_score": self._ema,
                    "fusion_state": candidate,
                    "stable_fusion_state": self._stable_state,
                    "fusion_risk_state": self._stable_state,
                    "degraded_mode": degraded_mode,
                    "fusion_mode": fusion_mode,
                    "reason_codes": sorted(set(reason_codes)),
                }
            )

    def _candidate_for_score(
        self,
        score: float,
        camera: CameraEvidence,
        radar: RadarEvidence,
        *,
        constrained_watch: bool,
    ) -> str:
        if constrained_watch:
            return "WATCH"
        if (
            score >= self.config.imminent_enter
            and camera.camera_score is not None
            and radar.radar_score is not None
            and camera.camera_score >= self.config.high_enter
            and radar.radar_score >= self.config.high_enter
        ):
            return "IMMINENT"
        if self._stable_state in {"HIGH", "IMMINENT"}:
            if score >= self.config.high_exit:
                return self._stable_state
            return "WATCH"
        if score >= self.config.high_enter:
            return "HIGH"
        if self._stable_state == "WATCH":
            return "WATCH" if score >= self.config.watch_exit else "NORMAL"
        return "WATCH" if score >= self.config.watch_enter else "NORMAL"

    def _advance(self, candidate: str) -> None:
        if candidate == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = candidate
            self._candidate_count = 1
        required = (
            self.config.high_confirmation_windows
            if candidate in {"HIGH", "IMMINENT"}
            else self.config.watch_confirmation_windows
            if candidate == "WATCH"
            else self.config.normal_confirmation_windows
        )
        if self._candidate_count >= required:
            self._stable_state = candidate

    def _has_conflict(
        self,
        camera: CameraEvidence,
        radar: RadarEvidence,
        raw: FusionResult,
    ) -> bool:
        if not camera.available or not radar.available:
            return False
        if raw.radar_eligibility.assessed and not raw.radar_eligibility.eligible:
            return raw.fusion_mode == "RADAR_CONFLICT"
        assert camera.camera_score is not None and radar.radar_score is not None
        if abs(camera.camera_score - radar.radar_score) >= self.config.conflict_score_gap:
            return True
        high_low = (
            camera.camera_score >= self.config.high_enter
            and radar.radar_score < self.config.watch_exit
        ) or (
            radar.radar_score >= self.config.high_enter
            and camera.camera_score < self.config.watch_exit
        )
        return high_low

    def _reset_unlocked(self) -> None:
        self._last_signature = None
        self._ema = None
        self._stable_state = "UNKNOWN"
        self._candidate_state = "UNKNOWN"
        self._candidate_count = 0

    def reset(self) -> None:
        with self._lock:
            self._stream_key = None
            self._reset_unlocked()


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
                    delta = abs(
                        (camera.source_timestamp - radar.source_timestamp).total_seconds()
                    ) * 1000.0
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
        fusion = response.fusion
        signature = (
            camera.source_timestamp,
            radar.source_timestamp,
            fusion.method,
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
                        "physio_quality": (
                            physio.quality_coverage if physio.enabled else None
                        ),
                        "dynamic_score": response.dynamic_risk.dynamic_risk_score,
                        "short_score": (
                            response.short_term_warning.short_term_fall_score
                            if response.short_term_warning is not None
                            else None
                        ),
                        "event_state": response.fall_event.fall_event_status,
                        "camera_contribution": fusion.contribution_camera,
                        "radar_contribution": fusion.contribution_radar,
                        "dominant": fusion.dominant_modality,
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
        contribution_samples = samples[-30:]
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
            contribution_trend=[
                ContributionTrendPoint(
                    timestamp=sample["timestamp"],
                    camera=sample["camera_contribution"],
                    radar=sample["radar_contribution"],
                    dominant_modality=sample["dominant"],
                )
                for sample in contribution_samples
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
                physiological=(
                    sum(physio_values) / len(physio_values) if physio_values else None
                ),
            ),
            mean_contribution_camera=(
                sum(float(sample["camera_contribution"]) for sample in samples) / count
            ),
            mean_contribution_radar=(
                sum(float(sample["radar_contribution"]) for sample in samples) / count
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
        fusion = response.fusion
        temporal = response.temporal_associated_fusion
        associated = response.associated_risk_augmentation
        camera_led_v2 = response.camera_led_evidence_fusion_v2
        signature = (camera.source_timestamp, radar.source_timestamp, fusion.method)
        with self._lock:
            if signature == self._last_signature:
                return
            self._last_signature = signature
            payload = {
                "schema_version": "real_camera_radar_shadow_v1",
                "logged_at": datetime.now(timezone.utc).isoformat(),
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
                "raw_fusion_score": fusion.raw_fusion_score,
                "fusion_score": (
                    fusion.stable_fusion_score
                    if fusion.stable_fusion_score is not None
                    else fusion.raw_fusion_score
                ),
                "stable_fusion_score": fusion.stable_fusion_score,
                "fusion_method": fusion.method,
                "camera_quality": camera.camera_quality,
                "radar_quality": radar.radar_quality,
                "contribution_camera": fusion.contribution_camera,
                "contribution_radar": fusion.contribution_radar,
                "sync_delta_ms": fusion.sync_delta_ms,
                "degraded_mode": fusion.degraded_mode,
                "fusion_mode": fusion.fusion_mode,
                "fusion_v2_mode": (
                    camera_led_v2.fusion_mode
                    if camera_led_v2 is not None
                    else None
                ),
                "fusion_v2_reason_codes": (
                    camera_led_v2.reason_codes
                    if camera_led_v2 is not None
                    else []
                ),
                "radar_eligibility": fusion.radar_eligibility.model_dump(mode="json"),
                "fusion_state": fusion.fusion_state,
                "stable_fusion_state": fusion.stable_fusion_state,
                "risk_state": {
                    "camera": camera.camera_risk_state,
                    "radar": radar.radar_risk_state,
                    "fusion": fusion.fusion_risk_state,
                },
                "dynamic_risk_score": response.dynamic_risk.dynamic_risk_score,
                "dynamic_risk_level": response.dynamic_risk.risk_level,
                "dynamic_risk_available": response.dynamic_risk.available,
                "dynamic_risk_reasons": [
                    reason.model_dump(mode="json")
                    for reason in response.dynamic_risk.reasons
                ],
                "short_term_fall_score": (
                    response.short_term_warning.short_term_fall_score
                    if response.short_term_warning is not None
                    else None
                ),
                "fall_event_status": response.fall_event.fall_event_status,
                "physiological_evidence": response.physiological_evidence.model_dump(
                    mode="json"
                ),
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
                    "mean_contribution_camera": (
                        response.runtime_statistics.mean_contribution_camera
                    ),
                    "mean_contribution_radar": (
                        response.runtime_statistics.mean_contribution_radar
                    ),
                },
                "reason_codes": fusion.reason_codes,
                "room": radar.room,
                "device": radar.device_id or camera.device_id,
                "model_version": {
                    "camera": camera.model_version,
                    "radar": radar.model_version,
                    "fusion": "decision_fusion_state_v1",
                    "fusion_v2": "camera-led-evidence-fusion-v2-realtime-v1",
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
                "radar_evidence_snapshot": (
                    temporal.radar_evidence_snapshot
                    if temporal is not None
                    else radar.radar_feature
                ),
                "temporal_associated_fusion": (
                    temporal.model_dump(mode="json") if temporal is not None else None
                ),
                "associated_risk_augmentation": (
                    associated.model_dump(mode="json")
                    if associated is not None
                    else None
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
                    if camera_led_v2 is not None
                    else None
                ),
                "alignment": response.alignment.model_dump(mode="json"),
                "shadow_extensions_schema": (
                    "camera_led_evidence_fusion_v2_log_v1"
                ),
            }
            try:
                self._logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                logging.getLogger(__name__).exception("fusion shadow log write failed")


class FusionShadowSampler:
    """Evaluate the formal fixed baseline even when no dashboard is open."""

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
