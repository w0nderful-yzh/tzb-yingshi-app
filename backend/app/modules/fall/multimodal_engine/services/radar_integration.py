from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.algorithm_runtime import AdapterContext, RiskEventFactory
from app.modules.fall.multimodal_engine.algorithm_runtime.adapters.radar_risk import RadarRiskAdapter
from app.modules.fall.multimodal_engine.data_sources.adapters.radar_service import RadarServiceDataSourceAdapter
from app.modules.fall.multimodal_engine.data_sources.contracts import UnifiedDataPacket
from app.modules.fall.multimodal_engine.database.session import SessionLocal
from app.modules.fall.multimodal_engine.schemas.radar import (
    RadarCalibratedTcnPredictionPayload,
    RadarDebugPayload,
    RadarEvidencePayload,
    RadarLatestPayload,
    RadarPointNetPredictionPayload,
    RadarSensorMetricsPayload,
    RadarStatusResponse,
    RadarTcnPredictionPayload,
)
from app.modules.fall.multimodal_engine.services.monitoring import MonitoringService
from app.modules.fall.multimodal_engine.services.risk_event import RiskEventService
from app.modules.fall.multimodal_engine.services.camera_radar_alignment import RadarTrackEvidenceBuffer


logger = logging.getLogger(__name__)
SessionFactory = Callable[[], Session]


class RadarIntegrationService:
    """Radar FastAPI到现有风险事件链路的最小后台Runner。"""

    def __init__(
        self,
        source_adapter: RadarServiceDataSourceAdapter,
        *,
        poll_interval_seconds: float = 0.05,
        session_factory: SessionFactory = SessionLocal,
        radar_risk_events_enabled: bool = True,
        allow_formal_predictions: bool = False,
        radar_track_buffer: RadarTrackEvidenceBuffer | None = None,
        session_enabled: bool = True,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.source_adapter = source_adapter
        self.poll_interval_seconds = poll_interval_seconds
        self.session_factory = session_factory
        self.radar_risk_events_enabled = radar_risk_events_enabled
        self.allow_formal_predictions = allow_formal_predictions
        self.radar_track_buffer = radar_track_buffer
        self._session_enabled = session_enabled
        self._session_id: str | None = None
        self._radar_risk_latched = False
        self._status = RadarStatusResponse(online=False)
        self._status_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def session_enabled(self) -> bool:
        with self._status_lock:
            return self._session_enabled

    def enable_for_session(self, session_id: str) -> None:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id must not be blank")
        with self._status_lock:
            self._session_enabled = True
            self._session_id = normalized

    def disable_for_session(self) -> None:
        with self._status_lock:
            self._session_enabled = False
            self._session_id = None
        self._set_offline("RADAR_SESSION_NOT_ENABLED")

    def start(self) -> None:
        if self.is_running:
            return
        self.source_adapter.start("radar-background")
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run,
            name="radar-integration-poller",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(2.0, self.poll_interval_seconds * 2))
        self._worker = None
        self.source_adapter.close()
        self._set_offline(None)

    def get_status(self) -> RadarStatusResponse:
        with self._status_lock:
            return self._status.model_copy(deep=True)

    def process_once(self) -> None:
        """执行一次轮询，便于后台线程和测试复用。"""

        packet = self.source_adapter.read()
        latest = self.source_adapter.latest_payload
        if not self.source_adapter.online or latest is None:
            self._set_offline(self.source_adapter.last_error)
            return

        if not self.session_enabled:
            self._set_offline("RADAR_SESSION_NOT_ENABLED")
            return

        self._set_online(latest)
        if packet is not None:
            self._route_risk_event(packet)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.process_once()
            except Exception as exc:
                logger.exception("Radar integration polling failed")
                self._set_offline(f"{type(exc).__name__}: {exc}")
            self._stop_event.wait(self.poll_interval_seconds)

    def _route_risk_event(self, packet: UnifiedDataPacket) -> None:
        event_packet = self._prepare_event_packet(packet)
        if event_packet is None:
            return
        try:
            with self.session_factory() as db:
                current_session = MonitoringService(db).get_current_session()
                if current_session is None:
                    logger.info(
                        "Radar event ignored because no monitoring session is running"
                    )
                    return

                rebound_packet = event_packet.model_copy(
                    update={"session_id": current_session.id}
                )
                context = AdapterContext(
                    session_id=current_session.id,
                    device_id=rebound_packet.device_id,
                )
                adapter = RadarRiskAdapter(
                    allow_formal_predictions=self.allow_formal_predictions
                )
                adapter.load()
                adapter.start(context)
                try:
                    finding = adapter.consume(rebound_packet)
                finally:
                    adapter.stop()
                if finding is None:
                    return

                event = RiskEventFactory().create(finding, context)
                RiskEventService(db).save_event(event)
                logger.info(
                    "Radar risk event persisted: %s (%s)",
                    event.event_id,
                    event.event_type,
                )
        except SQLAlchemyError:
            logger.exception("Radar risk event could not be persisted")

    def _prepare_event_packet(
        self, packet: UnifiedDataPacket
    ) -> UnifiedDataPacket | None:
        """Select one record-worthy radar event from the latest packet.

        Pre-fall prediction and action risk are separate evidence channels for
        one unified fall-risk incident.  One episode creates at most one
        record.  Hysteresis rearms immediately below the low threshold, so no
        fixed time cooldown delays the next independent risk episode.
        """

        research = packet.data.get("research")
        tcn_prediction = packet.data.get("tcn_prediction")
        pointnet_prediction = packet.data.get("pointnet_prediction")
        if isinstance(tcn_prediction, dict) or isinstance(pointnet_prediction, dict):
            # The deployed learned-model contracts are shadow-only. Their scores are exposed
            # to the dashboard but must never enter the formal event channel.
            return None
        if isinstance(research, dict):
            quality = str(research.get("data_quality", "INSUFFICIENT_DATA"))
            action_score = float(research.get("fall_risk_score", 0.0))
            prediction_score = float(research.get("pre_fall_score", 0.0))
            prediction_state = str(research.get("prediction_state", "UNKNOWN"))
            combined_score = max(action_score, prediction_score)
            trigger_reasons: list[str] = []
            if action_score >= 0.30:
                trigger_reasons.append("ACTION_RISK")
            if prediction_state == "IMMINENT":
                trigger_reasons.append("PREFALL_PREDICTION")
            if quality == "INSUFFICIENT_DATA":
                return None
            if combined_score < 0.20 and prediction_state != "IMMINENT":
                self._radar_risk_latched = False
            if (
                self.radar_risk_events_enabled
                and trigger_reasons
            ):
                if not self._radar_risk_latched:
                    self._radar_risk_latched = True
                    data = dict(packet.data)
                    data.update(
                        {
                            "event_kind": "UNIFIED_FALL_RISK",
                            "trigger_reasons": trigger_reasons,
                            "human_state": "FALL_RISK",
                            "risk_score": combined_score,
                            "event_triggered": True,
                        }
                    )
                    return packet.model_copy(update={"data": data})
                return None

        if bool(packet.data.get("event_triggered", False)):
            return packet
        return None

    def _set_online(
        self,
        latest: (
            RadarLatestPayload
            | RadarTcnPredictionPayload
            | RadarPointNetPredictionPayload
            | RadarCalibratedTcnPredictionPayload
        ),
    ) -> None:
        if isinstance(
            latest,
            (
                RadarTcnPredictionPayload,
                RadarPointNetPredictionPayload,
                RadarCalibratedTcnPredictionPayload,
            ),
        ):
            health = self.source_adapter.latest_health
            tcn_baseline = self.source_adapter.latest_tcn_baseline
            evidence_source = (
                tcn_baseline
                if isinstance(latest, RadarPointNetPredictionPayload)
                and tcn_baseline is not None
                else latest
            )
            evidence = RadarEvidencePayload(
                radar_score=(
                    evidence_source.pre_fall_score
                    if evidence_source.score_valid
                    else None
                ),
                risk_state=(
                    "UNKNOWN"
                    if not evidence_source.score_valid
                    else latest.gate_state
                    if isinstance(latest, RadarCalibratedTcnPredictionPayload)
                    else evidence_source.risk_state
                ),
                timestamp=evidence_source.timestamp,
                room=evidence_source.room,
                device_id=evidence_source.device_id,
                quality=evidence_source.data_quality,
                model_version=evidence_source.model_version,
            )
            debug = RadarDebugPayload(
                descent_prediction=self.source_adapter.latest_descent,
                fall_risk_assessment=self.source_adapter.latest_risk_assessment,
            )
            status = RadarStatusResponse(
                online=True,
                room=latest.room,
                device_id=latest.device_id,
                source_mode=latest.source_mode,
                model_mode=latest.model_mode,
                human_state=None,
                risk_score=None,
                timestamp=latest.timestamp,
                disclaimer=latest.disclaimer,
                research=None,
                tcn_prediction=(
                    latest if isinstance(latest, RadarTcnPredictionPayload) else None
                ),
                pointnet_prediction=(
                    latest if isinstance(latest, RadarPointNetPredictionPayload) else None
                ),
                tcn_baseline=(
                    self.source_adapter.latest_tcn_baseline
                    if isinstance(
                        latest,
                        (RadarPointNetPredictionPayload, RadarCalibratedTcnPredictionPayload),
                    )
                    else None
                ),
                calibrated_tcn_prediction=(
                    latest
                    if isinstance(latest, RadarCalibratedTcnPredictionPayload)
                    else None
                ),
                radar_debug=debug,
                radar_evidence=evidence,
                sensor_metrics=RadarSensorMetricsPayload(
                    frame_rate_hz=(health.frame_rate_hz if health else None),
                    point_count=(health.point_count if health else None),
                ),
                alignment_evidence=self.source_adapter.latest_alignment_evidence,
                error=None,
                checked_at=datetime.now(timezone.utc),
            )
            if self.radar_track_buffer is not None:
                self.radar_track_buffer.observe(status)
            with self._status_lock:
                self._status = status
            return
        risk_score = latest.risk_score
        human_state = latest.human_state
        if latest.research is not None:
            risk_score = max(
                latest.research.pre_fall_score,
                latest.research.fall_risk_score,
            )
            if latest.human_state != "NO_PERSON":
                human_state = (
                    "FALL_RISK"
                    if latest.research.fall_risk_score >= 0.30
                    or latest.research.prediction_state == "IMMINENT"
                    else "NORMAL"
                )
        status = RadarStatusResponse(
            online=True,
            room=latest.room,
            device_id=latest.device_id,
            source_mode=latest.source_mode,
            model_mode=latest.model_mode,
            human_state=human_state,
            risk_score=risk_score,
            timestamp=latest.timestamp,
            disclaimer=latest.disclaimer,
            research=latest.research,
            tcn_prediction=None,
            pointnet_prediction=None,
            tcn_baseline=None,
            radar_evidence=None,
            sensor_metrics=None,
            alignment_evidence=self.source_adapter.latest_alignment_evidence,
            error=None,
            checked_at=datetime.now(timezone.utc),
        )
        if self.radar_track_buffer is not None:
            self.radar_track_buffer.observe(status)
        with self._status_lock:
            self._status = status

    def _set_offline(self, error: str | None) -> None:
        if self.radar_track_buffer is not None:
            self.radar_track_buffer.clear()
        with self._status_lock:
            self._status = RadarStatusResponse(
                online=False,
                error=error,
                checked_at=datetime.now(timezone.utc),
            )
