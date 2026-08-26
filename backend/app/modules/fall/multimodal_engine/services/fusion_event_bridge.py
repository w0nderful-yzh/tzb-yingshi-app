from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable

from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.algorithm_runtime import AdapterContext, AlgorithmFinding, RiskEventFactory
from app.modules.fall.multimodal_engine.database.session import SessionLocal
from app.modules.fall.multimodal_engine.schemas.monitoring import MonitoringMode, MonitoringSessionCreate
from app.modules.fall.multimodal_engine.schemas.multimodal import MultimodalLatestResponse
from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule
from app.modules.fall.multimodal_engine.services.monitoring import MonitoringService
from app.modules.fall.multimodal_engine.services.risk_event import RiskEventService


logger = logging.getLogger(__name__)
SessionFactory = Callable[[], Session]


@dataclass(frozen=True, slots=True)
class FusionFinding:
    """Auditable bridge object between Fusion and the existing event factory."""

    finding: AlgorithmFinding
    device_id: str
    room: str | None


class FusionFindingFactory:
    model_version = "decision-fusion-state-v1"

    def create(self, response: MultimodalLatestResponse) -> FusionFinding | None:
        fusion = response.fusion
        if response.operating_mode != "LIVE_CAMERA_RADAR":
            return None
        if fusion.method != "fixed_weighted":
            return None
        if fusion.stable_fusion_state not in {"HIGH", "IMMINENT"}:
            return None
        if fusion.stable_fusion_score is None:
            return None
        # A degraded or conflicting result remains observable in shadow logs,
        # but is never promoted into a formal event.
        if fusion.degraded_mode != "NONE" or not fusion.synchronized:
            return None
        device_id = response.radar.device_id or response.camera.device_id
        if not device_id:
            return None
        evidence = [
            EvidenceItem(
                code="camera_score",
                label="摄像头风险分数",
                value=response.camera.camera_score,
            ),
            EvidenceItem(
                code="radar_score",
                label="毫米波TCN风险分数",
                value=response.radar.radar_score,
            ),
            EvidenceItem(
                code="raw_fusion_score",
                label="原始多模态风险分数",
                value=fusion.raw_fusion_score,
            ),
            EvidenceItem(
                code="stable_fusion_state",
                label="稳定融合状态",
                value=fusion.stable_fusion_state,
            ),
            EvidenceItem(
                code="sync_delta_ms",
                label="模态同步时间差",
                value=fusion.sync_delta_ms,
                unit="ms",
            ),
        ]
        finding = AlgorithmFinding(
            module=RiskModule.FALL,
            event_type="MULTIMODAL_PRE_FALL_RISK",
            occurred_at=max(
                response.camera.source_timestamp,
                response.radar.source_timestamp,
            ),
            risk_score=fusion.stable_fusion_score,
            risk_level=RiskLevel.HIGH,
            summary="摄像头与毫米波证据经稳定状态机确认达到多模态风险阈值",
            evidence=evidence,
            recommended_action="立即查看实时画面并确认老人状态",
            model_version=self.model_version,
        )
        return FusionFinding(
            finding=finding,
            device_id=device_id,
            room=response.radar.room,
        )


class FusionRiskEventBridge:
    """Persist FusionFinding only when the separately guarded switch is enabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        cooldown_seconds: float = 30.0,
        auto_create_session: bool = True,
        session_factory: SessionFactory = SessionLocal,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.auto_create_session = auto_create_session
        self.session_factory = session_factory
        self.finding_factory = FusionFindingFactory()
        self._last_persisted_monotonic = 0.0
        self._last_signature: tuple[object, ...] | None = None
        self._lock = threading.Lock()

    def handle(self, response: MultimodalLatestResponse) -> None:
        if not self.enabled:
            return
        fusion_finding = self.finding_factory.create(response)
        if fusion_finding is None:
            return
        signature = (
            response.camera.source_timestamp,
            response.radar.source_timestamp,
            response.fusion.stable_fusion_state,
        )
        with self._lock:
            now = time.monotonic()
            if signature == self._last_signature:
                return
            if now - self._last_persisted_monotonic < self.cooldown_seconds:
                return
            try:
                self._persist(fusion_finding)
            except Exception:
                logger.exception("FusionFinding persistence failed; fusion output remains shadow-only")
                return
            self._last_signature = signature
            self._last_persisted_monotonic = now

    def _persist(self, fusion_finding: FusionFinding) -> None:
        with self.session_factory() as db:
            monitoring = MonitoringService(db)
            session = monitoring.get_current_session(device_id=fusion_finding.device_id)
            if session is None and self.auto_create_session:
                session = monitoring.create_session(
                    MonitoringSessionCreate(
                        mode=MonitoringMode.LIVE,
                        device_id=fusion_finding.device_id,
                        enabled_modules=[RiskModule.FALL],
                    )
                )
            if session is None:
                logger.warning("FusionFinding ignored because no monitoring session is active")
                return
            context = AdapterContext(
                session_id=session.id,
                device_id=fusion_finding.device_id,
            )
            event = RiskEventFactory().create(fusion_finding.finding, context)
            RiskEventService(db).save_event(event)
