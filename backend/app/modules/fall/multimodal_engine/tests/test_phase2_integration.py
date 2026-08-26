import unittest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import delete

from app.modules.fall.multimodal_engine.database.models import MonitoringSession, RiskEvent
from app.modules.fall.multimodal_engine.database.session import SessionLocal
from app.modules.fall.multimodal_engine.schemas.monitoring import MonitoringSessionCreate
from app.modules.fall.multimodal_engine.schemas.risk_event import (
    EventSource,
    EvidenceItem,
    RiskEventInput,
    RiskEventStatus,
    RiskEventStatusUpdate,
    RiskLevel,
    RiskModule,
)
from app.modules.fall.multimodal_engine.services import DuplicateRiskEventError, MonitoringService, RiskEventService


class Phase2IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid4().hex
        self.session_id = f"test-session-{suffix}"
        self.event_id = f"test-event-{suffix}"
        self.device_id = f"test-device-{suffix}"
        self.session = SessionLocal()

    def tearDown(self) -> None:
        self.session.rollback()
        self.session.execute(delete(RiskEvent).where(RiskEvent.event_id == self.event_id))
        self.session.execute(
            delete(MonitoringSession).where(MonitoringSession.id == self.session_id)
        )
        self.session.commit()
        self.session.close()

    def test_core_risk_event_chain(self) -> None:
        monitoring_service = MonitoringService(self.session)
        risk_event_service = RiskEventService(self.session)

        created_session = monitoring_service.create_session(
            MonitoringSessionCreate(
                session_id=self.session_id,
                device_id=self.device_id,
                enabled_modules=[RiskModule.FALL],
            )
        )
        self.assertEqual(created_session.status, "RUNNING")
        self.assertEqual(
            monitoring_service.get_current_session(self.device_id).id,
            self.session_id,
        )

        event_payload = RiskEventInput(
            schema_version="1.0",
            event_id=self.event_id,
            session_id=self.session_id,
            device_id=self.device_id,
            module=RiskModule.FALL,
            event_type="PRE_FALL_RISK",
            occurred_at=datetime.now(timezone.utc),
            risk_score=0.82,
            risk_level=RiskLevel.HIGH,
            summary="Phase 2 integration test",
            evidence=[
                EvidenceItem(code="test", label="Integration test evidence")
            ],
            model_version="test-only",
            source=EventSource.ALGORITHM,
        )
        saved_event = risk_event_service.save_event(event_payload)
        self.assertEqual(saved_event.status, RiskEventStatus.PENDING.value)

        with self.assertRaises(DuplicateRiskEventError):
            risk_event_service.save_event(event_payload)

        updated_event = risk_event_service.update_event_status(
            self.event_id,
            RiskEventStatusUpdate(
                status=RiskEventStatus.ACKNOWLEDGED,
                handling_note="confirmed by integration test",
            ),
        )
        self.assertEqual(updated_event.status, RiskEventStatus.ACKNOWLEDGED.value)
        self.assertIsNotNone(updated_event.handled_at)


if __name__ == "__main__":
    unittest.main()
