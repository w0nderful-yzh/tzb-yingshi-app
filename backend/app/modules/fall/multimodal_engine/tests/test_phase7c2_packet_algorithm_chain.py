import unittest
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.modules.fall.multimodal_engine.algorithm_runtime import AdapterContext, EventPublisher, RiskEventFactory
from app.modules.fall.multimodal_engine.algorithm_runtime.adapters.mock_radar_risk import MockRadarRiskAdapter
from app.modules.fall.multimodal_engine.algorithm_runtime.adapters.packet_mock_fall import PacketBasedMockFallAdapter
from app.modules.fall.multimodal_engine.data_sources import UnifiedDataPacket
from app.modules.fall.multimodal_engine.data_sources.adapters import DummyRadarAdapter, MockCameraAdapter
from app.modules.fall.multimodal_engine.database.models import MonitoringSession, RiskEvent
from app.modules.fall.multimodal_engine.database.session import SessionLocal
from app.modules.fall.multimodal_engine.main import app


class Phase7C2PacketAlgorithmChainTest(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid4().hex
        self.session_id = f"packet-session-{suffix}"
        self.api_client = TestClient(app)

        def forward_to_fastapi(request: httpx.Request) -> httpx.Response:
            api_response = self.api_client.request(
                request.method,
                request.url.path,
                content=request.content,
                headers={"content-type": "application/json"},
            )
            return httpx.Response(
                status_code=api_response.status_code,
                content=api_response.content,
                headers={"content-type": "application/json"},
            )

        self.http_client = httpx.Client(
            base_url="http://testserver",
            transport=httpx.MockTransport(forward_to_fastapi),
        )
        self.publisher = EventPublisher("http://testserver", client=self.http_client)

    def tearDown(self) -> None:
        self.publisher.close()
        self.http_client.close()
        with SessionLocal() as session:
            session.execute(
                delete(RiskEvent).where(RiskEvent.session_id == self.session_id)
            )
            session.execute(
                delete(MonitoringSession).where(
                    MonitoringSession.id == self.session_id
                )
            )
            session.commit()
        self.api_client.close()

    def test_video_packet_reaches_existing_risk_event_and_dashboard_chain(self) -> None:
        source = MockCameraAdapter(device_id="phase7c2-camera")
        source.start(self.session_id)
        packet = source.read()
        self._create_monitoring_session(packet)

        adapter = PacketBasedMockFallAdapter()
        context = self._start_algorithm(adapter, packet)
        finding = adapter.consume(packet)
        event = RiskEventFactory().create(finding, context)
        published = self.publisher.publish(event)

        self.assertEqual(finding.occurred_at, packet.timestamp)
        self.assertEqual(published.module.value, "FALL")
        self.assertEqual(published.risk_level.value, "HIGH")
        self.assertEqual(published.source.value, "ALGORITHM")
        self.assertEqual(published.device_id, packet.device_id)
        self._assert_persisted_and_visible(published.event_id, "HIGH")

        adapter.stop()
        source.stop()

    def test_radar_packet_reaches_same_existing_risk_event_boundary(self) -> None:
        source = DummyRadarAdapter(device_id="phase7c2-radar")
        source.start(self.session_id)
        packet = source.read()
        self._create_monitoring_session(packet)

        adapter = MockRadarRiskAdapter()
        context = self._start_algorithm(adapter, packet)
        finding = adapter.consume(packet)
        event = RiskEventFactory().create(finding, context)
        published = self.publisher.publish(event)

        self.assertEqual(finding.occurred_at, packet.timestamp)
        self.assertEqual(published.module.value, "FALL")
        self.assertEqual(published.risk_level.value, "MEDIUM")
        self.assertEqual(published.source.value, "ALGORITHM")
        self.assertEqual(published.device_id, packet.device_id)
        self._assert_persisted_and_visible(published.event_id, "MEDIUM")

        adapter.stop()
        source.stop()

    def test_packet_adapters_reject_wrong_modality(self) -> None:
        video_packet = self._packet(modality="VIDEO")
        radar_packet = self._packet(modality="RADAR")

        video_adapter = PacketBasedMockFallAdapter()
        video_adapter.load()
        video_adapter.start(
            AdapterContext(
                session_id=video_packet.session_id,
                device_id=video_packet.device_id,
            )
        )
        with self.assertRaises(ValueError):
            video_adapter.consume(radar_packet)

        radar_adapter = MockRadarRiskAdapter()
        radar_adapter.load()
        radar_adapter.start(
            AdapterContext(
                session_id=radar_packet.session_id,
                device_id=radar_packet.device_id,
            )
        )
        with self.assertRaises(ValueError):
            radar_adapter.consume(video_packet)

    def test_packet_adapter_rejects_context_mismatch(self) -> None:
        packet = self._packet(modality="VIDEO")
        adapter = PacketBasedMockFallAdapter()
        adapter.load()
        adapter.start(
            AdapterContext(
                session_id="different-session",
                device_id=packet.device_id,
            )
        )

        with self.assertRaises(ValueError):
            adapter.consume(packet)

    def _create_monitoring_session(self, packet: UnifiedDataPacket) -> None:
        response = self.api_client.post(
            "/api/monitoring/sessions",
            json={
                "session_id": packet.session_id,
                "mode": "FILE",
                "device_id": packet.device_id,
                "enabled_modules": ["FALL"],
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.assertEqual(response.status_code, 201)

    @staticmethod
    def _start_algorithm(
        adapter: PacketBasedMockFallAdapter | MockRadarRiskAdapter,
        packet: UnifiedDataPacket,
    ) -> AdapterContext:
        context = AdapterContext(
            session_id=packet.session_id,
            device_id=packet.device_id,
        )
        adapter.load()
        adapter.start(context)
        return context

    def _assert_persisted_and_visible(
        self,
        event_id: str,
        expected_level: str,
    ) -> None:
        with SessionLocal() as session:
            persisted = session.scalar(
                select(RiskEvent).where(RiskEvent.event_id == event_id)
            )
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.source, "ALGORITHM")

        response = self.api_client.get("/api/dashboard/summary")
        self.assertEqual(response.status_code, 200)
        fall_state = next(
            item
            for item in response.json()["latest_module_risks"]
            if item["module"] == "FALL"
        )
        self.assertEqual(fall_state["event_id"], event_id)
        self.assertEqual(fall_state["risk_level"], expected_level)

    def _packet(self, *, modality: str) -> UnifiedDataPacket:
        return UnifiedDataPacket(
            packet_id=f"packet-{uuid4().hex}",
            session_id=self.session_id,
            source_id="phase7c2-test",
            device_id="phase7c2-device",
            modality=modality,
            timestamp=datetime.now(timezone.utc),
            data={},
        )


if __name__ == "__main__":
    unittest.main()
