import unittest
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.modules.fall.multimodal_engine.algorithm_runtime import (
    AdapterContext,
    AdapterState,
    EventPublishError,
    EventPublisher,
    RiskEventFactory,
)
from app.modules.fall.multimodal_engine.algorithm_runtime.adapters import (
    MockFallAdapter,
    MockFraudAdapter,
    MockMentalAdapter,
)
from app.modules.fall.multimodal_engine.database.models import MonitoringSession, RiskEvent
from app.modules.fall.multimodal_engine.database.session import SessionLocal
from app.modules.fall.multimodal_engine.main import app


class Phase7BAlgorithmRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid4().hex
        self.session_id = f"algorithm-session-{suffix}"
        self.device_id = f"algorithm-device-{suffix}"
        self.api_client = TestClient(app)

        create_response = self.api_client.post(
            "/api/monitoring/sessions",
            json={
                "session_id": self.session_id,
                "mode": "FILE",
                "device_id": self.device_id,
                "enabled_modules": ["FALL", "MENTAL_STATE", "FRAUD"],
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.assertEqual(create_response.status_code, 201)

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
        self.publisher = EventPublisher(
            "http://testserver",
            client=self.http_client,
        )

    def tearDown(self) -> None:
        self.publisher.close()
        self.http_client.close()
        with SessionLocal() as session:
            session.execute(
                delete(RiskEvent).where(RiskEvent.session_id == self.session_id)
            )
            session.execute(
                delete(MonitoringSession).where(MonitoringSession.id == self.session_id)
            )
            session.commit()
        self.api_client.close()

    def test_mock_adapter_event_reaches_mysql_and_dashboard(self) -> None:
        context = AdapterContext(
            session_id=self.session_id,
            device_id=self.device_id,
        )
        adapter = MockFallAdapter()

        self.assertEqual(adapter.health(), AdapterState.CREATED)
        adapter.load()
        adapter.start(context)
        self.assertEqual(adapter.health(), AdapterState.RUNNING)

        finding = adapter.consume(None)
        event = RiskEventFactory().create(finding, context)
        published = self.publisher.publish(event)

        self.assertEqual(published.event_id, event.event_id)
        self.assertEqual(published.session_id, self.session_id)
        self.assertEqual(published.device_id, self.device_id)
        self.assertEqual(published.module.value, "FALL")
        self.assertEqual(published.source.value, "ALGORITHM")
        self.assertEqual(published.risk_level.value, "HIGH")

        with SessionLocal() as session:
            persisted = session.scalar(
                select(RiskEvent).where(RiskEvent.event_id == event.event_id)
            )
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.source, "ALGORITHM")
        self.assertEqual(persisted.model_version, adapter.model_version)

        dashboard_response = self.api_client.get("/api/dashboard/summary")
        self.assertEqual(dashboard_response.status_code, 200)
        dashboard = dashboard_response.json()
        fall_state = next(
            item
            for item in dashboard["latest_module_risks"]
            if item["module"] == "FALL"
        )
        self.assertEqual(fall_state["event_id"], event.event_id)
        self.assertEqual(fall_state["risk_level"], "HIGH")
        self.assertEqual(dashboard["highest_risk_level"], "HIGH")

        detail_response = self.api_client.get(f"/api/events/{event.event_id}")
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()["event_id"], event.event_id)
        self.assertEqual(detail_response.json()["status"], "PENDING")

        acknowledge_response = self.api_client.patch(
            f"/api/events/{event.event_id}/status",
            json={
                "status": "ACKNOWLEDGED",
                "handling_note": "Phase 7-B系统级联调确认",
            },
        )
        self.assertEqual(acknowledge_response.status_code, 200)
        self.assertEqual(acknowledge_response.json()["status"], "ACKNOWLEDGED")
        self.assertIsNotNone(acknowledge_response.json()["handled_at"])

        confirmed_detail = self.api_client.get(f"/api/events/{event.event_id}")
        self.assertEqual(confirmed_detail.status_code, 200)
        self.assertEqual(confirmed_detail.json()["status"], "ACKNOWLEDGED")

        adapter.stop()
        self.assertEqual(adapter.health(), AdapterState.STOPPED)

    def test_three_mock_adapters_reach_mysql_and_dashboard(self) -> None:
        context = AdapterContext(
            session_id=self.session_id,
            device_id=self.device_id,
        )
        cases = (
            (MockFallAdapter, "FALL", "HIGH"),
            (MockMentalAdapter, "MENTAL_STATE", "MEDIUM"),
            (MockFraudAdapter, "FRAUD", "HIGH"),
        )
        published_event_ids: dict[str, str] = {}

        for adapter_class, expected_module, expected_level in cases:
            with self.subTest(module=expected_module):
                adapter = adapter_class()
                adapter.load()
                adapter.start(context)

                finding = adapter.consume(None)
                event = RiskEventFactory().create(finding, context)
                published = self.publisher.publish(event)

                self.assertEqual(published.module.value, expected_module)
                self.assertEqual(published.risk_level.value, expected_level)
                self.assertEqual(published.source.value, "ALGORITHM")
                self.assertEqual(published.model_version, adapter.model_version)
                published_event_ids[expected_module] = published.event_id

                adapter.stop()
                self.assertEqual(adapter.health(), AdapterState.STOPPED)

        self.assertEqual(len(set(published_event_ids.values())), 3)

        with SessionLocal() as session:
            persisted_events = list(
                session.scalars(
                    select(RiskEvent).where(RiskEvent.session_id == self.session_id)
                )
            )
        self.assertEqual(len(persisted_events), 3)
        self.assertTrue(all(event.source == "ALGORITHM" for event in persisted_events))
        self.assertEqual(
            {event.module: event.risk_level for event in persisted_events},
            {
                "FALL": "HIGH",
                "MENTAL_STATE": "MEDIUM",
                "FRAUD": "HIGH",
            },
        )

        dashboard_response = self.api_client.get("/api/dashboard/summary")
        self.assertEqual(dashboard_response.status_code, 200)
        dashboard = dashboard_response.json()
        module_states = {
            item["module"]: item for item in dashboard["latest_module_risks"]
        }
        for _, expected_module, expected_level in cases:
            with self.subTest(dashboard_module=expected_module):
                self.assertEqual(
                    module_states[expected_module]["event_id"],
                    published_event_ids[expected_module],
                )
                self.assertEqual(
                    module_states[expected_module]["risk_level"],
                    expected_level,
                )
        self.assertEqual(dashboard["highest_risk_level"], "HIGH")
        self.assertEqual(dashboard["recent_event_count"], 3)

    def test_publisher_surfaces_existing_duplicate_check(self) -> None:
        context = AdapterContext(
            session_id=self.session_id,
            device_id=self.device_id,
        )
        adapter = MockFallAdapter()
        adapter.load()
        adapter.start(context)
        event = RiskEventFactory(
            event_id_factory=lambda finding: f"fixed-{self.session_id}"
        ).create(adapter.consume(None), context)

        self.publisher.publish(event)
        with self.assertRaises(EventPublishError) as raised:
            self.publisher.publish(event)

        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("already exists", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
