import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.modules.fall.multimodal_engine.database.models import MonitoringSession, RiskEvent
from app.modules.fall.multimodal_engine.database.session import SessionLocal
from app.modules.fall.multimodal_engine.main import app


class Phase3ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid4().hex
        self.session_id = f"api-session-{suffix}"
        self.event_id = f"api-event-{suffix}"
        self.device_id = f"api-device-{suffix}"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        with SessionLocal() as session:
            session.execute(delete(RiskEvent).where(RiskEvent.event_id == self.event_id))
            session.execute(
                delete(MonitoringSession).where(MonitoringSession.id == self.session_id)
            )
            session.commit()
        self.client.close()

    def test_minimum_http_api_chain(self) -> None:
        now = datetime.now(timezone.utc).isoformat()

        create_session_response = self.client.post(
            "/api/monitoring/sessions",
            json={
                "session_id": self.session_id,
                "mode": "SIMULATION",
                "device_id": self.device_id,
                "enabled_modules": ["FALL", "MENTAL_STATE", "FRAUD"],
                "started_at": now,
            },
        )
        self.assertEqual(create_session_response.status_code, 201)
        self.assertEqual(create_session_response.json()["status"], "RUNNING")

        current_session_response = self.client.get(
            "/api/monitoring/sessions/current"
        )
        self.assertEqual(current_session_response.status_code, 200)
        self.assertEqual(
            current_session_response.json()["session_id"], self.session_id
        )

        event_payload = {
            "schema_version": "1.0",
            "event_id": self.event_id,
            "session_id": self.session_id,
            "device_id": self.device_id,
            "module": "FALL",
            "event_type": "PRE_FALL_RISK",
            "occurred_at": now,
            "risk_score": 0.82,
            "risk_level": "HIGH",
            "summary": "Phase 3 API integration test",
            "evidence": [
                {
                    "code": "test",
                    "label": "API integration test evidence",
                    "value": 0.82,
                }
            ],
            "recommended_action": "Verify the event",
            "model_version": "test-only",
            "source": "ALGORITHM",
        }
        create_event_response = self.client.post(
            "/api/algorithm/events",
            json=event_payload,
        )
        self.assertEqual(create_event_response.status_code, 201)
        self.assertEqual(create_event_response.json()["status"], "PENDING")

        duplicate_event_response = self.client.post(
            "/api/algorithm/events",
            json=event_payload,
        )
        self.assertEqual(duplicate_event_response.status_code, 409)

        list_response = self.client.get(
            "/api/events",
            params={
                "module": "FALL",
                "risk_level": "HIGH",
                "status": "PENDING",
                "limit": 10,
            },
        )
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()), 1)
        self.assertEqual(list_response.json()[0]["event_id"], self.event_id)

        detail_response = self.client.get(f"/api/events/{self.event_id}")
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()["event_id"], self.event_id)

        status_response = self.client.patch(
            f"/api/events/{self.event_id}/status",
            json={
                "status": "ACKNOWLEDGED",
                "handling_note": "confirmed by API test",
            },
        )
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["status"], "ACKNOWLEDGED")

        dashboard_response = self.client.get("/api/dashboard/summary")
        self.assertEqual(dashboard_response.status_code, 200)
        dashboard = dashboard_response.json()
        self.assertEqual(dashboard["current_session"]["session_id"], self.session_id)
        self.assertEqual(dashboard["highest_risk_level"], "HIGH")
        self.assertEqual(dashboard["recent_event_count"], 1)
        self.assertEqual(len(dashboard["latest_module_risks"]), 3)

        stop_response = self.client.post(
            f"/api/monitoring/sessions/{self.session_id}/stop"
        )
        self.assertEqual(stop_response.status_code, 200)
        self.assertEqual(stop_response.json()["status"], "STOPPED")

        delete_response = self.client.delete(f"/api/events/{self.event_id}")
        self.assertEqual(delete_response.status_code, 204)
        self.assertEqual(self.client.get(f"/api/events/{self.event_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
