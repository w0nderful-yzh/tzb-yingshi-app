import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.modules.fall.multimodal_engine.database.models import MonitoringSession, RiskEvent
from app.modules.fall.multimodal_engine.database.session import SessionLocal
from app.modules.fall.multimodal_engine.main import app


class Phase4SimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid4().hex
        self.session_id = f"simulation-session-{suffix}"
        self.device_id = f"simulation-device-{suffix}"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        with SessionLocal() as session:
            session.execute(
                delete(RiskEvent).where(RiskEvent.session_id == self.session_id)
            )
            session.execute(
                delete(MonitoringSession).where(MonitoringSession.id == self.session_id)
            )
            session.commit()
        self.client.close()

    def test_all_scenarios_persist_unique_simulation_events(self) -> None:
        create_session_response = self.client.post(
            "/api/monitoring/sessions",
            json={
                "session_id": self.session_id,
                "mode": "SIMULATION",
                "device_id": self.device_id,
                "enabled_modules": ["FALL", "MENTAL_STATE", "FRAUD", "DEVICE"],
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.assertEqual(create_session_response.status_code, 201)

        expected = {
            "normal": ("DEVICE", "LOW"),
            "fall-high": ("FALL", "HIGH"),
            "mental-medium": ("MENTAL_STATE", "MEDIUM"),
            "fraud-high": ("FRAUD", "HIGH"),
        }
        generated_event_ids: set[str] = set()
        for scenario_name, (module, risk_level) in expected.items():
            response = self.client.post(
                f"/api/simulation/scenarios/{scenario_name}"
            )
            self.assertEqual(response.status_code, 201)
            event = response.json()
            self.assertEqual(event["session_id"], self.session_id)
            self.assertEqual(event["source"], "SIMULATION")
            self.assertEqual(event["module"], module)
            self.assertEqual(event["risk_level"], risk_level)
            generated_event_ids.add(event["event_id"])

        repeated_response = self.client.post(
            "/api/simulation/scenarios/fall-high"
        )
        self.assertEqual(repeated_response.status_code, 201)
        generated_event_ids.add(repeated_response.json()["event_id"])
        self.assertEqual(len(generated_event_ids), 5)

        with SessionLocal() as session:
            persisted_events = list(
                session.scalars(
                    select(RiskEvent).where(RiskEvent.session_id == self.session_id)
                )
            )
        self.assertEqual(len(persisted_events), 5)
        self.assertTrue(all(event.source == "SIMULATION" for event in persisted_events))

        stop_response = self.client.post(
            f"/api/monitoring/sessions/{self.session_id}/stop"
        )
        self.assertEqual(stop_response.status_code, 200)

        no_session_response = self.client.post(
            "/api/simulation/scenarios/normal"
        )
        self.assertEqual(no_session_response.status_code, 409)
        self.assertIn("running monitoring session", no_session_response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
