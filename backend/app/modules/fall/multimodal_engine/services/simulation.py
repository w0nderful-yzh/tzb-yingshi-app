import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.core.config import BACKEND_DIR
from app.modules.fall.multimodal_engine.database.models import RiskEvent
from app.modules.fall.multimodal_engine.schemas.risk_event import RiskEventInput
from app.modules.fall.multimodal_engine.services.monitoring import MonitoringService
from app.modules.fall.multimodal_engine.services.risk_event import RiskEventService


class SimulationScenarioNotFoundError(LookupError):
    pass


class NoActiveMonitoringSessionError(RuntimeError):
    pass


_SCENARIO_DIRECTORY = BACKEND_DIR / "mock" / "scenarios"
_SCENARIO_FILES = {
    "normal": "normal.json",
    "fall-high": "fall_high.json",
    "mental-medium": "mental_medium.json",
    "fraud-high": "fraud_high.json",
}


class SimulationService:
    def __init__(
        self,
        session: Session,
        monitoring_service: MonitoringService | None = None,
        risk_event_service: RiskEventService | None = None,
    ) -> None:
        self.monitoring_service = monitoring_service or MonitoringService(session)
        self.risk_event_service = risk_event_service or RiskEventService(session)

    def trigger_scenario(self, scenario_name: str) -> RiskEvent:
        scenario_path = self._get_scenario_path(scenario_name)
        current_session = self.monitoring_service.get_current_session()
        if current_session is None:
            raise NoActiveMonitoringSessionError(
                "A running monitoring session is required before triggering a scenario"
            )

        scenario_data = self._read_scenario(scenario_path)
        scenario_data.update(
            {
                "event_id": f"sim-{scenario_name}-{uuid4().hex}",
                "session_id": current_session.id,
                "device_id": current_session.device_id,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "source": "SIMULATION",
            }
        )
        payload = RiskEventInput.model_validate(scenario_data)
        return self.risk_event_service.save_event(payload)

    @staticmethod
    def _read_scenario(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _get_scenario_path(scenario_name: str) -> Path:
        filename = _SCENARIO_FILES.get(scenario_name)
        if filename is None:
            raise SimulationScenarioNotFoundError(
                f"Unsupported simulation scenario: {scenario_name}"
            )
        return _SCENARIO_DIRECTORY / filename
