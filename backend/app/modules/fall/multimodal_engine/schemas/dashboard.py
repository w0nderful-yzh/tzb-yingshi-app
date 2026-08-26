from datetime import datetime

from pydantic import BaseModel

from app.modules.fall.multimodal_engine.schemas.monitoring import MonitoringSessionResponse
from app.modules.fall.multimodal_engine.schemas.risk_event import RiskLevel, RiskModule


class ModuleRiskSummary(BaseModel):
    module: RiskModule
    event_id: str | None = None
    risk_level: RiskLevel | None = None
    risk_score: float | None = None
    occurred_at: datetime | None = None


class DashboardSummaryResponse(BaseModel):
    current_session: MonitoringSessionResponse | None
    latest_module_risks: list[ModuleRiskSummary]
    highest_risk_level: RiskLevel | None
    recent_event_count: int
