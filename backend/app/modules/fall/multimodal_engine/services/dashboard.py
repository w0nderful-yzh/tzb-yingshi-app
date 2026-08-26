from datetime import timezone

from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.repositories import MonitoringRepository, RiskEventRepository
from app.modules.fall.multimodal_engine.schemas.dashboard import DashboardSummaryResponse, ModuleRiskSummary
from app.modules.fall.multimodal_engine.schemas.monitoring import MonitoringSessionResponse
from app.modules.fall.multimodal_engine.schemas.risk_event import RiskLevel, RiskModule


_DASHBOARD_MODULES = (
    RiskModule.FALL,
    RiskModule.MENTAL_STATE,
    RiskModule.FRAUD,
)
_RISK_PRIORITY = {
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
}


class DashboardService:
    def __init__(
        self,
        session: Session,
        monitoring_repository: MonitoringRepository | None = None,
        risk_event_repository: RiskEventRepository | None = None,
    ) -> None:
        self.monitoring_repository = monitoring_repository or MonitoringRepository(session)
        self.risk_event_repository = risk_event_repository or RiskEventRepository(session)

    def get_summary(self) -> DashboardSummaryResponse:
        current_session = self.monitoring_repository.get_current()
        if current_session is None:
            return DashboardSummaryResponse(
                current_session=None,
                latest_module_risks=[
                    ModuleRiskSummary(module=module) for module in _DASHBOARD_MODULES
                ],
                highest_risk_level=None,
                recent_event_count=0,
            )

        module_risks: list[ModuleRiskSummary] = []
        risk_levels: list[RiskLevel] = []
        for module in _DASHBOARD_MODULES:
            event = self.risk_event_repository.get_latest_for_module(
                session_id=current_session.id,
                module=module.value,
            )
            if event is None:
                module_risks.append(ModuleRiskSummary(module=module))
                continue

            risk_level = RiskLevel(event.risk_level)
            risk_levels.append(risk_level)
            occurred_at = event.occurred_at.replace(tzinfo=timezone.utc)
            module_risks.append(
                ModuleRiskSummary(
                    module=module,
                    event_id=event.event_id,
                    risk_level=risk_level,
                    risk_score=float(event.risk_score),
                    occurred_at=occurred_at,
                )
            )

        highest_risk_level = (
            max(risk_levels, key=_RISK_PRIORITY.__getitem__)
            if risk_levels
            else None
        )
        return DashboardSummaryResponse(
            current_session=MonitoringSessionResponse.model_validate(current_session),
            latest_module_risks=module_risks,
            highest_risk_level=highest_risk_level,
            recent_event_count=self.risk_event_repository.count_by_session(
                current_session.id
            ),
        )
