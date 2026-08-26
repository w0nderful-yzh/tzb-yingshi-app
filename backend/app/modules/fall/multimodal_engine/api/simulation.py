from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.session import get_db_session
from app.modules.fall.multimodal_engine.schemas.risk_event import RiskEventResponse
from app.modules.fall.multimodal_engine.services import (
    NoActiveMonitoringSessionError,
    SimulationScenarioNotFoundError,
    SimulationService,
)


router = APIRouter(prefix="/api/simulation", tags=["simulation"])
DatabaseSession = Annotated[Session, Depends(get_db_session)]


@router.post(
    "/scenarios/{scenario_name}",
    response_model=RiskEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def trigger_simulation_scenario(
    scenario_name: str,
    db: DatabaseSession,
) -> RiskEventResponse:
    try:
        risk_event = SimulationService(db).trigger_scenario(scenario_name)
    except SimulationScenarioNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NoActiveMonitoringSessionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return RiskEventResponse.model_validate(risk_event)
