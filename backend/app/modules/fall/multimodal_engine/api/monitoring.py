from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.session import get_db_session
from app.modules.fall.multimodal_engine.schemas.monitoring import MonitoringSessionCreate, MonitoringSessionResponse
from app.modules.fall.multimodal_engine.services import (
    MonitoringService,
    MonitoringSessionAlreadyExistsError,
    MonitoringSessionNotFoundError,
)


router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])
DatabaseSession = Annotated[Session, Depends(get_db_session)]


@router.post(
    "/sessions",
    response_model=MonitoringSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_monitoring_session(
    payload: MonitoringSessionCreate,
    db: DatabaseSession,
) -> MonitoringSessionResponse:
    try:
        monitoring_session = MonitoringService(db).create_session(payload)
    except MonitoringSessionAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return MonitoringSessionResponse.model_validate(monitoring_session)


@router.get("/sessions/current", response_model=MonitoringSessionResponse)
def get_current_monitoring_session(db: DatabaseSession) -> MonitoringSessionResponse:
    monitoring_session = MonitoringService(db).get_current_session()
    if monitoring_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No running monitoring session",
        )
    return MonitoringSessionResponse.model_validate(monitoring_session)


@router.post(
    "/sessions/{session_id}/stop",
    response_model=MonitoringSessionResponse,
)
def stop_monitoring_session(
    session_id: str,
    db: DatabaseSession,
) -> MonitoringSessionResponse:
    try:
        monitoring_session = MonitoringService(db).stop_session(session_id)
    except MonitoringSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return MonitoringSessionResponse.model_validate(monitoring_session)
