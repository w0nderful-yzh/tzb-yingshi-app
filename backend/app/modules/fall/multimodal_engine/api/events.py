from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.database.session import get_db_session
from app.modules.fall.multimodal_engine.schemas.risk_event import (
    RiskEventBulkDeleteRequest,
    RiskEventDeleteResponse,
    RiskEventInput,
    RiskEventResponse,
    RiskEventStatus,
    RiskEventStatusUpdate,
    RiskLevel,
    RiskModule,
)
from app.modules.fall.multimodal_engine.services import (
    DuplicateRiskEventError,
    RiskEventNotFoundError,
    RiskEventService,
    RiskEventSessionNotFoundError,
)


router = APIRouter(tags=["risk-events"])
DatabaseSession = Annotated[Session, Depends(get_db_session)]


@router.post(
    "/api/algorithm/events",
    response_model=RiskEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def receive_algorithm_event(
    payload: RiskEventInput,
    db: DatabaseSession,
) -> RiskEventResponse:
    try:
        risk_event = RiskEventService(db).save_event(payload)
    except DuplicateRiskEventError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RiskEventSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return RiskEventResponse.model_validate(risk_event)


@router.get("/api/events", response_model=list[RiskEventResponse])
def list_risk_events(
    db: DatabaseSession,
    module: RiskModule | None = None,
    risk_level: RiskLevel | None = None,
    status_filter: Annotated[
        RiskEventStatus | None,
        Query(alias="status"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[RiskEventResponse]:
    events = RiskEventService(db).list_events(
        module=module,
        risk_level=risk_level,
        status=status_filter,
        limit=limit,
    )
    return [RiskEventResponse.model_validate(event) for event in events]


@router.post(
    "/api/events/bulk-delete",
    response_model=RiskEventDeleteResponse,
)
def bulk_delete_risk_events(
    payload: RiskEventBulkDeleteRequest,
    db: DatabaseSession,
) -> RiskEventDeleteResponse:
    deleted_count = RiskEventService(db).delete_events(payload.event_ids)
    return RiskEventDeleteResponse(deleted_count=deleted_count)


@router.delete(
    "/api/events",
    response_model=RiskEventDeleteResponse,
)
def clear_risk_events(db: DatabaseSession) -> RiskEventDeleteResponse:
    deleted_count = RiskEventService(db).delete_all_events()
    return RiskEventDeleteResponse(deleted_count=deleted_count)


@router.get("/api/events/{event_id}", response_model=RiskEventResponse)
def get_risk_event(event_id: str, db: DatabaseSession) -> RiskEventResponse:
    try:
        risk_event = RiskEventService(db).get_event(event_id)
    except RiskEventNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return RiskEventResponse.model_validate(risk_event)


@router.delete("/api/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_risk_event(event_id: str, db: DatabaseSession) -> None:
    try:
        RiskEventService(db).delete_event(event_id)
    except RiskEventNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch("/api/events/{event_id}/status", response_model=RiskEventResponse)
def update_risk_event_status(
    event_id: str,
    payload: RiskEventStatusUpdate,
    db: DatabaseSession,
) -> RiskEventResponse:
    try:
        risk_event = RiskEventService(db).update_event_status(event_id, payload)
    except RiskEventNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return RiskEventResponse.model_validate(risk_event)
