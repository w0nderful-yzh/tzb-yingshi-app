from typing import Annotated

from fastapi import APIRouter, Header, Query, Request

from app.api.dependencies import CurrentIdentity, DatabaseSession
from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.modules.app_client.service import AppClientService
from app.modules.guarding.schemas import GuardianSessionStatus
from app.modules.guarding.service import GuardianSessionService

router = APIRouter(prefix="/guard-session", tags=["guard-session"])
IdempotencyHeader = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=8, max_length=128),
]


@router.post("/start", response_model=ApiResponse[GuardianSessionStatus])
async def start_guard_session(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    idempotency_key: IdempotencyHeader,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[GuardianSessionStatus]:
    del idempotency_key
    resolver = AppClientService(
        session=session,
        settings=request.app.state.settings,
        live_address_provider=request.app.state.ys7_api_client,
        sdk_credential_provider=request.app.state.ys7_api_client,
    )
    elder = await resolver.resolve_elder(identity, elder_id)
    subject_key = elder.external_subject or str(elder.id)
    service: GuardianSessionService = request.app.state.guardian_session_service
    return ApiResponse(
        data=await service.start(subject_key=subject_key),
        request_id=get_request_id(request),
    )


@router.post("/stop", response_model=ApiResponse[GuardianSessionStatus])
async def stop_guard_session(
    request: Request,
    identity: CurrentIdentity,
    idempotency_key: IdempotencyHeader,
) -> ApiResponse[GuardianSessionStatus]:
    del identity, idempotency_key
    service: GuardianSessionService = request.app.state.guardian_session_service
    return ApiResponse(
        data=await service.stop(),
        request_id=get_request_id(request),
    )


@router.get("/status", response_model=ApiResponse[GuardianSessionStatus])
async def get_guard_session_status(
    request: Request,
    identity: CurrentIdentity,
) -> ApiResponse[GuardianSessionStatus]:
    del identity
    service: GuardianSessionService = request.app.state.guardian_session_service
    return ApiResponse(
        data=await service.get_status(),
        request_id=get_request_id(request),
    )
