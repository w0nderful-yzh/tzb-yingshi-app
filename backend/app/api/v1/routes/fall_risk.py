from typing import Annotated

from fastapi import APIRouter, Header, Query, Request

from app.api.dependencies import CurrentIdentity, DatabaseSession
from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.modules.app_client.service import AppClientService
from app.modules.fall.schemas import CameraMonitoringStatus, FallRiskOverview
from app.modules.fall.service import FallRiskService

router = APIRouter(prefix="/fall-risk", tags=["fall-risk"])
IdempotencyHeader = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=8, max_length=128),
]


@router.get("/overview", response_model=ApiResponse[FallRiskOverview])
async def get_fall_risk_overview(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[FallRiskOverview]:
    resolver = AppClientService(
        session=session,
        settings=request.app.state.settings,
        live_address_provider=request.app.state.ys7_api_client,
        sdk_credential_provider=request.app.state.ys7_api_client,
    )
    elder = await resolver.resolve_elder(identity, elder_id)
    external_elder_id = elder.external_subject or str(elder.id)
    service: FallRiskService = request.app.state.fall_risk_service
    return ApiResponse(
        data=await service.get_overview(elder_id=external_elder_id),
        request_id=get_request_id(request),
    )


@router.post(
    "/camera-monitoring/start",
    response_model=ApiResponse[CameraMonitoringStatus],
)
async def start_camera_monitoring(
    request: Request,
    identity: CurrentIdentity,
    idempotency_key: IdempotencyHeader,
) -> ApiResponse[CameraMonitoringStatus]:
    del identity, idempotency_key
    service: FallRiskService = request.app.state.fall_risk_service
    return ApiResponse(
        data=await service.start_camera_monitoring(),
        request_id=get_request_id(request),
    )


@router.post(
    "/camera-monitoring/stop",
    response_model=ApiResponse[CameraMonitoringStatus],
)
async def stop_camera_monitoring(
    request: Request,
    identity: CurrentIdentity,
    idempotency_key: IdempotencyHeader,
) -> ApiResponse[CameraMonitoringStatus]:
    del identity, idempotency_key
    service: FallRiskService = request.app.state.fall_risk_service
    return ApiResponse(
        data=await service.stop_camera_monitoring(),
        request_id=get_request_id(request),
    )


@router.get(
    "/camera-monitoring/status",
    response_model=ApiResponse[CameraMonitoringStatus],
)
async def get_camera_monitoring_status(
    request: Request,
    identity: CurrentIdentity,
) -> ApiResponse[CameraMonitoringStatus]:
    del identity
    service: FallRiskService = request.app.state.fall_risk_service
    return ApiResponse(
        data=await service.get_camera_monitoring_status(),
        request_id=get_request_id(request),
    )
