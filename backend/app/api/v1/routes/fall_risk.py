from fastapi import APIRouter, Query, Request

from app.api.dependencies import CurrentIdentity, DatabaseSession
from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.modules.app_client.service import AppClientService
from app.modules.fall.schemas import FallRiskOverview
from app.modules.fall.service import FallRiskService

router = APIRouter(prefix="/fall-risk", tags=["fall-risk"])


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
