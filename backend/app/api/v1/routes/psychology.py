from fastapi import APIRouter, Query, Request

from app.api.dependencies import CurrentIdentity, DatabaseSession
from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.modules.app_client.service import AppClientService
from app.modules.psychology.cognitive.schemas import CognitiveOverview
from app.modules.psychology.cognitive.service import CognitiveOverviewService
from app.modules.psychology.schemas import PsychologyOverview
from app.modules.psychology.service import PsychologyService

router = APIRouter(prefix="/psychology", tags=["psychology"])


@router.get("/overview", response_model=ApiResponse[PsychologyOverview])
async def get_psychology_overview(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[PsychologyOverview]:
    resolver = AppClientService(
        session=session,
        settings=request.app.state.settings,
        live_address_provider=request.app.state.ys7_api_client,
        sdk_credential_provider=request.app.state.ys7_api_client,
    )
    elder = await resolver.resolve_elder(identity, elder_id)
    subject_key = elder.external_subject or str(elder.id)
    service: PsychologyService = request.app.state.psychology_service
    return ApiResponse(
        data=await service.get_overview(subject_key=subject_key),
        request_id=get_request_id(request),
    )


@router.get("/cognitive-overview", response_model=ApiResponse[CognitiveOverview])
async def get_cognitive_overview(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[CognitiveOverview]:
    resolver = AppClientService(
        session=session,
        settings=request.app.state.settings,
        live_address_provider=request.app.state.ys7_api_client,
        sdk_credential_provider=request.app.state.ys7_api_client,
    )
    elder = await resolver.resolve_elder(identity, elder_id)
    subject_key = elder.external_subject or str(elder.id)
    service: CognitiveOverviewService = request.app.state.cognitive_overview_service
    return ApiResponse(
        data=await service.get_overview(subject_key=subject_key),
        request_id=get_request_id(request),
    )
