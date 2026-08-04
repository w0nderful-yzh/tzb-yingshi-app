from fastapi import APIRouter, Request

from app.common.responses import ApiResponse
from app.core.config import Settings
from app.core.request_id import get_request_id

router = APIRouter(tags=["system"])


@router.get("/health", response_model=ApiResponse[dict[str, str]])
async def health(request: Request) -> ApiResponse[dict[str, str]]:
    settings: Settings = request.app.state.settings
    return ApiResponse(
        data={
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        },
        request_id=get_request_id(request),
    )
