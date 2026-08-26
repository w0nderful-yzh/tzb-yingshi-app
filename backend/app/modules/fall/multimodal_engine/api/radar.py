from fastapi import APIRouter, Request

from app.modules.fall.multimodal_engine.schemas.radar import RadarStatusResponse


router = APIRouter(prefix="/api/radar", tags=["radar"])


@router.get("/status", response_model=RadarStatusResponse)
def get_radar_status(request: Request) -> RadarStatusResponse:
    return request.app.state.radar_integration_service.get_status()
