from fastapi import APIRouter, Request

from app.modules.fall.multimodal_engine.schemas.guard_session import (
    GuardSessionStartRequest,
    MultimodalGuardSessionStatus,
)


router = APIRouter(prefix="/api/guard-session", tags=["guard-session"])


@router.post("/start", response_model=MultimodalGuardSessionStatus)
def start_guard_session(
    payload: GuardSessionStartRequest,
    request: Request,
) -> MultimodalGuardSessionStatus:
    return request.app.state.guard_session_service.start(payload.session_id)


@router.post("/stop", response_model=MultimodalGuardSessionStatus)
def stop_guard_session(request: Request) -> MultimodalGuardSessionStatus:
    return request.app.state.guard_session_service.stop()


@router.get("/status", response_model=MultimodalGuardSessionStatus)
def get_guard_session_status(request: Request) -> MultimodalGuardSessionStatus:
    return request.app.state.guard_session_service.get_status()
