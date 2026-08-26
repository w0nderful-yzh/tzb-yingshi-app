from fastapi import APIRouter, HTTPException, Request, status

from app.modules.fall.multimodal_engine.schemas.fall_live import (
    BrowserFallFrameRequest,
    BrowserFallFrameResponse,
    FallLiveStatusResponse,
)


router = APIRouter(prefix="/api/fall-live", tags=["fall-live"])


@router.get("/status", response_model=FallLiveStatusResponse)
def get_fall_live_status(request: Request) -> FallLiveStatusResponse:
    return request.app.state.fall_live_monitor_service.get_status()


@router.post(
    "/start",
    response_model=FallLiveStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_fall_live_monitor(request: Request) -> FallLiveStatusResponse:
    service = request.app.state.fall_live_monitor_service
    service.start()
    return service.get_status()


@router.post("/stop", response_model=FallLiveStatusResponse)
def stop_fall_live_monitor(request: Request) -> FallLiveStatusResponse:
    service = request.app.state.fall_live_monitor_service
    service.stop()
    return service.get_status()


@router.post(
    "/frames",
    response_model=BrowserFallFrameResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_browser_fall_frame(
    payload: BrowserFallFrameRequest,
    request: Request,
) -> BrowserFallFrameResponse:
    try:
        queue_depth = request.app.state.fall_live_monitor_service.submit_browser_frame(
            device_id=payload.device_id,
            captured_at=payload.captured_at,
            frame_base64=payload.frame_base64,
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return BrowserFallFrameResponse(accepted=True, queue_depth=queue_depth)
