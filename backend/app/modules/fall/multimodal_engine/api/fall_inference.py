from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.modules.fall.multimodal_engine.core.config import get_settings
from app.modules.fall.multimodal_engine.database.session import get_db_session
from app.modules.fall.multimodal_engine.api.ezviz import get_ezviz_client
from app.modules.fall.multimodal_engine.integrations.ezviz import EzvizApiError, EzvizClient, EzvizConfigurationError
from app.modules.fall.multimodal_engine.integrations.ezviz.live_capture import EzvizLiveCapture
from app.modules.fall.multimodal_engine.schemas.fall_inference import FallInferenceJobResponse, FallInferenceSystemStatus
from app.modules.fall.multimodal_engine.services.fall_inference import (
    FallInferenceBusyError,
    FallInferenceJobNotFoundError,
    FallInferenceJobService,
)
from app.modules.fall.multimodal_engine.services.monitoring import MonitoringService


router = APIRouter(prefix="/api/fall-inference", tags=["fall-inference"])
DatabaseSession = Annotated[Session, Depends(get_db_session)]


@lru_cache
def get_fall_inference_service() -> FallInferenceJobService:
    settings = get_settings()
    return FallInferenceJobService(
        project_dir=settings.fall_inference_project_dir,
        python_executable=settings.fall_inference_python,
        runtime_dir=settings.fall_inference_runtime_dir,
        device=settings.fall_inference_device,
        timeout_seconds=settings.fall_inference_timeout_seconds,
        live_capture=EzvizLiveCapture(
            python_executable=settings.fall_inference_python,
            timeout_seconds=settings.fall_inference_live_capture_timeout_seconds,
        ),
    )


FallService = Annotated[FallInferenceJobService, Depends(get_fall_inference_service)]
EzvizClientDependency = Annotated[EzvizClient, Depends(get_ezviz_client)]


def _require_fall_session(db: Session):
    monitoring_session = MonitoringService(db).get_current_session()
    if monitoring_session is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No running monitoring session",
        )
    if "FALL" not in monitoring_session.enabled_modules:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Current session does not enable the FALL module",
        )
    return monitoring_session


@router.get("/status", response_model=FallInferenceSystemStatus)
def get_fall_inference_status(service: FallService) -> FallInferenceSystemStatus:
    return service.system_status()


@router.post(
    "/jobs",
    response_model=FallInferenceJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_fall_inference_job(
    request: Request,
    background_tasks: BackgroundTasks,
    db: DatabaseSession,
    service: FallService,
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    device_id: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    record_non_alert_test_event: bool = False,
) -> FallInferenceJobResponse:
    readiness = service.system_status()
    if not readiness.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BioSTGCN inference runtime is not ready",
        )
    monitoring_session = _require_fall_session(db)
    try:
        job_id, video_path, job_dir = service.allocate_upload(filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc

    settings = get_settings()
    max_bytes = settings.fall_inference_max_upload_mb * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="video is too large")
    size = 0
    try:
        with video_path.open("wb") as stream:
            async for chunk in request.stream():
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="video is too large",
                    )
                stream.write(chunk)
        if size == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="video is empty")
        response = service.create_job(
            job_id=job_id,
            session_id=monitoring_session.id,
            device_id=device_id or monitoring_session.device_id,
            filename=Path(filename).name,
            video_path=video_path,
            job_dir=job_dir,
            record_non_alert_test_event=record_non_alert_test_event,
        )
    except FallInferenceBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    background_tasks.add_task(service.run_job, job_id)
    return response


@router.post(
    "/ezviz-live-jobs",
    response_model=FallInferenceJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_ezviz_live_fall_inference_job(
    background_tasks: BackgroundTasks,
    db: DatabaseSession,
    service: FallService,
    ezviz_client: EzvizClientDependency,
    device_id: Annotated[str, Query(min_length=1, max_length=64)],
    channel_no: Annotated[int, Query(ge=1)] = 1,
    capture_seconds: Annotated[int | None, Query(ge=6, le=60)] = None,
    record_non_alert_test_event: bool = False,
) -> FallInferenceJobResponse:
    readiness = service.system_status()
    if not readiness.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BioSTGCN inference runtime is not ready",
        )
    if not readiness.ezviz_live_capture_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EZVIZ live capture runtime is not ready",
        )
    monitoring_session = _require_fall_session(db)
    settings = get_settings()
    duration = capture_seconds or settings.fall_inference_live_capture_seconds
    try:
        stream = ezviz_client.get_standard_live_address(
            device_id,
            channel_no=channel_no,
            protocol=settings.fall_inference_live_protocol,
            quality=settings.fall_inference_live_quality,
            expire_seconds=settings.fall_inference_live_address_expire_seconds,
        )
    except EzvizConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EZVIZ AppKey/AppSecret are not configured on the backend",
        ) from exc
    except EzvizApiError as exc:
        detail = "EZVIZ standard live stream is unavailable"
        if exc.code is not None:
            detail = f"{detail} (platform code: {exc.code})"
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc

    job_id, video_path, job_dir = service.allocate_live_capture()
    try:
        response = service.create_job(
            job_id=job_id,
            session_id=monitoring_session.id,
            device_id=device_id,
            filename=f"ezviz-live-{duration}s.mp4",
            video_path=video_path,
            job_dir=job_dir,
            record_non_alert_test_event=record_non_alert_test_event,
        )
    except FallInferenceBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    background_tasks.add_task(
        service.run_live_job,
        job_id,
        stream_url=stream.play_url,
        capture_seconds=duration,
    )
    return response


@router.get("/jobs/{job_id}", response_model=FallInferenceJobResponse)
def get_fall_inference_job(
    job_id: str,
    service: FallService,
) -> FallInferenceJobResponse:
    try:
        return service.get_job(job_id)
    except FallInferenceJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found") from exc
