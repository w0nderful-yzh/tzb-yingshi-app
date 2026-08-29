from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import CurrentIdentity
from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.infrastructure.database.session import get_database_session
from app.infrastructure.external.ys7.api_client import Ys7ApiError, Ys7LiveAddressProvider
from app.infrastructure.external.ys7.pcm_relay import AppPcmRelaySource
from app.modules.app_client.schemas import (
    ActivityData,
    ConfirmRequest,
    ContactsData,
    DeviceListData,
    EldersData,
    EmptyData,
    EventDetailData,
    EventListData,
    EventsStatsData,
    HistoryPlaybackData,
    InterventionReminderRequest,
    LiveSdkSessionData,
    LiveUrlData,
    SafetyStatus,
    SosRequest,
    SosResult,
    StatusPatchRequest,
    UserInfo,
)
from app.modules.app_client.service import AppClientService
from app.modules.psychology.cognitive.collector import CognitiveAudioCollector

router = APIRouter(tags=["app-client"])

DatabaseSession = Annotated[AsyncSession, Depends(get_database_session)]
IdempotencyHeader = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=8, max_length=128),
]
HistoryAtQuery = Annotated[datetime | None, Query()]
HistoryDurationQuery = Annotated[int, Query(ge=10, le=300)]


def _service(request: Request, session: AsyncSession) -> AppClientService:
    provider: Ys7LiveAddressProvider = request.app.state.ys7_api_client
    return AppClientService(
        session=session,
        settings=request.app.state.settings,
        live_address_provider=provider,
        sdk_credential_provider=request.app.state.ys7_api_client,
    )


@router.get("/users/me", response_model=ApiResponse[UserInfo])
async def get_me(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
) -> ApiResponse[UserInfo]:
    service = _service(request, session)
    return ApiResponse(data=await service.get_me(identity), request_id=get_request_id(request))


@router.get("/safety/status", response_model=ApiResponse[SafetyStatus])
async def get_safety_status(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[SafetyStatus]:
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    return ApiResponse(
        data=await service.get_safety_status(elder),
        request_id=get_request_id(request),
    )


@router.post(
    "/sos",
    response_model=ApiResponse[SosResult],
    status_code=status.HTTP_201_CREATED,
)
async def create_sos(
    request: Request,
    payload: SosRequest,
    session: DatabaseSession,
    idempotency_key: IdempotencyHeader,
    identity: CurrentIdentity,
) -> ApiResponse[SosResult]:
    service = _service(request, session)
    if identity.role != "elder":
        raise HTTPException(status_code=403, detail="elder role required")
    elder = await service.resolve_elder(identity, None)
    result = await service.create_sos(elder, payload, idempotency_key)
    return ApiResponse(data=result, request_id=get_request_id(request))


@router.get("/events", response_model=ApiResponse[EventListData])
async def list_events(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
    level: str | None = Query(default=None),
    event_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> ApiResponse[EventListData]:
    del cursor
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    result = await service.list_events(
        elder,
        level=level,
        status=event_status,
        limit=limit,
    )
    return ApiResponse(data=result, request_id=get_request_id(request))


@router.get("/events/{event_id}", response_model=ApiResponse[EventDetailData])
async def get_event(
    request: Request,
    event_id: str,
    session: DatabaseSession,
    identity: CurrentIdentity,
) -> ApiResponse[EventDetailData]:
    service = _service(request, session)
    result = await service.get_event(identity, event_id)
    return ApiResponse(data=result, request_id=get_request_id(request))


@router.post("/events/{event_id}/confirm", response_model=ApiResponse[EmptyData])
async def confirm_event(
    request: Request,
    event_id: str,
    payload: ConfirmRequest,
    session: DatabaseSession,
    idempotency_key: IdempotencyHeader,
    identity: CurrentIdentity,
) -> ApiResponse[EmptyData]:
    service = _service(request, session)
    result = await service.confirm_event(
        identity,
        event_id,
        action=payload.action,
        version=payload.version,
        idempotency_key=idempotency_key,
    )
    return ApiResponse(data=result, request_id=get_request_id(request))


@router.patch("/events/{event_id}/status", response_model=ApiResponse[EmptyData])
async def patch_event_status(
    request: Request,
    event_id: str,
    payload: StatusPatchRequest,
    session: DatabaseSession,
    idempotency_key: IdempotencyHeader,
    identity: CurrentIdentity,
) -> ApiResponse[EmptyData]:
    service = _service(request, session)
    result = await service.patch_event_status(
        identity,
        event_id,
        status=payload.status,
        note=payload.note,
        version=payload.version,
        idempotency_key=idempotency_key,
    )
    return ApiResponse(data=result, request_id=get_request_id(request))


@router.post(
    "/events/{event_id}/intervention-reminder",
    response_model=ApiResponse[EmptyData],
)
async def send_intervention_reminder(
    request: Request,
    event_id: str,
    payload: InterventionReminderRequest,
    session: DatabaseSession,
    idempotency_key: IdempotencyHeader,
    identity: CurrentIdentity,
) -> ApiResponse[EmptyData]:
    del payload, idempotency_key
    service = _service(request, session)
    await service.get_event(identity, event_id)
    # TODO(YS7): 接入设备语音播报或家属外呼，并记录实际送达状态。
    raise HTTPException(status_code=501, detail="intervention reminder is not implemented")


@router.get("/devices", response_model=ApiResponse[DeviceListData])
async def list_devices(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[DeviceListData]:
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    return ApiResponse(
        data=await service.list_devices(elder),
        request_id=get_request_id(request),
    )


@router.get("/devices/{device_id}/live-url", response_model=ApiResponse[LiveUrlData])
async def get_live_url(
    request: Request,
    device_id: str,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[LiveUrlData]:
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    try:
        data = await service.get_live_url(elder, device_id)
    except Ys7ApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ApiResponse(data=data, request_id=get_request_id(request))


@router.get(
    "/devices/{device_id}/live-sdk-session",
    response_model=ApiResponse[LiveSdkSessionData],
)
async def get_live_sdk_session(
    request: Request,
    response: Response,
    device_id: str,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[LiveSdkSessionData]:
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    try:
        data = await service.get_live_sdk_session(elder, device_id)
    except Ys7ApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    response.headers["Cache-Control"] = "no-store"
    return ApiResponse(data=data, request_id=get_request_id(request))


@router.post(
    "/devices/{device_id}/audio-pcm",
    response_model=ApiResponse[EmptyData],
)
async def relay_device_audio_pcm(
    request: Request,
    device_id: str,
    session: DatabaseSession,
    identity: CurrentIdentity,
    pcm: Annotated[bytes, Body(media_type="application/octet-stream")],
    sample_rate: Annotated[int, Header(alias="X-Audio-Sample-Rate", ge=8_000, le=48_000)],
    elder_id: str | None = Query(default=None),
) -> ApiResponse[EmptyData]:
    settings = request.app.state.settings
    if not settings.ys7_media_enabled or settings.ys7_media_source != "app_relay":
        raise HTTPException(status_code=503, detail="App PCM relay is disabled")
    if len(pcm) > 64_000:
        raise HTTPException(status_code=413, detail="PCM relay payload is too large")
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    devices = await service.list_devices(elder)
    if not any(device.device_id == device_id for device in devices.devices):
        raise HTTPException(status_code=404, detail="device not found")
    relay: AppPcmRelaySource = request.app.state.ys7_pcm_relay
    try:
        relay.push(device_id=device_id, pcm=pcm, sample_rate=sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    cognitive_collector: CognitiveAudioCollector = request.app.state.cognitive_collector
    cognitive_collector.push(
        subject_key=elder.external_subject or str(elder.id),
        device_id=device_id,
        pcm=pcm,
        sample_rate=sample_rate,
    )
    return ApiResponse(data=EmptyData(), request_id=get_request_id(request))


@router.get(
    "/devices/{device_id}/history-playback",
    response_model=ApiResponse[HistoryPlaybackData],
)
async def get_history_playback(
    request: Request,
    device_id: str,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
    at: HistoryAtQuery = None,
    duration_seconds: HistoryDurationQuery = 30,
) -> ApiResponse[HistoryPlaybackData]:
    del at, duration_seconds
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    devices = await service.list_devices(elder)
    if not any(device.device_id == device_id for device in devices.devices):
        raise HTTPException(status_code=404, detail="device not found")
    # TODO(YS7): 生成事件时间点附近的短时历史回放地址。
    raise HTTPException(status_code=501, detail="history playback is not implemented")


@router.get("/contacts", response_model=ApiResponse[ContactsData])
async def list_contacts(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
) -> ApiResponse[ContactsData]:
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    return ApiResponse(
        data=await service.list_contacts(elder),
        request_id=get_request_id(request),
    )


@router.get("/family/elders", response_model=ApiResponse[EldersData])
async def list_elders(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
) -> ApiResponse[EldersData]:
    service = _service(request, session)
    return ApiResponse(
        data=await service.list_elders(identity),
        request_id=get_request_id(request),
    )


@router.get("/stats/events", response_model=ApiResponse[EventsStatsData])
async def get_event_stats(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
    days: int = Query(default=30, ge=1, le=365),
) -> ApiResponse[EventsStatsData]:
    service = _service(request, session)
    elder = await service.resolve_elder(identity, elder_id)
    return ApiResponse(
        data=await service.get_event_stats(elder, days),
        request_id=get_request_id(request),
    )


@router.get("/stats/activity", response_model=ApiResponse[ActivityData])
async def get_activity_stats(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
    elder_id: str | None = Query(default=None),
    days: int = Query(default=7, ge=1, le=365),
) -> ApiResponse[ActivityData]:
    del days
    service = _service(request, session)
    await service.resolve_elder(identity, elder_id)
    return ApiResponse(
        data=service.get_activity_stats(),
        request_id=get_request_id(request),
    )
