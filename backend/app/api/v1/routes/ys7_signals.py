from hmac import compare_digest
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ValidationError

from app.common.responses import ApiResponse
from app.core.config import Settings
from app.core.request_id import get_request_id
from app.infrastructure.event_queue import SignalQueueFullError, Ys7EventQueue
from app.infrastructure.external.ys7.signal_listener import Ys7SignalListener
from app.modules.fraud.schemas import VisualEvent
from app.modules.fraud.visual_event_store import VisualEventStore
from app.workers.ys7_alarm_poll_worker import Ys7AlarmPollWorker
from app.workers.ys7_event_worker import Ys7EventWorker
from app.workers.ys7_media_stream_worker import Ys7MediaStreamWorker

router = APIRouter(tags=["ys7"])


class SignalReceiptData(BaseModel):
    status: Literal["accepted", "duplicate"]
    source_event_id: str
    raw_event_ref: str | None


class SignalStatusData(BaseModel):
    enabled: bool
    worker_running: bool
    queue_depth: int
    polling_enabled: bool
    poller_running: bool
    poll_interval_seconds: float
    polls_completed: int
    alarms_seen: int
    signals_accepted: int
    signals_duplicate: int
    alarms_ignored: int
    last_polled_at: str | None
    last_ignored_alarm_type: str | None
    polling_last_error: str | None


class MediaStatusData(BaseModel):
    enabled: bool
    source: Literal["cloud", "app_relay"]
    running: bool
    connected: bool
    session_id: str | None
    queue_depth: int
    chunks_processed: int
    chunks_dropped: int
    streaming_enabled: bool
    partials_processed: int
    partials_failed: int
    reconnect_attempts: int
    last_error: str | None
    models_ready: Literal["DISABLED", "WARMING_UP", "READY", "FAILED"]
    classifier_ready: bool
    warmup_error: str | None


def _authorize(request: Request, provided_token: str | None) -> Settings:
    settings: Settings = request.app.state.settings
    if not settings.ys7_signal_enabled:
        raise HTTPException(status_code=503, detail="YS7 signal ingestion is disabled")
    configured_token = settings.ys7_webhook_token
    if configured_token is None:
        raise HTTPException(status_code=503, detail="YS7 webhook token is not configured")
    if provided_token is None or not compare_digest(
        provided_token,
        configured_token.get_secret_value(),
    ):
        raise HTTPException(status_code=401, detail="invalid YS7 webhook token")
    return settings


@router.post(
    "/integrations/ys7/events",
    status_code=202,
    response_model=ApiResponse[SignalReceiptData],
)
async def receive_ys7_event(
    request: Request,
    payload: dict[str, object],
    webhook_token: str | None = Header(default=None, alias="X-YS7-Webhook-Token"),
) -> ApiResponse[SignalReceiptData]:
    _authorize(request, webhook_token)
    listener: Ys7SignalListener = request.app.state.ys7_signal_listener
    try:
        receipt = await listener.receive(payload)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid YS7 event payload") from exc
    except SignalQueueFullError as exc:
        raise HTTPException(status_code=503, detail="YS7 event queue is full") from exc
    return ApiResponse(
        data=SignalReceiptData(
            status=receipt.status,
            source_event_id=receipt.source_event_id,
            raw_event_ref=receipt.raw_event_ref,
        ),
        request_id=get_request_id(request),
    )


@router.get(
    "/integrations/ys7/status",
    response_model=ApiResponse[SignalStatusData],
)
async def get_ys7_status(request: Request) -> ApiResponse[SignalStatusData]:
    settings: Settings = request.app.state.settings
    queue: Ys7EventQueue = request.app.state.ys7_event_queue
    worker: Ys7EventWorker = request.app.state.ys7_event_worker
    poller: Ys7AlarmPollWorker = request.app.state.ys7_alarm_poll_worker
    return ApiResponse(
        data=SignalStatusData(
            enabled=settings.ys7_signal_enabled,
            worker_running=worker.running,
            queue_depth=queue.depth,
            polling_enabled=settings.ys7_alarm_poll_enabled,
            poller_running=poller.running,
            poll_interval_seconds=settings.ys7_alarm_poll_interval_seconds,
            polls_completed=poller.polls_completed,
            alarms_seen=poller.alarms_seen,
            signals_accepted=poller.signals_accepted,
            signals_duplicate=poller.signals_duplicate,
            alarms_ignored=poller.alarms_ignored,
            last_polled_at=(
                poller.last_polled_at.isoformat() if poller.last_polled_at is not None else None
            ),
            last_ignored_alarm_type=poller.last_ignored_alarm_type,
            polling_last_error=poller.last_error,
        ),
        request_id=get_request_id(request),
    )


@router.get(
    "/integrations/ys7/media/status",
    response_model=ApiResponse[MediaStatusData],
)
async def get_ys7_media_status(request: Request) -> ApiResponse[MediaStatusData]:
    settings: Settings = request.app.state.settings
    worker: Ys7MediaStreamWorker = request.app.state.ys7_media_worker
    readiness = request.app.state.model_readiness.snapshot()
    return ApiResponse(
        data=MediaStatusData(
            enabled=settings.ys7_media_enabled,
            source=settings.ys7_media_source,
            running=worker.running,
            connected=worker.connected,
            session_id=worker.session_id,
            queue_depth=worker.queue_depth,
            chunks_processed=worker.chunks_processed,
            chunks_dropped=worker.chunks_dropped,
            streaming_enabled=request.app.state.fraud_audio_service.streaming_enabled,
            partials_processed=worker.partials_processed,
            partials_failed=worker.partials_failed,
            reconnect_attempts=worker.reconnect_attempts,
            last_error=worker.last_error,
            models_ready=readiness.models_ready,
            classifier_ready=readiness.classifier_ready,
            warmup_error=readiness.warmup_error,
        ),
        request_id=get_request_id(request),
    )


@router.get(
    "/fraud/visual-events",
    response_model=ApiResponse[list[VisualEvent]],
)
async def list_visual_events(
    request: Request,
    device_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
) -> ApiResponse[list[VisualEvent]]:
    store: VisualEventStore = request.app.state.visual_event_store
    events = await store.list(device_id=device_id, limit=limit)
    return ApiResponse(data=events, request_id=get_request_id(request))
