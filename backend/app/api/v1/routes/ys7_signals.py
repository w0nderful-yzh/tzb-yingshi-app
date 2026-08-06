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


class MediaStatusData(BaseModel):
    enabled: bool
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
    return ApiResponse(
        data=SignalStatusData(
            enabled=settings.ys7_signal_enabled,
            worker_running=worker.running,
            queue_depth=queue.depth,
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
    return ApiResponse(
        data=MediaStatusData(
            enabled=settings.ys7_media_enabled,
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
