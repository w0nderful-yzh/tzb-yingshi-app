import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select

from app.api.dependencies import CurrentIdentity, DatabaseSession
from app.common.responses import ApiResponse
from app.core.request_id import get_request_id
from app.infrastructure.database.models import FamilyBindingModel
from app.infrastructure.realtime_events import RealtimeEventBroker

router = APIRouter(tags=["realtime"])


class WebSocketTicketData(BaseModel):
    ticket: str
    expires_in: int


@router.post("/ws/tickets", response_model=ApiResponse[WebSocketTicketData])
async def create_websocket_ticket(
    request: Request,
    session: DatabaseSession,
    identity: CurrentIdentity,
) -> ApiResponse[WebSocketTicketData]:
    if identity.role == "elder":
        elder_user_ids = {identity.user.id}
    else:
        elder_user_ids = set(
            (
                await session.scalars(
                    select(FamilyBindingModel.elder_user_id).where(
                        FamilyBindingModel.guardian_user_id == identity.user.id,
                        FamilyBindingModel.status == "ACTIVE",
                    )
                )
            ).all()
        )
    if not elder_user_ids:
        raise HTTPException(status_code=403, detail="当前账号没有可守护的老人")
    broker: RealtimeEventBroker = request.app.state.realtime_event_broker
    ticket, expires_in = await broker.issue_ticket(elder_user_ids)
    return ApiResponse(
        data=WebSocketTicketData(ticket=ticket, expires_in=expires_in),
        request_id=get_request_id(request),
    )


@router.websocket("/ws/events")
async def realtime_events(
    websocket: WebSocket,
    ticket: str = Query(min_length=16, max_length=256),
) -> None:
    broker: RealtimeEventBroker = websocket.app.state.realtime_event_broker
    elder_user_ids = await broker.consume_ticket(ticket)
    if elder_user_ids is None:
        await websocket.close(code=4401, reason="invalid or expired ticket")
        return
    await websocket.accept()
    try:
        async with broker.subscribe(elder_user_ids) as queue:
            await websocket.send_json(
                {"type": "connected", "sent_at": datetime.now(UTC).isoformat()}
            )
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                except TimeoutError:
                    await websocket.send_json(
                        {"type": "ping", "sent_at": datetime.now(UTC).isoformat()}
                    )
                    continue
                try:
                    await websocket.send_json(event.payload())
                finally:
                    queue.task_done()
    except WebSocketDisconnect:
        return
