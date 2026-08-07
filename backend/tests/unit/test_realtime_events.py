from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.infrastructure.realtime_events import RealtimeEventBroker, RealtimeRiskEvent


def test_websocket_rejects_unknown_ticket(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as error,
        client.websocket_connect("/api/v1/ws/events?ticket=unknown-ticket-value"),
    ):
        pass

    assert error.value.code == 4401


@pytest.mark.asyncio
async def test_ticket_is_one_time_and_scopes_event_delivery() -> None:
    broker = RealtimeEventBroker(ticket_ttl_seconds=60, queue_maxsize=2)
    elder_id = uuid4()
    other_elder_id = uuid4()
    ticket, expires_in = await broker.issue_ticket({elder_id})

    scope = await broker.consume_ticket(ticket)
    assert expires_in == 60
    assert scope == frozenset({elder_id})
    assert await broker.consume_ticket(ticket) is None

    async with broker.subscribe(scope) as queue:
        await broker.publish(_event(other_elder_id, "ignored"))
        assert queue.empty()
        await broker.publish(_event(elder_id, "delivered"))
        assert (await queue.get()).event_id == "delivered"
        queue.task_done()


@pytest.mark.asyncio
async def test_slow_subscriber_keeps_latest_events() -> None:
    broker = RealtimeEventBroker(queue_maxsize=2)
    elder_id = uuid4()
    async with broker.subscribe(frozenset({elder_id})) as queue:
        await broker.publish(_event(elder_id, "one"))
        await broker.publish(_event(elder_id, "two"))
        await broker.publish(_event(elder_id, "three"))
        assert [(await queue.get()).event_id, (await queue.get()).event_id] == ["two", "three"]
        queue.task_done()
        queue.task_done()


def _event(elder_user_id: UUID, event_id: str) -> RealtimeRiskEvent:
    return RealtimeRiskEvent(
        event_id=event_id,
        elder_user_id=elder_user_id,
        event_type="FRAUD_SUSPECTED",
        level="WARNING",
        title="诈骗风险",
        summary="检测到可疑话术",
        device_id="camera-1",
        occurred_at=datetime.now(UTC),
    )
