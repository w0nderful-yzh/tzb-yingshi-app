import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RealtimeRiskEvent:
    event_id: str
    elder_user_id: UUID
    event_type: str
    level: str
    title: str
    summary: str
    device_id: str
    occurred_at: datetime
    status: str = "open"

    def payload(self) -> dict[str, object]:
        return {
            "type": "risk_event.upserted",
            "event": {
                "event_id": self.event_id,
                "type": self.event_type.lower(),
                "level": self.level.lower(),
                "title": self.title,
                "summary": self.summary,
                "device_id": self.device_id,
                "occurred_at": self.occurred_at.isoformat(),
                "status": self.status.lower(),
            },
        }


@dataclass(frozen=True, slots=True)
class _Ticket:
    elder_user_ids: frozenset[UUID]
    expires_at: float


@dataclass(frozen=True, slots=True)
class _Subscriber:
    elder_user_ids: frozenset[UUID]
    queue: asyncio.Queue[RealtimeRiskEvent]


class RealtimeEventBroker:
    """Single-process ticket store and bounded fan-out for the current one-worker deployment."""

    def __init__(self, *, ticket_ttl_seconds: int = 60, queue_maxsize: int = 32) -> None:
        self._ticket_ttl_seconds = ticket_ttl_seconds
        self._queue_maxsize = queue_maxsize
        self._tickets: dict[str, _Ticket] = {}
        self._subscribers: set[_Subscriber] = set()
        self._lock = asyncio.Lock()

    async def issue_ticket(self, elder_user_ids: set[UUID]) -> tuple[str, int]:
        if not elder_user_ids:
            raise ValueError("At least one elder is required for a realtime ticket")
        ticket = secrets.token_urlsafe(32)
        now = monotonic()
        async with self._lock:
            self._purge_expired(now)
            self._tickets[ticket] = _Ticket(
                elder_user_ids=frozenset(elder_user_ids),
                expires_at=now + self._ticket_ttl_seconds,
            )
        return ticket, self._ticket_ttl_seconds

    async def consume_ticket(self, ticket: str) -> frozenset[UUID] | None:
        now = monotonic()
        async with self._lock:
            self._purge_expired(now)
            issued = self._tickets.pop(ticket, None)
        if issued is None or issued.expires_at <= now:
            return None
        return issued.elder_user_ids

    @asynccontextmanager
    async def subscribe(
        self,
        elder_user_ids: frozenset[UUID],
    ) -> AsyncIterator[asyncio.Queue[RealtimeRiskEvent]]:
        subscriber = _Subscriber(
            elder_user_ids=elder_user_ids,
            queue=asyncio.Queue(maxsize=self._queue_maxsize),
        )
        async with self._lock:
            self._subscribers.add(subscriber)
        try:
            yield subscriber.queue
        finally:
            async with self._lock:
                self._subscribers.discard(subscriber)

    async def publish(self, event: RealtimeRiskEvent) -> None:
        async with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            if event.elder_user_id not in subscriber.elder_user_ids:
                continue
            if subscriber.queue.full():
                with suppress(asyncio.QueueEmpty):
                    subscriber.queue.get_nowait()
                    subscriber.queue.task_done()
            subscriber.queue.put_nowait(event)

    def _purge_expired(self, now: float) -> None:
        expired = [ticket for ticket, issued in self._tickets.items() if issued.expires_at <= now]
        for ticket in expired:
            self._tickets.pop(ticket, None)
