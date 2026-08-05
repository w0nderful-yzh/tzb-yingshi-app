import asyncio

from app.modules.fraud.schemas import VisualEvent


class VisualEventStore:
    def __init__(self) -> None:
        self._events: dict[str, VisualEvent] = {}
        self._lock = asyncio.Lock()

    async def add(self, event: VisualEvent) -> None:
        async with self._lock:
            self._events[event.source_event_id] = event

    async def list(self, *, device_id: str | None = None, limit: int = 100) -> list[VisualEvent]:
        async with self._lock:
            events = list(self._events.values())
        if device_id is not None:
            events = [event for event in events if event.device_id == device_id]
        events.sort(key=lambda event: (event.occurred_at, event.received_at), reverse=True)
        return events[:limit]
