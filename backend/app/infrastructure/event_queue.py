import asyncio
from dataclasses import dataclass

from app.infrastructure.external.ys7.models import Ys7Signal


class SignalQueueFullError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QueuedYs7Signal:
    signal: Ys7Signal
    dedup_key: str
    raw_event_ref: str


class Ys7EventQueue:
    def __init__(self, *, maxsize: int) -> None:
        self._queue: asyncio.Queue[QueuedYs7Signal] = asyncio.Queue(maxsize=maxsize)

    def publish_nowait(self, event: QueuedYs7Signal) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull as exc:
            raise SignalQueueFullError("YS7 signal queue is full") from exc

    async def get(self) -> QueuedYs7Signal:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    @property
    def depth(self) -> int:
        return self._queue.qsize()
