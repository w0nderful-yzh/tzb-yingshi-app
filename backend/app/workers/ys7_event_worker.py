import asyncio
import logging
from contextlib import suppress

from app.infrastructure.event_queue import Ys7EventQueue
from app.infrastructure.external.ys7.event_adapter import Ys7EventAdapter
from app.modules.fraud.visual_event_store import VisualEventStore

logger = logging.getLogger(__name__)


class Ys7EventWorker:
    def __init__(
        self,
        *,
        event_queue: Ys7EventQueue,
        adapter: Ys7EventAdapter,
        visual_event_store: VisualEventStore,
    ) -> None:
        self._event_queue = event_queue
        self._adapter = adapter
        self._visual_event_store = visual_event_store
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self._run(), name="ys7-event-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            queued = await self._event_queue.get()
            try:
                event = self._adapter.adapt(queued)
                await self._visual_event_store.add(event)
            except Exception:
                logger.exception(
                    "Failed to adapt YS7 signal",
                    extra={"source_event_id": queued.signal.source_event_id},
                )
            finally:
                self._event_queue.task_done()
