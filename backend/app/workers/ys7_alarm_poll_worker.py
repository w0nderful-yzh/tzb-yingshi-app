import asyncio
import logging
import time
from contextlib import suppress
from datetime import UTC, datetime

from app.infrastructure.external.ys7.alarm_mapper import Ys7AlarmMapper
from app.infrastructure.external.ys7.api_client import Ys7AlarmProvider
from app.infrastructure.external.ys7.signal_listener import Ys7SignalListener

logger = logging.getLogger(__name__)


class Ys7AlarmPollWorker:
    def __init__(
        self,
        *,
        alarm_provider: Ys7AlarmProvider,
        signal_listener: Ys7SignalListener,
        mapper: Ys7AlarmMapper,
        device_serial: str,
        interval_seconds: float,
        lookback_seconds: int,
        page_size: int,
    ) -> None:
        self._alarm_provider = alarm_provider
        self._signal_listener = signal_listener
        self._mapper = mapper
        self._device_serial = device_serial
        self._interval_seconds = interval_seconds
        self._lookback_seconds = lookback_seconds
        self._page_size = page_size
        self._task: asyncio.Task[None] | None = None
        self.polls_completed = 0
        self.alarms_seen = 0
        self.signals_accepted = 0
        self.signals_duplicate = 0
        self.alarms_ignored = 0
        self.last_polled_at: datetime | None = None
        self.last_ignored_alarm_type: str | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self._run(), name="ys7-alarm-poll-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def poll_once(self) -> None:
        end_time_ms = round(time.time() * 1000)
        alarms = await self._alarm_provider.list_device_alarms(
            device_serial=self._device_serial,
            start_time_ms=end_time_ms - self._lookback_seconds * 1000,
            end_time_ms=end_time_ms,
            page_size=self._page_size,
        )
        self.polls_completed += 1
        self.alarms_seen += len(alarms)
        self.last_polled_at = datetime.now(UTC)
        self.last_error = None
        for alarm in alarms:
            payload = self._mapper.map(alarm)
            if payload is None:
                self.alarms_ignored += 1
                self.last_ignored_alarm_type = self._mapper.alarm_type_label(alarm)
                continue
            try:
                receipt = await self._signal_listener.receive(payload)
            except Exception:
                logger.exception(
                    "Failed to ingest polled YS7 alarm",
                    extra={"alarm_type": self._mapper.alarm_type_label(alarm)},
                )
                continue
            if receipt.status == "accepted":
                self.signals_accepted += 1
            else:
                self.signals_duplicate += 1

    async def _run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception as exc:
                self.last_error = str(exc)
                logger.warning("YS7 alarm polling failed: %s", exc)
            await asyncio.sleep(self._interval_seconds)
