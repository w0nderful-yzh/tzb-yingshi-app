from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime

from app.modules.psychology.schemas import SourceStatus
from app.modules.psychology.service import PsychologyService


class PsychologyObservationController:
    """Periodically reads bounded psychology assessments.

    This controller never launches continuous OpenFace processing. It only
    refreshes the latest completed/processing assessment at a configured
    interval while a guardian session is active.
    """

    def __init__(self, service: PsychologyService, *, interval_seconds: float) -> None:
        self._service = service
        self._interval_seconds = interval_seconds
        self._subject_key: str | None = None
        self._task: asyncio.Task[None] | None = None
        self.last_checked_at: datetime | None = None
        self.last_available = False
        self.last_error: str | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, subject_key: str) -> None:
        if self.active:
            return
        self._subject_key = subject_key
        self.last_error = None
        self._task = asyncio.create_task(
            self._run(),
            name="psychology-periodic-observation",
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        self._subject_key = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        while True:
            await self.observe_once()
            await asyncio.sleep(self._interval_seconds)

    async def observe_once(self) -> None:
        subject_key = self._subject_key
        if subject_key is None:
            return
        try:
            overview = await self._service.get_overview(subject_key=subject_key)
            self.last_available = overview.source_status is not SourceStatus.UNAVAILABLE
            self.last_error = None
        except Exception as exc:  # observation must never stop other modules
            self.last_available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_checked_at = datetime.now(UTC)
