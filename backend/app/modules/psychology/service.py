"""Read-only psychology reference-observation orchestration."""

import logging

from app.modules.psychology.mapping import map_psychology_snapshot, unavailable_overview
from app.modules.psychology.ports import PsychologySource, PsychologySourceError
from app.modules.psychology.schemas import PsychologyOverview

logger = logging.getLogger(__name__)


class PsychologyService:
    def __init__(self, source: PsychologySource | None) -> None:
        self._source = source

    async def get_overview(self, *, subject_key: str) -> PsychologyOverview:
        if self._source is None:
            return unavailable_overview()
        try:
            snapshot = await self._source.get_latest_assessment(subject_key=subject_key)
        except (PsychologySourceError, ValueError) as exc:
            logger.warning("psychology source unavailable: %s", type(exc).__name__)
            return unavailable_overview()
        latest_completed = None
        if snapshot.status == "processing":
            try:
                latest_completed = await self._source.get_latest_completed_assessment(
                    subject_key=subject_key
                )
            except (PsychologySourceError, ValueError):
                # A first-time assessment legitimately has no completed result.
                latest_completed = None
        return map_psychology_snapshot(snapshot, latest_completed=latest_completed)
