"""Read-only Cognitive overview orchestration."""

import logging

from pydantic import ValidationError

from app.modules.psychology.cognitive.mapping import (
    map_cognitive_snapshot,
    unavailable_cognitive_overview,
)
from app.modules.psychology.cognitive.result_store import CognitiveResultStore
from app.modules.psychology.cognitive.schemas import CognitiveOverview

logger = logging.getLogger(__name__)


class CognitiveOverviewService:
    def __init__(self, store: CognitiveResultStore | None) -> None:
        self._store = store

    async def get_overview(self, *, subject_key: str) -> CognitiveOverview:
        if self._store is None:
            return unavailable_cognitive_overview()
        try:
            snapshot = self._store.read_latest(subject_key)
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("cognitive snapshot unavailable: %s", type(exc).__name__)
            return unavailable_cognitive_overview()
        if snapshot is None:
            return unavailable_cognitive_overview()

        latest_completed = None
        if snapshot.status != "completed":
            try:
                latest_completed = self._store.read_latest_completed(subject_key)
            except (OSError, ValueError, ValidationError):
                latest_completed = None
        return map_cognitive_snapshot(snapshot, latest_completed=latest_completed)
