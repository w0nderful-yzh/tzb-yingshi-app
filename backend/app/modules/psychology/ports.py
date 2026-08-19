"""Ports implemented by psychology algorithm adapters."""

from typing import Protocol

from app.modules.psychology.source_schemas import PsychologySourceSnapshot


class PsychologySourceError(RuntimeError):
    """The psychology projection service did not provide a usable response."""


class PsychologySource(Protocol):
    async def get_latest_assessment(self, *, subject_key: str) -> PsychologySourceSnapshot: ...
