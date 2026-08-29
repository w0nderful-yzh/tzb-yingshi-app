"""Lifecycle port used by the shared guardian-session orchestrator."""

from typing import Protocol


class CognitiveCollectionControl(Protocol):
    @property
    def enabled(self) -> bool: ...

    def attach(self, *, subject_key: str, session_id: str) -> bool: ...

    async def detach(self, *, subject_key: str) -> None: ...
