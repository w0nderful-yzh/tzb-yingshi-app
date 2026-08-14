from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class FraudRiskEventWrite:
    source_event_id: str
    external_device_id: str
    risk_level: str
    confidence: float
    summary: str
    occurred_at: datetime
    received_at: datetime
    evidence: dict[str, Any]
    model_name: str
    model_version: str
    verification_status: str | None = None


class FraudRiskEventSink(Protocol):
    async def upsert(self, event: FraudRiskEventWrite) -> None: ...

    async def retract_preliminary(
        self,
        *,
        source_event_id: str,
        reason: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class FraudSessionRecord:
    session_id: str
    device_id: str
    elder_alone: bool
    status: str
    started_at: datetime
    last_activity_at: datetime
    ended_at: datetime | None
    speech_events: dict[str, dict[str, Any]]
    llm_evidence: dict[str, dict[str, Any]]
    last_llm_review_id: str | None


class FraudSessionStore(Protocol):
    async def load(self, *, device_id: str, session_id: str) -> FraudSessionRecord | None: ...

    async def upsert(self, record: FraudSessionRecord) -> None: ...

    async def close_other_active(
        self,
        *,
        device_id: str,
        active_session_id: str,
        ended_at: datetime,
    ) -> None: ...


class SemanticEvidenceRetriever(Protocol):
    """Optional semantic layer producing weak/medium evidence only.

    Semantic similarity must never alone drive S4/S5; the state machine
    requires strong action kinds for those transitions. When the model is
    unavailable the adapter must return no evidence and keep `available` False
    so the rules/classifier/state-machine chain still works.
    """

    @property
    def available(self) -> bool: ...

    async def retrieve(self, *, text: str, session_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class RecentFraudContext:
    """Desensitized recent risk context for a device (no full transcripts)."""

    device_id: str
    session_id: str
    recent_risk_events: int
    last_risk_level: str | None
    last_kinds: tuple[str, ...]
    last_occurred_at: datetime | None


class RecentFraudRiskStore(Protocol):
    """Reads the last 24-72 h of risk events per device for context evidence."""

    async def load_recent_context(
        self,
        *,
        device_id: str,
        session_id: str,
        lookback_hours: int = 24,
    ) -> RecentFraudContext | None: ...
