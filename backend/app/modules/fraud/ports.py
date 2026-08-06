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


class FraudRiskEventSink(Protocol):
    async def upsert(self, event: FraudRiskEventWrite) -> None: ...


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
