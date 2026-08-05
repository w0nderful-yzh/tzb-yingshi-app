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
