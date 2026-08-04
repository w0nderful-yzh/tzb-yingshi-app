import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from app.modules.fraud.ports import FraudRiskEventSink, FraudRiskEventWrite
from app.modules.fraud.risk_engine import (
    RISK_MODEL_NAME,
    RISK_MODEL_VERSION,
    build_risk_snapshot,
    to_epoch_ms,
)
from app.modules.fraud.schemas import (
    FraudAnalyzeData,
    FraudAnalyzeRequest,
    FraudRiskSnapshot,
)
from app.modules.fraud.speech_risk import build_speech_events
from app.modules.fraud.visual_event_store import VisualEventStore


@dataclass(slots=True)
class _FraudSession:
    session_id: str
    device_id: str
    elder_alone: bool = False
    speech_events: dict[str, dict[str, Any]] = field(default_factory=dict)


class FraudSessionService:
    """Owns short-lived fraud sessions and persists their latest risk snapshot."""

    def __init__(
        self,
        *,
        visual_event_store: VisualEventStore,
        risk_event_sink: FraudRiskEventSink | None = None,
        memory_ms: int = 120_000,
    ) -> None:
        self._visual_event_store = visual_event_store
        self._risk_event_sink = risk_event_sink
        self._memory_ms = memory_ms
        self._sessions: dict[tuple[str, str], _FraudSession] = {}
        self._lock = asyncio.Lock()

    async def analyze(self, payload: FraudAnalyzeRequest) -> FraudAnalyzeData:
        key = (payload.device_id, payload.session_id)
        async with self._lock:
            session = self._sessions.setdefault(
                key,
                _FraudSession(
                    session_id=payload.session_id,
                    device_id=payload.device_id,
                    elder_alone=payload.elder_alone,
                ),
            )
            session.elder_alone = session.elder_alone or payload.elder_alone
            existing = session.speech_events.get(payload.source_event_id)
            if existing is None:
                speech_events = await asyncio.to_thread(
                    build_speech_events,
                    [
                        {
                            "start_ms": to_epoch_ms(payload.occurred_at),
                            "end_ms": to_epoch_ms(payload.ended_at),
                            "text": payload.text,
                        }
                    ],
                    event_id_offset=len(session.speech_events),
                )
                speech_event = speech_events[0]
                speech_event["source_event_id"] = payload.source_event_id
                session.speech_events[payload.source_event_id] = speech_event
                status: Literal["accepted", "duplicate"] = "accepted"
            else:
                speech_event = existing
                status = "duplicate"

            risk = await self._snapshot(session)
            if risk.state != "S0_NORMAL":
                await self._persist(risk)
            return FraudAnalyzeData(status=status, speech_event=speech_event, risk=risk)

    async def get_session(
        self,
        *,
        device_id: str,
        session_id: str,
    ) -> FraudRiskSnapshot | None:
        async with self._lock:
            session = self._sessions.get((device_id, session_id))
            if session is None:
                return None
            return await self._snapshot(session)

    async def _snapshot(self, session: _FraudSession) -> FraudRiskSnapshot:
        visual_events = await self._visual_event_store.list(
            device_id=session.device_id,
            limit=200,
        )
        return build_risk_snapshot(
            session_id=session.session_id,
            device_id=session.device_id,
            speech_events=list(session.speech_events.values()),
            visual_events=visual_events,
            elder_alone=session.elder_alone,
            memory_ms=self._memory_ms,
        )

    async def _persist(self, risk: FraudRiskSnapshot) -> None:
        if self._risk_event_sink is None:
            return
        stable_key = hashlib.sha256(f"{risk.device_id}\0{risk.session_id}".encode()).hexdigest()
        await self._risk_event_sink.upsert(
            FraudRiskEventWrite(
                source_event_id=f"fraud-session:{stable_key}",
                external_device_id=risk.device_id,
                risk_level=risk.risk_level,
                confidence=risk.confidence,
                summary=f"{risk.state_label}：{risk.transition_reason}"[:500],
                occurred_at=risk.occurred_at,
                received_at=datetime.now(UTC),
                evidence=risk.model_dump(mode="json"),
                model_name=RISK_MODEL_NAME,
                model_version=RISK_MODEL_VERSION,
            )
        )
