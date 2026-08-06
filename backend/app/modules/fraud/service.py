import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from app.modules.fraud.llm import FraudLlmReviewQueue, FraudLlmReviewRequest
from app.modules.fraud.ports import (
    FraudRiskEventSink,
    FraudRiskEventWrite,
    FraudSessionRecord,
    FraudSessionStore,
)
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
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "ACTIVE"
    ended_at: datetime | None = None
    speech_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    llm_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_llm_review_id: str | None = None


class FraudSessionService:
    """Owns short-lived fraud sessions and persists their latest risk snapshot."""

    def __init__(
        self,
        *,
        visual_event_store: VisualEventStore,
        risk_event_sink: FraudRiskEventSink | None = None,
        llm_review_queue: FraudLlmReviewQueue | None = None,
        llm_trigger_state_index: int = 2,
        llm_max_transcript_chars: int = 6_000,
        llm_vision_enabled: bool = False,
        llm_max_images: int = 4,
        memory_ms: int = 120_000,
        session_store: FraudSessionStore | None = None,
    ) -> None:
        self._visual_event_store = visual_event_store
        self._risk_event_sink = risk_event_sink
        self._llm_review_queue = llm_review_queue
        self._llm_trigger_state_index = llm_trigger_state_index
        self._llm_max_transcript_chars = llm_max_transcript_chars
        self._llm_vision_enabled = llm_vision_enabled
        self._llm_max_images = llm_max_images
        self._memory_ms = memory_ms
        self._session_store = session_store
        self._sessions: dict[tuple[str, str], _FraudSession] = {}
        self._lock = asyncio.Lock()

    async def analyze(self, payload: FraudAnalyzeRequest) -> FraudAnalyzeData:
        key = (payload.device_id, payload.session_id)
        async with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = await self._restore(payload.device_id, payload.session_id)
            if session is None:
                session = _FraudSession(
                    session_id=payload.session_id,
                    device_id=payload.device_id,
                    elder_alone=payload.elder_alone,
                    started_at=payload.occurred_at,
                    last_activity_at=payload.ended_at,
                )
                await self._close_other_sessions(
                    device_id=payload.device_id,
                    active_session_id=payload.session_id,
                    ended_at=payload.occurred_at,
                )
            self._sessions[key] = session
            session.elder_alone = session.elder_alone or payload.elder_alone
            session.started_at = min(session.started_at, payload.occurred_at)
            session.last_activity_at = max(session.last_activity_at, payload.ended_at)
            existing = session.speech_events.get(payload.source_event_id)
            should_update = existing is not None and existing.get("transcript_status") == "PARTIAL"
            if existing is None or should_update:
                speech_events = await asyncio.to_thread(
                    build_speech_events,
                    [
                        {
                            "start_ms": to_epoch_ms(payload.occurred_at),
                            "end_ms": to_epoch_ms(payload.ended_at),
                            "text": payload.text,
                            "language": payload.language,
                            "emotion": payload.emotion,
                            "audio_events": payload.audio_events,
                            "transcript_status": payload.transcript_status,
                        }
                    ],
                    event_id_offset=len(session.speech_events) - (1 if should_update else 0),
                )
                speech_event = speech_events[0]
                speech_event["source_event_id"] = payload.source_event_id
                session.speech_events[payload.source_event_id] = speech_event
                status: Literal["accepted", "updated", "duplicate"] = (
                    "updated" if should_update else "accepted"
                )
            else:
                speech_event = existing
                status = "duplicate"

            risk = await self._snapshot(session)
            if payload.transcript_status == "FINAL" and risk.state != "S0_NORMAL":
                await self._persist(risk)
            if payload.transcript_status == "FINAL":
                await self._submit_llm_review(session, risk)
            await self._persist_session(session)
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
                session = await self._restore(device_id, session_id)
            if session is None:
                return None
            self._sessions[(device_id, session_id)] = session
            return await self._snapshot(session)

    async def apply_llm_evidence(
        self,
        *,
        device_id: str,
        session_id: str,
        evidence: list[dict[str, Any]],
    ) -> FraudRiskSnapshot | None:
        async with self._lock:
            session = self._sessions.get((device_id, session_id))
            if session is None:
                session = await self._restore(device_id, session_id)
            if session is None:
                return None
            self._sessions[(device_id, session_id)] = session
            for item in evidence:
                session.llm_evidence[str(item["evidence_id"])] = dict(item)
            risk = await self._snapshot(session)
            if risk.state != "S0_NORMAL":
                await self._persist(risk)
            await self._persist_session(session)
            return risk

    async def _restore(self, device_id: str, session_id: str) -> _FraudSession | None:
        if self._session_store is None:
            return None
        record = await self._session_store.load(device_id=device_id, session_id=session_id)
        if record is None:
            return None
        return _FraudSession(
            session_id=record.session_id,
            device_id=record.device_id,
            elder_alone=record.elder_alone,
            started_at=record.started_at,
            last_activity_at=record.last_activity_at,
            status=record.status,
            ended_at=record.ended_at,
            speech_events={key: dict(value) for key, value in record.speech_events.items()},
            llm_evidence={key: dict(value) for key, value in record.llm_evidence.items()},
            last_llm_review_id=record.last_llm_review_id,
        )

    async def _persist_session(self, session: _FraudSession) -> None:
        if self._session_store is None:
            return
        await self._session_store.upsert(
            FraudSessionRecord(
                session_id=session.session_id,
                device_id=session.device_id,
                elder_alone=session.elder_alone,
                status=session.status,
                started_at=session.started_at,
                last_activity_at=session.last_activity_at,
                ended_at=session.ended_at,
                speech_events={key: dict(value) for key, value in session.speech_events.items()},
                llm_evidence={key: dict(value) for key, value in session.llm_evidence.items()},
                last_llm_review_id=session.last_llm_review_id,
            )
        )

    async def _close_other_sessions(
        self,
        *,
        device_id: str,
        active_session_id: str,
        ended_at: datetime,
    ) -> None:
        for (stored_device_id, stored_session_id), session in self._sessions.items():
            if (
                stored_device_id != device_id
                or stored_session_id == active_session_id
                or session.status != "ACTIVE"
            ):
                continue
            session.status = "CLOSED"
            session.ended_at = max(session.started_at, ended_at)
            await self._persist_session(session)
        if self._session_store is not None:
            await self._session_store.close_other_active(
                device_id=device_id,
                active_session_id=active_session_id,
                ended_at=ended_at,
            )

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
            extra_evidence=list(session.llm_evidence.values()),
        )

    async def _submit_llm_review(
        self,
        session: _FraudSession,
        risk: FraudRiskSnapshot,
    ) -> None:
        if (
            self._llm_review_queue is None
            or risk.state_index < self._llm_trigger_state_index
            or risk.state_index >= 5
        ):
            return
        at_ms = to_epoch_ms(risk.occurred_at)
        lower_bound = at_ms - self._memory_ms
        recent = sorted(
            (
                event
                for event in session.speech_events.values()
                if int(event["end_ms"]) >= lower_bound
            ),
            key=lambda event: int(event["start_ms"]),
        )
        selected: list[dict[str, Any]] = []
        character_count = 0
        for event in reversed(recent):
            text = str(event.get("text", ""))
            if selected and character_count + len(text) > self._llm_max_transcript_chars:
                break
            selected.append(event)
            character_count += len(text)
        selected.reverse()
        if not selected:
            return
        visual_inputs: list[dict[str, Any]] = []
        if self._llm_vision_enabled:
            visual_events = await self._visual_event_store.list(
                device_id=session.device_id,
                limit=200,
            )
            seen_urls: set[str] = set()
            for visual_event in visual_events:
                occurred_ms = to_epoch_ms(visual_event.occurred_at)
                image_url = visual_event.image_url
                if (
                    image_url is None
                    or not image_url.startswith(("http://", "https://"))
                    or not lower_bound <= occurred_ms <= at_ms
                    or image_url in seen_urls
                ):
                    continue
                seen_urls.add(image_url)
                visual_inputs.append(
                    {
                        "source_event_id": visual_event.source_event_id,
                        "occurred_ms": occurred_ms,
                        "event_type": visual_event.event_type,
                        "confidence": visual_event.confidence,
                        "people_count": visual_event.people_count,
                        "image_url": image_url,
                    }
                )
                if len(visual_inputs) >= self._llm_max_images:
                    break
            visual_inputs.reverse()
        review_content = json.dumps(
            {
                "transcript": [
                    {
                        "start_ms": item["start_ms"],
                        "end_ms": item["end_ms"],
                        "text": item["text"],
                        "language": item.get("language"),
                        "emotion": item.get("emotion"),
                        "audio_events": item.get("audio_events") or [],
                    }
                    for item in selected
                ],
                "visual_inputs": [
                    {key: value for key, value in item.items() if key != "image_url"}
                    for item in visual_inputs
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        review_id = hashlib.sha256(
            f"{session.device_id}\0{session.session_id}\0{review_content}".encode()
        ).hexdigest()
        if session.last_llm_review_id == review_id:
            return
        submitted = self._llm_review_queue.submit_nowait(
            FraudLlmReviewRequest(
                review_id=review_id,
                session_id=session.session_id,
                device_id=session.device_id,
                current_state=risk.state,
                at_ms=at_ms,
                transcript_segments=tuple(dict(item) for item in selected),
                evidence_chain=tuple(
                    dict(item) for item in risk.evidence_chain if item.get("source") != "llm"
                ),
                visual_inputs=tuple(dict(item) for item in visual_inputs),
            )
        )
        if submitted:
            session.last_llm_review_id = review_id

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
