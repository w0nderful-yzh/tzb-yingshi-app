import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from app.modules.fraud.latency import finish_trace, latency_stage, start_trace
from app.modules.fraud.llm import FraudLlmReviewQueue, FraudLlmReviewRequest
from app.modules.fraud.ports import (
    FraudRiskEventSink,
    FraudRiskEventWrite,
    FraudSessionRecord,
    FraudSessionStore,
    RecentFraudRiskStore,
    SemanticEvidenceRetriever,
)
from app.modules.fraud.risk_engine import (
    RISK_MODEL_NAME,
    RISK_MODEL_VERSION,
    build_risk_snapshot,
    to_epoch_ms,
)
from app.modules.fraud.risk_profile import recent_context_evidence
from app.modules.fraud.schemas import (
    FraudAnalyzeData,
    FraudAnalyzeRequest,
    FraudRiskSnapshot,
)
from app.modules.fraud.speech_risk import build_speech_events
from app.modules.fraud.visual_event_store import VisualEventStore

STRONG_ACTION_KINDS = frozenset(
    {"credential_request", "remote_control_instruction", "money_instruction"}
)
SYSTEM_RETRACT_REASON = "final_transcript_retracted_preliminary"


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
        preliminary_alert_enabled: bool = False,
        preliminary_min_confidence: float = 0.90,
        preliminary_stable_revisions: int = 2,
        preliminary_confirm_min_state_index: int = 2,
        semantic_retriever: SemanticEvidenceRetriever | None = None,
        recent_risk_store: RecentFraudRiskStore | None = None,
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
        self._preliminary_enabled = preliminary_alert_enabled
        self._preliminary_min_confidence = preliminary_min_confidence
        self._preliminary_stable_revisions = preliminary_stable_revisions
        self._preliminary_confirm_min_state_index = preliminary_confirm_min_state_index
        self._semantic_retriever = semantic_retriever
        self._recent_risk_store = recent_risk_store
        self._sessions: dict[tuple[str, str], _FraudSession] = {}
        self._lock = asyncio.Lock()

    async def analyze(self, payload: FraudAnalyzeRequest) -> FraudAnalyzeData:
        trace = start_trace(
            device_id=payload.device_id,
            session_id=payload.session_id,
            source_event_id=payload.source_event_id,
            transcript_status=payload.transcript_status,
            model_name=RISK_MODEL_NAME,
        )
        try:
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
                should_update = (
                    existing is not None and existing.get("transcript_status") == "PARTIAL"
                )
                if existing is None or should_update:
                    with latency_stage("evidence_extract"):
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
                            event_id_offset=(
                                len(session.speech_events) - (1 if should_update else 0)
                            ),
                        )
                    speech_event = speech_events[0]
                    speech_event["source_event_id"] = payload.source_event_id
                    if should_update and existing is not None:
                        stability = existing.get("partial_stability")
                        if isinstance(stability, dict):
                            speech_event["partial_stability"] = dict(stability)
                    session.speech_events[payload.source_event_id] = speech_event
                    status: Literal["accepted", "updated", "duplicate"] = (
                        "updated" if should_update else "accepted"
                    )
                else:
                    speech_event = existing
                    status = "duplicate"

                with latency_stage("state_machine"):
                    risk = await self._snapshot(session)
                if payload.transcript_status == "PARTIAL":
                    await self._maybe_preliminary(session, payload, speech_event, risk)
                elif payload.transcript_status == "FINAL":
                    await self._settle_preliminary(session, payload, speech_event, risk)
                if payload.transcript_status == "FINAL":
                    await self._submit_llm_review(session, risk)
                with latency_stage("session_persist"):
                    await self._persist_session(session)
                return FraudAnalyzeData(status=status, speech_event=speech_event, risk=risk)
        finally:
            finish_trace(trace)

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
        extra_evidence = [dict(item) for item in session.llm_evidence.values()]
        at_ms = (
            max(int(event["end_ms"]) for event in session.speech_events.values())
            if session.speech_events
            else to_epoch_ms(datetime.now(UTC))
        )
        if (
            self._semantic_retriever is not None
            and self._semantic_retriever.available
            and session.speech_events
        ):
            transcript = " ".join(
                str(event.get("text", ""))
                for event in sorted(
                    session.speech_events.values(),
                    key=lambda event: int(event["start_ms"]),
                )
            )[:6_000]
            extra_evidence.extend(
                await self._semantic_retriever.retrieve(
                    text=transcript,
                    session_id=session.session_id,
                )
            )
        if self._recent_risk_store is not None:
            context = await self._recent_risk_store.load_recent_context(
                device_id=session.device_id,
                session_id=session.session_id,
            )
            if context is not None:
                extra_evidence.extend(recent_context_evidence(context, at_ms=at_ms))
        return build_risk_snapshot(
            session_id=session.session_id,
            device_id=session.device_id,
            speech_events=list(session.speech_events.values()),
            visual_events=visual_events,
            elder_alone=session.elder_alone,
            memory_ms=self._memory_ms,
            extra_evidence=extra_evidence,
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

    async def _maybe_preliminary(
        self,
        session: _FraudSession,
        payload: FraudAnalyzeRequest,
        speech_event: dict[str, Any],
        risk: FraudRiskSnapshot,
    ) -> None:
        """Decide whether consecutive strong-action PARTIALs create a PRELIMINARY.

        Only strong action evidence above the confidence gate and free of
        protective evidence may trigger. Stability metadata lives on the
        speech event inside fraud_sessions JSONB so a process restart restores
        the same revision count and never creates a duplicate PRELIMINARY.
        """
        if not self._preliminary_enabled:
            return
        observations = speech_event.get("evidence_observations") or []
        if any(str(item.get("polarity")) == "protective" for item in observations):
            self._reset_partial_stability(speech_event)
            return
        strong = [
            item
            for item in observations
            if str(item.get("stage")) == "action"
            and str(item.get("strength")) == "strong"
            and str(item.get("kind")) in STRONG_ACTION_KINDS
            and float(item.get("confidence", 0.0)) >= self._preliminary_min_confidence
        ]
        if not strong:
            self._reset_partial_stability(speech_event)
            return
        kind = str(max(strong, key=lambda item: float(item.get("confidence", 0.0)))["kind"])
        stability = speech_event.get("partial_stability")
        previous = dict(stability) if isinstance(stability, dict) else {}
        if previous.get("candidate_kind") == kind:
            revisions = int(previous.get("revisions", 0)) + 1
        else:
            revisions = 1
        preliminary_created = bool(previous.get("preliminary_created"))
        speech_event["partial_stability"] = {
            "revisions": revisions,
            "candidate_kind": kind,
            "preliminary_created": preliminary_created,
        }
        if revisions < self._preliminary_stable_revisions or preliminary_created:
            return
        speech_event["partial_stability"]["preliminary_created"] = True
        evidence = risk.model_dump(mode="json")
        evidence.update(
            {
                "verification_status": "PRELIMINARY",
                "preliminary_source_event_id": payload.source_event_id,
                "preliminary_kind": kind,
                "preliminary_created_at": datetime.now(UTC).isoformat(),
            }
        )
        with latency_stage("event_persist"):
            await self._persist(
                risk,
                source_event_id=payload.source_event_id,
                verification_status="PRELIMINARY",
                evidence=evidence,
            )

    async def _settle_preliminary(
        self,
        session: _FraudSession,
        payload: FraudAnalyzeRequest,
        speech_event: dict[str, Any],
        risk: FraudRiskSnapshot,
    ) -> None:
        """Confirm or retract a PRELIMINARY once the FINAL transcript arrives.

        FINAL state >= confirm threshold confirms the same event with the
        formal S2-S5 level; FINAL falling back below threshold retracts it
        (system RESOLVE action, actor empty). Turns without a PRELIMINARY keep
        the existing session-level risk event behaviour.
        """
        stability = speech_event.get("partial_stability")
        preliminary_created = bool(
            isinstance(stability, dict) and stability.get("preliminary_created")
        )
        if not preliminary_created:
            if risk.state != "S0_NORMAL":
                with latency_stage("event_persist"):
                    await self._persist(risk)
            return
        if risk.state_index >= self._preliminary_confirm_min_state_index:
            evidence = risk.model_dump(mode="json")
            evidence.update(
                {
                    "verification_status": "CONFIRMED",
                    "preliminary_source_event_id": payload.source_event_id,
                    "preliminary_kind": (
                        str(stability.get("candidate_kind") or "unknown")
                        if isinstance(stability, dict)
                        else "unknown"
                    ),
                    "confirmed_at": datetime.now(UTC).isoformat(),
                }
            )
            with latency_stage("event_persist"):
                await self._persist(
                    risk,
                    source_event_id=payload.source_event_id,
                    verification_status="CONFIRMED",
                    evidence=evidence,
                )
            return
        if self._risk_event_sink is not None:
            with latency_stage("event_persist"):
                await self._risk_event_sink.retract_preliminary(
                    source_event_id=payload.source_event_id,
                    reason=SYSTEM_RETRACT_REASON,
                )

    @staticmethod
    def _reset_partial_stability(speech_event: dict[str, Any]) -> None:
        stability = speech_event.get("partial_stability")
        if isinstance(stability, dict) and stability.get("preliminary_created"):
            speech_event["partial_stability"] = {
                "revisions": 0,
                "candidate_kind": None,
                "preliminary_created": True,
            }

    async def _persist(
        self,
        risk: FraudRiskSnapshot,
        *,
        source_event_id: str | None = None,
        verification_status: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if self._risk_event_sink is None:
            return
        if source_event_id is None:
            stable_key = hashlib.sha256(f"{risk.device_id}\0{risk.session_id}".encode()).hexdigest()
            source_event_id = f"fraud-session:{stable_key}"
        await self._risk_event_sink.upsert(
            FraudRiskEventWrite(
                source_event_id=source_event_id,
                external_device_id=risk.device_id,
                risk_level=risk.risk_level,
                confidence=risk.confidence,
                summary=f"{risk.state_label}：{risk.transition_reason}"[:500],
                occurred_at=risk.occurred_at,
                received_at=datetime.now(UTC),
                evidence=evidence or risk.model_dump(mode="json"),
                model_name=RISK_MODEL_NAME,
                model_version=RISK_MODEL_VERSION,
                verification_status=verification_status,
            )
        )
