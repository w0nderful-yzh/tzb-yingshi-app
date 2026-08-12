"""Lightweight latency tracing for the fraud decision pipeline.

Tracing is strictly opt-in via a module-level flag configured from settings at
application startup. When disabled every helper is a no-op so business results
are unchanged and overhead is effectively zero. When enabled, identifiers are
recorded only as irreversible digests; transcripts, tokens, device credentials
and raw audio are never logged.

A single trace propagates through the async call chain (media worker → audio
service → session service → risk event repository) using a ContextVar, so no
function signatures change. The outermost component that starts a trace owns
finishing it; inner components only contribute spans.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

FraudLatencyStage = Literal[
    "pcm_ingest",
    "vad_segment",
    "task_enqueue",
    "task_dequeue",
    "task_drop",
    "queue_wait",
    "asr_recognize",
    "evidence_extract",
    "state_machine",
    "session_persist",
    "event_persist",
    "event_commit",
    "broker_publish",
]

_current: ContextVar[FraudLatencyTrace | None] = ContextVar(
    "fraud_latency_trace",
    default=None,
)
_enabled: bool = False


def configure_tracing(*, enabled: bool) -> None:
    """Enable or disable latency tracing process-wide.

    Called once from application startup based on settings. When disabled all
    helpers short-circuit and no trace objects are created.
    """

    global _enabled
    _enabled = enabled


def tracing_enabled() -> bool:
    return _enabled


def privacy_digest(value: object, *, length: int = 16) -> str:
    """Return an irreversible short digest for an identifier.

    Latency logs must never contain raw device IDs, session IDs or event IDs;
    only their digests are safe to emit.
    """

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def _now_ms() -> float:
    return time.monotonic_ns() / 1_000_000


@dataclass(frozen=True, slots=True)
class FraudLatencySpan:
    """One timed stage of the pipeline."""

    stage: str
    duration_ms: float


@dataclass(slots=True)
class FraudLatencyTrace:
    """Accumulates spans for a single analysis pipeline run."""

    trace_id: str
    device_digest: str
    session_digest: str
    source_event_digest: str
    transcript_status: str
    model_name: str | None = None
    device_type: str | None = None
    model_warmed: bool = False
    started_ms: float = field(default_factory=_now_ms)
    spans: list[FraudLatencySpan] = field(default_factory=list)

    def record(self, stage: str, duration_ms: float) -> None:
        if duration_ms < 0:
            duration_ms = 0.0
        self.spans.append(FraudLatencySpan(stage=stage, duration_ms=duration_ms))

    def snapshot(self) -> FraudLatencySnapshot:
        total_ms = _now_ms() - self.started_ms
        by_stage: dict[str, float] = {}
        for span in self.spans:
            by_stage[span.stage] = round(by_stage.get(span.stage, 0.0) + span.duration_ms, 3)
        return FraudLatencySnapshot(
            trace_id=self.trace_id,
            device_digest=self.device_digest,
            session_digest=self.session_digest,
            source_event_digest=self.source_event_digest,
            transcript_status=self.transcript_status,
            model_name=self.model_name,
            device_type=self.device_type,
            model_warmed=self.model_warmed,
            total_ms=round(max(total_ms, 0.0), 3),
            stages=by_stage,
        )


@dataclass(frozen=True, slots=True)
class FraudLatencySnapshot:
    """Aggregated, log-ready view of a completed trace."""

    trace_id: str
    device_digest: str
    session_digest: str
    source_event_digest: str
    transcript_status: str
    model_name: str | None
    device_type: str | None
    model_warmed: bool
    total_ms: float
    stages: dict[str, float]

    def as_log_fields(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "device": self.device_digest,
            "session": self.session_digest,
            "source_event": self.source_event_digest,
            "transcript_status": self.transcript_status,
            "model_name": self.model_name,
            "device_type": self.device_type,
            "model_warmed": self.model_warmed,
            "total_ms": self.total_ms,
            "stages": self.stages,
        }


def start_trace(
    *,
    device_id: str,
    session_id: str,
    source_event_id: str,
    transcript_status: str,
    model_name: str | None = None,
    device_type: str | None = None,
    model_warmed: bool = False,
) -> FraudLatencyTrace | None:
    """Start a trace and become its owner.

    Returns the trace only to the caller that actually started it. If tracing
    is disabled, or a trace is already active in this context, returns None so
    the caller knows it is not the owner and must not finish the trace.
    """

    if not _enabled or _current.get() is not None:
        return None
    trace = FraudLatencyTrace(
        trace_id=privacy_digest(f"{device_id}{session_id}{source_event_id}"),
        device_digest=privacy_digest(device_id),
        session_digest=privacy_digest(session_id),
        source_event_digest=privacy_digest(source_event_id),
        transcript_status=transcript_status,
        model_name=model_name,
        device_type=device_type,
        model_warmed=model_warmed,
    )
    _current.set(trace)
    return trace


@contextmanager
def latency_stage(stage: FraudLatencyStage) -> Iterator[None]:
    """Time a block as a pipeline stage.

    No-op when tracing is disabled or no trace is active in this context.
    """

    trace = _current.get()
    if trace is None:
        yield
        return
    start = _now_ms()
    try:
        yield
    finally:
        trace.record(stage, _now_ms() - start)


def record_span(stage: FraudLatencyStage, duration_ms: float) -> None:
    """Record an externally-measured duration (e.g. queue wait) on the trace."""

    trace = _current.get()
    if trace is not None:
        trace.record(stage, duration_ms)


def finish_trace(trace: FraudLatencyTrace | None) -> FraudLatencySnapshot | None:
    """Finish an owned trace, emit a structured log line and clear context.

    Only the owner (the caller that received a non-None trace from
    start_trace) should call this. Returns the snapshot for tests and the
    benchmark harness.
    """

    if trace is None:
        return None
    snapshot = trace.snapshot()
    if _current.get() is trace:
        _current.set(None)
    logger.info("fraud_latency", extra={"fraud_latency": snapshot.as_log_fields()})
    return snapshot
