import logging

import pytest

from app.modules.fraud.latency import (
    FraudLatencyTrace,
    configure_tracing,
    finish_trace,
    latency_stage,
    privacy_digest,
    record_span,
    start_trace,
)


@pytest.fixture(autouse=True)
def _enable_tracing():
    configure_tracing(enabled=True)
    yield
    configure_tracing(enabled=False)


def test_start_trace_returns_none_when_disabled() -> None:
    configure_tracing(enabled=False)
    assert (
        start_trace(
            device_id="camera-01",
            session_id="s1",
            source_event_id="e1",
            transcript_status="FINAL",
        )
        is None
    )


def test_stage_durations_are_non_negative_and_ordered() -> None:
    trace = start_trace(
        device_id="camera-01",
        session_id="s1",
        source_event_id="e1",
        transcript_status="FINAL",
    )
    assert trace is not None
    with latency_stage("evidence_extract"):
        pass
    with latency_stage("state_machine"):
        pass
    snapshot = finish_trace(trace)
    assert snapshot is not None
    assert list(snapshot.stages) == ["evidence_extract", "state_machine"]
    assert all(duration >= 0 for duration in snapshot.stages.values())
    assert snapshot.total_ms >= 0


def test_nested_start_trace_returns_none_for_non_owner() -> None:
    outer = start_trace(
        device_id="camera-01",
        session_id="s1",
        source_event_id="e1",
        transcript_status="PARTIAL",
    )
    assert outer is not None
    inner = start_trace(
        device_id="camera-01",
        session_id="s1",
        source_event_id="e1",
        transcript_status="PARTIAL",
    )
    assert inner is None
    finish_trace(outer)


def test_latency_stage_is_noop_without_active_trace() -> None:
    # No active trace: must not raise.
    with latency_stage("asr_recognize"):
        pass
    record_span("queue_wait", 1.5)
    assert finish_trace(None) is None


def test_record_span_adds_custom_duration() -> None:
    trace = start_trace(
        device_id="camera-01",
        session_id="s1",
        source_event_id="e1",
        transcript_status="FINAL",
    )
    assert trace is not None
    record_span("queue_wait", 12.5)
    snapshot = finish_trace(trace)
    assert snapshot is not None
    assert snapshot.stages["queue_wait"] == 12.5


def test_finish_trace_clears_context() -> None:
    trace = start_trace(
        device_id="camera-01",
        session_id="s1",
        source_event_id="e1",
        transcript_status="FINAL",
    )
    assert trace is not None
    finish_trace(trace)
    # After finishing, a new trace can be started (context was cleared).
    second = start_trace(
        device_id="camera-01",
        session_id="s2",
        source_event_id="e2",
        transcript_status="FINAL",
    )
    assert second is not None
    finish_trace(second)


def test_snapshot_contains_only_digests_not_raw_ids(caplog: pytest.LogCaptureFixture) -> None:
    raw_device = "camera-device-secret-01"
    raw_session = "session-secret-abc"
    raw_event = "event-secret-xyz"
    trace = start_trace(
        device_id=raw_device,
        session_id=raw_session,
        source_event_id=raw_event,
        transcript_status="FINAL",
    )
    assert trace is not None
    with latency_stage("state_machine"):
        pass
    with caplog.at_level(logging.INFO, logger="app.modules.fraud.latency"):
        snapshot = finish_trace(trace)
    assert snapshot is not None
    # Raw identifiers must never appear in the snapshot or the emitted log.
    for raw in (raw_device, raw_session, raw_event):
        assert raw not in str(snapshot.as_log_fields())
    assert raw_device not in caplog.text
    assert raw_session not in caplog.text
    assert raw_event not in caplog.text


def test_privacy_digest_is_irreversible_and_stable() -> None:
    digest_a = privacy_digest("camera-01")
    digest_b = privacy_digest("camera-01")
    assert digest_a == digest_b
    assert digest_a != "camera-01"
    assert len(digest_a) == 16
    assert privacy_digest("camera-02") != digest_a


def test_missing_stage_end_does_not_break_snapshot() -> None:
    # A trace with no spans should still snapshot cleanly.
    trace = FraudLatencyTrace(
        trace_id="t",
        device_digest="d",
        session_digest="s",
        source_event_digest="e",
        transcript_status="FINAL",
    )
    snapshot = trace.snapshot()
    assert snapshot.stages == {}
    assert snapshot.total_ms >= 0
