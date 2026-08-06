from datetime import UTC, datetime, timedelta

from app.modules.fraud.session_tracker import FraudSessionTracker


def test_segments_within_idle_window_share_session() -> None:
    tracker = FraudSessionTracker()
    started_at = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)

    first = tracker.session_for_segment(
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=5),
    )
    second = tracker.session_for_segment(
        started_at=started_at + timedelta(seconds=20),
        ended_at=started_at + timedelta(seconds=24),
    )

    assert second == first


def test_thirty_seconds_of_silence_starts_new_session() -> None:
    tracker = FraudSessionTracker()
    started_at = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)
    first = tracker.session_for_segment(
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=5),
    )

    second = tracker.session_for_segment(
        started_at=started_at + timedelta(seconds=35),
        ended_at=started_at + timedelta(seconds=38),
    )

    assert second != first


def test_new_phone_call_and_ten_minute_limit_start_new_sessions() -> None:
    tracker = FraudSessionTracker()
    started_at = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)
    first = tracker.observe_phone_call(started_at)
    same_call = tracker.observe_phone_call(started_at + timedelta(seconds=10))
    next_call = tracker.observe_phone_call(started_at + timedelta(seconds=40))
    after_limit = tracker.session_for_segment(
        started_at=started_at + timedelta(minutes=10),
        ended_at=started_at + timedelta(minutes=10, seconds=5),
    )

    assert same_call == first
    assert next_call != first
    assert after_limit != next_call
