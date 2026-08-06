from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class FraudSessionIdentity:
    session_id: str
    started_at: datetime


class FraudSessionTracker:
    """Assigns independent fraud sessions to one device's live audio timeline."""

    def __init__(
        self,
        *,
        idle_timeout: timedelta = timedelta(seconds=30),
        max_duration: timedelta = timedelta(minutes=10),
    ) -> None:
        self._idle_timeout = idle_timeout
        self._max_duration = max_duration
        self._active: FraudSessionIdentity | None = None
        self._last_activity_at: datetime | None = None
        self._last_phone_call_at: datetime | None = None

    @property
    def active_session_id(self) -> str | None:
        return self._active.session_id if self._active is not None else None

    def session_for_segment(self, *, started_at: datetime, ended_at: datetime) -> str:
        if ended_at < started_at:
            raise ValueError("segment ended_at must not be before started_at")
        if self._needs_new_session(started_at):
            self._start_new(started_at)
        self._last_activity_at = max(self._last_activity_at or ended_at, ended_at)
        assert self._active is not None
        return self._active.session_id

    def observe_phone_call(self, occurred_at: datetime) -> str:
        is_new_call = (
            self._last_phone_call_at is None
            or occurred_at - self._last_phone_call_at >= self._idle_timeout
        )
        if is_new_call:
            self._start_new(occurred_at)
        self._last_phone_call_at = occurred_at
        self._last_activity_at = max(self._last_activity_at or occurred_at, occurred_at)
        assert self._active is not None
        return self._active.session_id

    def _needs_new_session(self, at: datetime) -> bool:
        if self._active is None:
            return True
        if at - self._active.started_at >= self._max_duration:
            return True
        return (
            self._last_activity_at is not None and at - self._last_activity_at >= self._idle_timeout
        )

    def _start_new(self, at: datetime) -> None:
        self._active = FraudSessionIdentity(
            session_id=f"fraud-{at.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:10]}",
            started_at=at,
        )
        self._last_activity_at = at
