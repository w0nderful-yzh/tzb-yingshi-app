"""Decision-level gating state machine for pre-fall evidence.

Purpose
-------
The frozen TCN/PointNet predictors emit a continuous pre_fall_score. A raw
score crossing the threshold is a *window-level* event; it is not yet a
*final* alert. This module adds decision-level gating that consumes the score
stream (optionally with a centroid-z / point-count side channel) and decides
whether to escalate to a formal event.

Why this helps false alarms
---------------------------
The dominant false-alarm mode is a *brief* high score caused by a controlled
lowering (sit down, squat, bend) followed by recovery. A real fall produces a
high score that *persists* while the body stays low / ballistic. Gating rules:

1. Confirmation windows (already in live predictors): require several
   consecutive high windows before IMMINENT.
2. Recovery gate: if an IMMINENT-pending state is followed by scores dropping
   back below threshold within a short window, mark the episode
   SUPPRESSED_RECOVERY instead of escalating. This directly removes
   "sit-and-stand", "squat-and-rise" false alarms.
3. Optional persistence confirm: if enabled, a formal alert only fires after
   the IMMINENT state persists for a minimum duration. This separates a
   sustained fall signature from a transient spike.

Contract
--------
This module does NOT modify the frozen checkpoint, the feature extractor, the
model threshold, or the live inference chain. It sits on top of any predictor
that emits a timestamped score. It is fully deterministic and unit-testable.

Version: radar_decision_gate_v1
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


GATE_SCHEMA_VERSION = "radar_decision_gate_v1"


class GateState(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    IMMINENT = "IMMINENT"
    SUPPRESSED_RECOVERY = "SUPPRESSED_RECOVERY"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True, slots=True)
class GateDecisionV1:
    schema_version: str
    timestamp: str
    state: GateState
    pre_fall_score: float
    consecutive_high_windows: int
    recovery_window_active: bool
    recovery_count: int
    formal_alert: bool
    suppressed_reason: str | None
    episode_id: str | None
    # optional side-channel values echoed through for observability
    centroid_z: float | None
    point_count: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "pre_fall_score": self.pre_fall_score,
            "consecutive_high_windows": self.consecutive_high_windows,
            "recovery_window_active": self.recovery_window_active,
            "recovery_count": self.recovery_count,
            "formal_alert": self.formal_alert,
            "suppressed_reason": self.suppressed_reason,
            "episode_id": self.episode_id,
            "centroid_z": self.centroid_z,
            "point_count": self.point_count,
        }


class DecisionGateV1:
    """Deterministic decision-level gating over a score stream."""

    def __init__(
        self,
        *,
        threshold: float,
        confirmation_windows: int = 3,
        recovery_windows: int = 2,
        recovery_window_seconds: float = 1.5,
        persist_confirm_seconds: float = 0.0,
        emit_formal_alert: bool = True,
    ) -> None:
        if confirmation_windows < 1:
            raise ValueError("confirmation_windows must be at least one")
        if recovery_windows < 1:
            raise ValueError("recovery_windows must be at least one")
        if recovery_window_seconds <= 0.0:
            raise ValueError("recovery_window_seconds must be positive")
        if persist_confirm_seconds < 0.0:
            raise ValueError("persist_confirm_seconds must be non-negative")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = float(threshold)
        self.confirmation_windows = int(confirmation_windows)
        self.recovery_windows = int(recovery_windows)
        self.recovery_window_seconds = float(recovery_window_seconds)
        self.persist_confirm_seconds = float(persist_confirm_seconds)
        self.emit_formal_alert = bool(emit_formal_alert)

        self._consecutive_high = 0
        self._recovery_active = False
        self._recovery_count = 0
        self._imminent_since: datetime | None = None
        self._episode_id: str | None = None
        self._last_timestamp: datetime | None = None

    def reset(self) -> None:
        self._consecutive_high = 0
        self._recovery_active = False
        self._recovery_count = 0
        self._imminent_since = None
        self._episode_id = None
        self._last_timestamp = None

    def consume(
        self,
        *,
        timestamp: datetime,
        score: float,
        centroid_z: float | None = None,
        point_count: int | None = None,
    ) -> GateDecisionV1:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not (0.0 <= score <= 1.0):
            raise ValueError("score must be in [0, 1]")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._last_timestamp = timestamp

        high = score >= self.threshold
        suppressed_reason: str | None = None
        formal_alert = False

        if high:
            # Reset recovery tracking whenever we are high again.
            self._recovery_active = False
            self._recovery_count = 0
            self._consecutive_high += 1
            if self._consecutive_high >= self.confirmation_windows:
                if self._imminent_since is None:
                    self._imminent_since = timestamp
                    self._episode_id = _new_episode_id()
                state = GateState.IMMINENT
                # Persistence confirm: if configured and the IMMINENT state has
                # held for long enough, escalate to CONFIRMED / formal alert.
                if (
                    self.persist_confirm_seconds > 0.0
                    and timestamp - self._imminent_since
                    >= timedelta(seconds=self.persist_confirm_seconds)
                ):
                    state = GateState.CONFIRMED
                    formal_alert = self.emit_formal_alert
            else:
                state = GateState.WATCH
                self._imminent_since = None
        else:
            # Score below threshold.
            if self._imminent_since is not None:
                # We were in an IMMINENT episode; the score dropped. Begin the
                # recovery window. If the score stays low for recovery_windows,
                # treat the whole episode as a controlled recovery -> suppress.
                self._recovery_active = True
                self._recovery_count += 1
                if self._recovery_count >= self.recovery_windows:
                    suppressed_reason = (
                        "controlled_recovery: high score did not persist"
                    )
                    state = GateState.SUPPRESSED_RECOVERY
                    self._reset_episode()
                else:
                    # Still inside recovery window; keep reporting IMMINENT but
                    # with recovery_active so downstream can observe.
                    state = GateState.IMMINENT
            else:
                self._consecutive_high = 0
                state = GateState.NORMAL

        return GateDecisionV1(
            schema_version=GATE_SCHEMA_VERSION,
            timestamp=timestamp.isoformat(),
            state=state,
            pre_fall_score=float(score),
            consecutive_high_windows=self._consecutive_high,
            recovery_window_active=self._recovery_active,
            recovery_count=self._recovery_count,
            formal_alert=formal_alert,
            suppressed_reason=suppressed_reason,
            episode_id=self._episode_id,
            centroid_z=centroid_z,
            point_count=point_count,
        )

    def _reset_episode(self) -> None:
        self._consecutive_high = 0
        self._recovery_active = False
        self._recovery_count = 0
        self._imminent_since = None
        self._episode_id = None


def _new_episode_id() -> str:
    import uuid

    return f"prefall-episode-{uuid.uuid4().hex[:12]}"


def _build_parser() -> Any:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="DecisionGateV1 CLI smoke test (no file processing)."
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--confirmation-windows", type=int, default=3)
    parser.add_argument("--recovery-windows", type=int, default=2)
    parser.add_argument("--recovery-window-seconds", type=float, default=1.5)
    parser.add_argument("--persist-confirm-seconds", type=float, default=0.0)
    return parser


if __name__ == "__main__":  # pragma: no cover
    import json

    args = _build_parser().parse_args()
    gate = DecisionGateV1(
        threshold=args.threshold,
        confirmation_windows=args.confirmation_windows,
        recovery_windows=args.recovery_windows,
        recovery_window_seconds=args.recovery_window_seconds,
        persist_confirm_seconds=args.persist_confirm_seconds,
    )
    base = datetime(2026, 8, 10, 12, 0, 0, tzinfo=datetime.now().astimezone().tzinfo)
    # Synthetic sequence: high x3 -> drop x2 (should suppress), then high x4 (imminent)
    seq = [0.2, 0.6, 0.7, 0.8, 0.1, 0.05, 0.6, 0.7, 0.8, 0.9, 0.85]
    for i, s in enumerate(seq):
        d = gate.consume(
            timestamp=base + timedelta(seconds=i * 0.1),
            score=s,
            centroid_z=0.1,
            point_count=20,
        )
        print(json.dumps(d.to_dict()))
