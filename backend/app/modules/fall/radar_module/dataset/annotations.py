from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from radar_module.contracts import Room


ANNOTATION_SCHEMA_VERSION = "radar_fall_annotations_v1"
DEFAULT_MIN_LEAD_SECONDS = 0.2
DEFAULT_MAX_LEAD_SECONDS = 1.5
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SceneScope(str, Enum):
    SINGLE_TARGET = "single_target"
    MULTI_TARGET = "multi_target"
    ROOM_ONLY = "room_only"


class AnnotationCoverage(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class HardNegativeAction(str, Enum):
    SIT_DOWN = "sit_down"
    LIE_DOWN = "lie_down"
    BEND = "bend"
    CROUCH = "crouch"
    JUMP = "jump"
    PICK_UP_OBJECT = "pick_up_object"
    FAST_TURN = "fast_turn"
    OCCLUSION_REAPPEARANCE = "occlusion_reappearance"


class PredictionWindowLabel(str, Enum):
    PRE_FALL = "pre_fall"
    NEGATIVE = "negative"
    DETECTION_ONLY = "detection_only"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    session_id: str
    source_file: str
    source_sha256: str
    duration_seconds: float
    room: Room
    scene_scope: SceneScope
    annotation_coverage: AnnotationCoverage
    max_concurrent_tracks: int

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session.session_id")
        _require_text(self.source_file, "session.source_file")
        if not _SHA256_PATTERN.fullmatch(self.source_sha256):
            raise ValueError("session.source_sha256 must be 64 lowercase hex characters")
        _require_nonnegative_finite(self.duration_seconds, "session.duration_seconds")
        if self.duration_seconds == 0:
            raise ValueError("session.duration_seconds must be greater than zero")
        if self.max_concurrent_tracks < 1:
            raise ValueError("session.max_concurrent_tracks must be at least one")
        if (
            self.scene_scope is SceneScope.MULTI_TARGET
            and self.max_concurrent_tracks < 2
        ):
            raise ValueError(
                "multi_target sessions must allow at least two concurrent tracks"
            )


@dataclass(frozen=True, slots=True)
class FallEventAnnotation:
    event_id: str
    subject_group_id: str
    loss_of_balance_onset_seconds: float
    pre_impact_seconds: float
    impact_seconds: float
    post_fall_end_seconds: float
    track_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.event_id, "fall_event.event_id")
        _require_text(self.subject_group_id, "fall_event.subject_group_id")
        if self.track_id is not None:
            _require_text(self.track_id, "fall_event.track_id")
        times = (
            self.loss_of_balance_onset_seconds,
            self.pre_impact_seconds,
            self.impact_seconds,
            self.post_fall_end_seconds,
        )
        for value in times:
            _require_nonnegative_finite(value, "fall_event timestamp")
        if not times[0] <= times[1] < times[2] <= times[3]:
            raise ValueError(
                "fall event times must satisfy onset <= pre_impact < impact "
                "<= post_fall_end"
            )


@dataclass(frozen=True, slots=True)
class HardNegativeInterval:
    interval_id: str
    action: HardNegativeAction
    start_seconds: float
    end_seconds: float
    track_id: str | None = None

    def __post_init__(self) -> None:
        _validate_interval(
            self.interval_id,
            self.start_seconds,
            self.end_seconds,
            self.track_id,
            "hard_negative",
        )


@dataclass(frozen=True, slots=True)
class UnknownInterval:
    interval_id: str
    reason: str
    start_seconds: float
    end_seconds: float
    track_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.reason, "unknown.reason")
        _validate_interval(
            self.interval_id,
            self.start_seconds,
            self.end_seconds,
            self.track_id,
            "unknown",
        )


@dataclass(frozen=True, slots=True)
class AnnotationDocument:
    schema_version: str
    session: SessionMetadata
    fall_events: tuple[FallEventAnnotation, ...]
    hard_negatives: tuple[HardNegativeInterval, ...]
    unknown_intervals: tuple[UnknownInterval, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported annotation schema: {self.schema_version}"
            )
        identifiers = [event.event_id for event in self.fall_events]
        identifiers.extend(item.interval_id for item in self.hard_negatives)
        identifiers.extend(item.interval_id for item in self.unknown_intervals)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("annotation identifiers must be unique within a session")
        for event in self.fall_events:
            if event.post_fall_end_seconds > self.session.duration_seconds:
                raise ValueError(f"event {event.event_id} exceeds session duration")
        for interval in (*self.hard_negatives, *self.unknown_intervals):
            if interval.end_seconds > self.session.duration_seconds:
                raise ValueError(
                    f"interval {interval.interval_id} exceeds session duration"
                )


@dataclass(frozen=True, slots=True)
class WindowLabelDecision:
    label: PredictionWindowLabel
    reason: str
    event_id: str | None = None
    seconds_to_impact: float | None = None


def load_annotation_document(path: str | Path) -> AnnotationDocument:
    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("annotation document must be a JSON object")
    return annotation_document_from_dict(payload)


def annotation_document_from_dict(
    payload: Mapping[str, Any],
) -> AnnotationDocument:
    _expect_keys(
        payload,
        required={
            "schema_version",
            "session",
            "fall_events",
            "hard_negatives",
            "unknown_intervals",
        },
        context="document",
    )
    session_payload = _require_mapping(payload["session"], "session")
    _expect_keys(
        session_payload,
        required={
            "session_id",
            "source_file",
            "source_sha256",
            "duration_seconds",
            "room",
            "scene_scope",
            "annotation_coverage",
            "max_concurrent_tracks",
        },
        context="session",
    )
    session = SessionMetadata(
        session_id=str(session_payload["session_id"]),
        source_file=str(session_payload["source_file"]),
        source_sha256=str(session_payload["source_sha256"]),
        duration_seconds=float(session_payload["duration_seconds"]),
        room=Room(session_payload["room"]),
        scene_scope=SceneScope(session_payload["scene_scope"]),
        annotation_coverage=AnnotationCoverage(
            session_payload["annotation_coverage"]
        ),
        max_concurrent_tracks=int(session_payload["max_concurrent_tracks"]),
    )
    return AnnotationDocument(
        schema_version=str(payload["schema_version"]),
        session=session,
        fall_events=tuple(
            _parse_fall_event(item)
            for item in _require_list(payload["fall_events"], "fall_events")
        ),
        hard_negatives=tuple(
            _parse_hard_negative(item)
            for item in _require_list(
                payload["hard_negatives"], "hard_negatives"
            )
        ),
        unknown_intervals=tuple(
            _parse_unknown(item)
            for item in _require_list(
                payload["unknown_intervals"], "unknown_intervals"
            )
        ),
    )


def decide_prediction_window(
    annotations: AnnotationDocument,
    window_end_seconds: float,
    *,
    target_track_id: str | None = None,
    min_lead_seconds: float = DEFAULT_MIN_LEAD_SECONDS,
    max_lead_seconds: float = DEFAULT_MAX_LEAD_SECONDS,
) -> WindowLabelDecision:
    """Label a history window without leaking impact/post-impact frames.

    With ``target_track_id=None`` the decision is room-level and any annotated
    fall can make the window positive. A track-specific decision only uses an
    event carrying the same anonymous, session-local track ID.
    """

    _require_nonnegative_finite(window_end_seconds, "window_end_seconds")
    if window_end_seconds > annotations.session.duration_seconds:
        raise ValueError("window_end_seconds exceeds session duration")
    if not 0 <= min_lead_seconds < max_lead_seconds:
        raise ValueError("lead time must satisfy 0 <= min < max")
    if (
        target_track_id is not None
        and annotations.session.scene_scope is SceneScope.ROOM_ONLY
    ):
        raise ValueError("room_only annotations cannot produce track labels")

    for interval in annotations.unknown_intervals:
        if _interval_applies(interval.track_id, target_track_id) and (
            interval.start_seconds <= window_end_seconds <= interval.end_seconds
        ):
            return WindowLabelDecision(
                PredictionWindowLabel.EXCLUDED,
                f"unknown interval: {interval.reason}",
            )

    relevant_events = tuple(
        event
        for event in annotations.fall_events
        if (
            target_track_id is None
            or event.track_id == target_track_id
        )
    )
    for event in relevant_events:
        seconds_to_impact = event.impact_seconds - window_end_seconds
        if min_lead_seconds <= seconds_to_impact <= max_lead_seconds:
            return WindowLabelDecision(
                PredictionWindowLabel.PRE_FALL,
                "impact lies inside the prediction horizon",
                event.event_id,
                seconds_to_impact,
            )
        if (
            event.impact_seconds - min_lead_seconds
            < window_end_seconds
            <= event.post_fall_end_seconds
        ):
            return WindowLabelDecision(
                PredictionWindowLabel.DETECTION_ONLY,
                "window is too close to or after impact",
                event.event_id,
                seconds_to_impact,
            )
        if (
            event.loss_of_balance_onset_seconds
            <= window_end_seconds
            < event.impact_seconds - max_lead_seconds
        ):
            return WindowLabelDecision(
                PredictionWindowLabel.EXCLUDED,
                "transition is outside the declared prediction horizon",
                event.event_id,
                seconds_to_impact,
            )

    if target_track_id is not None:
        for event in annotations.fall_events:
            if event.track_id is not None:
                continue
            seconds_to_impact = event.impact_seconds - window_end_seconds
            if -0.001 <= seconds_to_impact <= max_lead_seconds:
                return WindowLabelDecision(
                    PredictionWindowLabel.EXCLUDED,
                    "room-level fall cannot be attributed to this track",
                    event.event_id,
                    seconds_to_impact,
                )

    for interval in annotations.hard_negatives:
        if _interval_applies(interval.track_id, target_track_id) and (
            interval.start_seconds <= window_end_seconds <= interval.end_seconds
        ):
            return WindowLabelDecision(
                PredictionWindowLabel.NEGATIVE,
                f"hard negative: {interval.action.value}",
            )

    if annotations.session.annotation_coverage is AnnotationCoverage.COMPLETE:
        return WindowLabelDecision(
            PredictionWindowLabel.NEGATIVE,
            "completely annotated non-event time",
        )
    return WindowLabelDecision(
        PredictionWindowLabel.EXCLUDED,
        "unannotated time in a partially annotated session",
    )


def _parse_fall_event(value: Any) -> FallEventAnnotation:
    item = _require_mapping(value, "fall_event")
    _expect_keys(
        item,
        required={
            "event_id",
            "subject_group_id",
            "loss_of_balance_onset_seconds",
            "pre_impact_seconds",
            "impact_seconds",
            "post_fall_end_seconds",
        },
        optional={"track_id"},
        context="fall_event",
    )
    return FallEventAnnotation(
        event_id=str(item["event_id"]),
        subject_group_id=str(item["subject_group_id"]),
        track_id=_optional_text(item.get("track_id")),
        loss_of_balance_onset_seconds=float(
            item["loss_of_balance_onset_seconds"]
        ),
        pre_impact_seconds=float(item["pre_impact_seconds"]),
        impact_seconds=float(item["impact_seconds"]),
        post_fall_end_seconds=float(item["post_fall_end_seconds"]),
    )


def _parse_hard_negative(value: Any) -> HardNegativeInterval:
    item = _require_mapping(value, "hard_negative")
    _expect_keys(
        item,
        required={"interval_id", "action", "start_seconds", "end_seconds"},
        optional={"track_id"},
        context="hard_negative",
    )
    return HardNegativeInterval(
        interval_id=str(item["interval_id"]),
        action=HardNegativeAction(item["action"]),
        start_seconds=float(item["start_seconds"]),
        end_seconds=float(item["end_seconds"]),
        track_id=_optional_text(item.get("track_id")),
    )


def _parse_unknown(value: Any) -> UnknownInterval:
    item = _require_mapping(value, "unknown")
    _expect_keys(
        item,
        required={"interval_id", "reason", "start_seconds", "end_seconds"},
        optional={"track_id"},
        context="unknown",
    )
    return UnknownInterval(
        interval_id=str(item["interval_id"]),
        reason=str(item["reason"]),
        start_seconds=float(item["start_seconds"]),
        end_seconds=float(item["end_seconds"]),
        track_id=_optional_text(item.get("track_id")),
    )


def _validate_interval(
    interval_id: str,
    start_seconds: float,
    end_seconds: float,
    track_id: str | None,
    context: str,
) -> None:
    _require_text(interval_id, f"{context}.interval_id")
    if track_id is not None:
        _require_text(track_id, f"{context}.track_id")
    _require_nonnegative_finite(start_seconds, f"{context}.start_seconds")
    _require_nonnegative_finite(end_seconds, f"{context}.end_seconds")
    if start_seconds >= end_seconds:
        raise ValueError(f"{context} start_seconds must be before end_seconds")


def _interval_applies(
    interval_track_id: str | None,
    target_track_id: str | None,
) -> bool:
    return (
        target_track_id is None
        or interval_track_id is None
        or interval_track_id == target_track_id
    )


def _expect_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    unexpected = value.keys() - required - optional
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {sorted(unexpected)}")


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _require_text(value: str, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-blank text")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    _require_text(text, "optional text")
    return text


def _require_nonnegative_finite(value: float, context: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{context} must be a nonnegative finite number")
