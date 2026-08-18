from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REVIEW_SCHEMA_VERSION = "radar_fall_review_candidates_v1"
MMFALL_ANCHOR_SEMANTICS = (
    "mmFall DS2 ground-truth fall frame index; it is not a verified impact "
    "timestamp and must not be used directly as a pre-fall training label"
)


@dataclass(frozen=True, slots=True)
class MmFallReviewSummary:
    replay_file: str
    anchor_file: str
    output_file: str
    replay_sha256: str
    anchor_sha256: str
    frame_count: int
    candidate_count: int
    duration_seconds: float
    schema_version: str = REVIEW_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class _ReplayMetadata:
    frame_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    device_id: str
    room: str
    anchor_seconds: Mapping[int, float]

    @property
    def duration_seconds(self) -> float:
        return (self.last_timestamp - self.first_timestamp).total_seconds()


def create_mmfall_review_candidates(
    replay_path: str | Path,
    anchor_path: str | Path,
    output_path: str | Path,
    *,
    session_id: str,
    index_base: int = 0,
    review_before_seconds: float = 3.0,
    review_after_seconds: float = 2.0,
) -> MmFallReviewSummary:
    """Create a non-training review document from mmFall DS2 anchors.

    The mmFall CSV values are dataset-provided fall-frame anchors. They are
    useful for locating events, but the repository does not establish that
    they are exact impact timestamps. This function therefore writes a
    separate review schema with all causal event timestamps left as ``null``.
    It never emits ``radar_fall_annotations_v1`` training labels.
    """

    source = Path(replay_path).resolve()
    anchors_source = Path(anchor_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"replay file does not exist: {source}")
    if not anchors_source.is_file():
        raise FileNotFoundError(
            f"anchor file does not exist: {anchors_source}"
        )
    if source.suffix.lower() != ".jsonl":
        raise ValueError("replay_path must end with .jsonl")
    if anchors_source.suffix.lower() != ".csv":
        raise ValueError("anchor_path must end with .csv")
    if destination.suffix.lower() != ".json":
        raise ValueError("output_path must end with .json")
    if not session_id.strip():
        raise ValueError("session_id must not be blank")
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    for value, name in (
        (review_before_seconds, "review_before_seconds"),
        (review_after_seconds, "review_after_seconds"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")

    raw_anchor_indices = _load_anchor_indices(anchors_source)
    anchor_indices = tuple(index - index_base for index in raw_anchor_indices)
    if any(index < 0 for index in anchor_indices):
        raise ValueError("anchor index is negative after applying index_base")
    replay = _inspect_replay(source, frozenset(anchor_indices))
    missing = sorted(set(anchor_indices).difference(replay.anchor_seconds))
    if missing:
        raise ValueError(
            "anchor indices exceed replay frame range: "
            + ", ".join(str(index + index_base) for index in missing)
        )

    candidates: list[dict[str, Any]] = []
    for candidate_number, (raw_index, zero_based_index) in enumerate(
        zip(raw_anchor_indices, anchor_indices, strict=True), start=1
    ):
        anchor_seconds = replay.anchor_seconds[zero_based_index]
        candidates.append(
            {
                "candidate_id": f"fall_candidate_{candidate_number:03d}",
                "dataset_anchor_frame": raw_index,
                "dataset_anchor_seconds": round(anchor_seconds, 6),
                "review_start_seconds": round(
                    max(0.0, anchor_seconds - review_before_seconds), 6
                ),
                "review_end_seconds": round(
                    min(
                        replay.duration_seconds,
                        anchor_seconds + review_after_seconds,
                    ),
                    6,
                ),
                "review_status": "pending",
                "loss_of_balance_onset_seconds": None,
                "pre_impact_seconds": None,
                "impact_seconds": None,
                "post_fall_end_seconds": None,
                "subject_group_id": None,
                "track_id": None,
                "notes": "",
            }
        )

    replay_sha256 = _sha256(source)
    anchor_sha256 = _sha256(anchors_source)
    payload = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "training_eligible": False,
        "warning": (
            "This is a review aid, not a training annotation. Complete and "
            "independently verify all four event timestamps before promotion."
        ),
        "session": {
            "session_id": session_id.strip(),
            "source_file": str(source),
            "source_sha256": replay_sha256,
            "frame_count": replay.frame_count,
            "duration_seconds": round(replay.duration_seconds, 6),
            "device_id": replay.device_id,
            "room": replay.room,
            "scene_scope": "single_target_unverified",
        },
        "anchor_source": {
            "file": str(anchors_source),
            "sha256": anchor_sha256,
            "index_base": index_base,
            "semantics": MMFALL_ANCHOR_SEMANTICS,
        },
        "fall_candidates": candidates,
    }
    _atomic_write_json(destination, payload)
    return MmFallReviewSummary(
        replay_file=str(source),
        anchor_file=str(anchors_source),
        output_file=str(destination),
        replay_sha256=replay_sha256,
        anchor_sha256=anchor_sha256,
        frame_count=replay.frame_count,
        candidate_count=len(candidates),
        duration_seconds=replay.duration_seconds,
    )


def _load_anchor_indices(path: Path) -> tuple[int, ...]:
    tokens = path.read_text(encoding="utf-8-sig").replace(",", " ").split()
    if not tokens:
        raise ValueError("anchor file is empty")
    indices: list[int] = []
    for token in tokens:
        try:
            numeric = float(token)
        except ValueError as exc:
            raise ValueError(f"invalid anchor index: {token!r}") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"anchor index must be an integer: {token!r}")
        indices.append(int(numeric))
    if indices != sorted(indices):
        raise ValueError("anchor indices must be sorted")
    if len(indices) != len(set(indices)):
        raise ValueError("anchor indices must be unique")
    return tuple(indices)


def _inspect_replay(
    path: Path,
    requested_anchor_indices: frozenset[int],
) -> _ReplayMetadata:
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    device_id: str | None = None
    room: str | None = None
    anchor_seconds: dict[int, float] = {}
    frame_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid replay JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"replay line {line_number} must be a JSON object"
                )
            timestamp = _parse_timestamp(value.get("timestamp"), line_number)
            current_device = str(value.get("device_id") or "").strip()
            current_room = str(value.get("room") or "").strip()
            if not current_device or not current_room:
                raise ValueError(
                    f"replay line {line_number} needs device_id and room"
                )
            if first_timestamp is None:
                first_timestamp = timestamp
                device_id = current_device
                room = current_room
            else:
                if timestamp <= last_timestamp:  # type: ignore[operator]
                    raise ValueError("replay timestamps must be strictly increasing")
                if current_device != device_id or current_room != room:
                    raise ValueError(
                        "replay device_id and room must stay constant"
                    )
            last_timestamp = timestamp
            if frame_count in requested_anchor_indices:
                anchor_seconds[frame_count] = (
                    timestamp - first_timestamp
                ).total_seconds()
            frame_count += 1

    if first_timestamp is None or last_timestamp is None:
        raise ValueError("replay file has no frames")
    if frame_count < 2:
        raise ValueError("replay file must contain at least two frames")
    return _ReplayMetadata(
        frame_count=frame_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        device_id=device_id or "",
        room=room or "",
        anchor_seconds=anchor_seconds,
    )


def _parse_timestamp(value: Any, line_number: int) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"replay line {line_number} timestamp must be text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"replay line {line_number} timestamp must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"replay line {line_number} timestamp needs timezone"
        )
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create non-training mmFall review candidates from DS2 anchors."
        )
    )
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--review-before-seconds", type=float, default=3.0)
    parser.add_argument("--review-after-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = create_mmfall_review_candidates(
        args.replay,
        args.anchors,
        args.output,
        session_id=args.session_id,
        index_base=args.index_base,
        review_before_seconds=args.review_before_seconds,
        review_after_seconds=args.review_after_seconds,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
