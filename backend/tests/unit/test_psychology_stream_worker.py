"""Unit tests for the streaming psychology worker's pure orchestration logic.

The worker module must stay importable without torch/pandas at top level (the
backend test env has neither). Tests that need pandas skip when it is absent.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

_HOME_DETECTION_PKG = (
    Path(__file__).resolve().parents[2] / "app" / "modules" / "psychology" / "home_detection_pkg"
)
sys.path.insert(0, str(_HOME_DETECTION_PKG))

import service.psychology_worker_main as worker  # noqa: E402


def _openface_csv_rows(num_frames: int, *, skip_frames: set[int] | None = None) -> list[dict]:
    """Build synthetic OpenFace rows: frame i has X_0 = i, Y_0 = i, rest zeroed."""
    au_columns = [f"AU{i:02d}_r" for i in (1, 2, 4, 5, 6, 9, 10, 12, 14, 15, 17, 20, 25, 26)]
    columns = (
        ["frame", "face_id"]
        + [f"X_{i}" for i in range(68)]
        + [f"Y_{i}" for i in range(68)]
        + [f"Z_{i}" for i in range(68)]
        + ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z"]
        + ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
        + au_columns
    )
    rows = []
    for frame in range(1, num_frames + 1):
        if skip_frames and frame in skip_frames:
            continue
        row = {column: 0.0 for column in columns}
        row["frame"] = frame
        row["face_id"] = 1
        row["X_0"] = float(frame)
        row["Y_0"] = float(frame)
        rows.append(row)
    return rows


def _write_openface_csv(path: Path, rows: list[dict]) -> None:
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row[column]) for column in columns) + "\n")


class TestTail1800:
    def test_tail_slices_last_1800(self) -> None:
        matrix = np.arange(2000 * 3, dtype=np.float32).reshape(2000, 3)
        clip = worker._tail_1800(matrix)
        assert clip.shape == (1800, 3)
        np.testing.assert_array_equal(clip, matrix[-1800:])

    def test_short_matrix_zero_padded_at_end(self) -> None:
        matrix = np.arange(1600 * 2, dtype=np.float32).reshape(1600, 2)
        clip = worker._tail_1800(matrix)
        assert clip.shape == (1800, 2)
        np.testing.assert_array_equal(clip[:1600], matrix)
        np.testing.assert_array_equal(clip[1600:], np.zeros((200, 2), dtype=np.float32))


class TestBuildClipFromCsv:
    def test_single_face_tail_slice(self, tmp_path: Path) -> None:
        pandas = pytest.importorskip("pandas")
        del pandas  # only need it installed for openface_to_mccl import
        csv_path = tmp_path / "window.csv"
        _write_openface_csv(csv_path, _openface_csv_rows(2000))
        clip = worker.build_clip_from_csv(csv_path, min_valid_frames=1500, work_dir=tmp_path)
        assert clip is not None
        kps, gaze, pose, aus = clip
        assert kps.shape == (1800, 68, 3)
        assert gaze.shape == (1800, 4, 3)
        assert pose.shape == (1800, 2, 3)
        assert aus.shape == (1800, 14)
        # Tail slice: clip starts at frame 201 (0-indexed row 200) -> X_0 = 201.
        assert kps[0, 0, 0] == 201.0
        assert kps[1799, 0, 0] == 2000.0

    def test_missing_frames_zero_filled(self, tmp_path: Path) -> None:
        pytest.importorskip("pandas")
        csv_path = tmp_path / "window.csv"
        _write_openface_csv(csv_path, _openface_csv_rows(1800, skip_frames={5}))
        clip = worker.build_clip_from_csv(csv_path, min_valid_frames=1500, work_dir=tmp_path)
        assert clip is not None
        kps = clip[0]
        # Frame 5 absent -> row index 4 must be zeros (frame 1..4 at index 0..3).
        np.testing.assert_array_equal(kps[4], np.zeros((68, 3), dtype=np.float32))
        assert kps[0, 0, 0] == 1.0

    def test_short_window_returns_none(self, tmp_path: Path) -> None:
        pytest.importorskip("pandas")
        csv_path = tmp_path / "window.csv"
        _write_openface_csv(csv_path, _openface_csv_rows(100))
        assert (
            worker.build_clip_from_csv(csv_path, min_valid_frames=1500, work_dir=tmp_path) is None
        )


class TestSnapshotBuilding:
    def _snapshot_kwargs(self, **overrides):
        base = {
            "assessment_id": "psy-abc",
            "subject_key": "elder-001",
            "window_started_at": datetime.now(UTC),
            "window_ended_at": datetime.now(UTC),
        }
        base.update(overrides)
        return base

    def test_processing_snapshot_valid(self) -> None:
        snapshot = worker.build_snapshot(
            status="processing", clip_count=3, **self._snapshot_kwargs()
        )
        assert snapshot.status == "processing"
        assert snapshot.clip_count == 3
        assert snapshot.estimated_phq8_score is None

    def test_completed_snapshot_valid(self) -> None:
        snapshot = worker.build_snapshot(
            status="completed",
            score=8.5,
            clip_count=7,
            completed_at=datetime.now(UTC),
            **self._snapshot_kwargs(),
        )
        assert snapshot.status == "completed"
        assert snapshot.estimated_phq8_score == 8.5
        assert snapshot.segment_scores == [8.5]

    def test_completed_without_score_rejected_by_validator(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            worker.build_snapshot(
                status="completed",
                clip_count=7,
                completed_at=datetime.now(UTC),
                **self._snapshot_kwargs(),
            )

    def test_completed_with_fewer_than_7_clips_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            worker.build_snapshot(
                status="completed",
                score=8.5,
                clip_count=3,
                completed_at=datetime.now(UTC),
                **self._snapshot_kwargs(),
            )

    def test_insufficient_data_snapshot_valid(self) -> None:
        snapshot = worker.build_snapshot(
            status="insufficient_data", clip_count=3, **self._snapshot_kwargs()
        )
        assert snapshot.status == "insufficient_data"
        assert snapshot.clip_count == 3
