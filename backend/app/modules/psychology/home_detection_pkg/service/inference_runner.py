"""Run the existing offline pipeline and publish one machine-readable snapshot.

This module is intentionally not imported by the HTTP API. Reading the latest
snapshot can therefore never start video inference.
"""

import argparse
import csv
import re
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from service.result_store import LatestAssessmentStore
from service.schemas import PsychologyAssessmentSnapshot

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HOME_DETECT = PACKAGE_ROOT / "scripts" / "home_detect.py"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "home_out"
DEFAULT_STORE_ROOT = DEFAULT_OUTPUT_ROOT / "latest"

_SEGMENT_SCORE = re.compile(r"PHQ-8\s*=\s*(-?\d+(?:\.\d+)?)")
_AVERAGE_SCORE = re.compile(r"平均抑郁分数:\s*(-?\d+(?:\.\d+)?)")


def run_inference(
    *,
    video: Path,
    subject_key: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    store_root: Path = DEFAULT_STORE_ROOT,
    window_started_at: datetime | None = None,
    window_ended_at: datetime | None = None,
    mccl_device: str = "cpu",
) -> PsychologyAssessmentSnapshot:
    if not video.is_file():
        raise FileNotFoundError(video)

    command = [
        sys.executable,
        str(HOME_DETECT),
        str(video),
        "--outdir",
        str(output_root),
        "--mccl-device",
        mccl_device,
    ]
    completed = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    clip_count = len(list((output_root / video.stem / "clips").glob(f"{video.stem}-*_kps.npy")))
    segment_scores = [float(value) for value in _SEGMENT_SCORE.findall(combined_output)]
    average_match = _AVERAGE_SCORE.search(combined_output)
    average_score = (
        float(average_match.group(1))
        if average_match
        else statistics.fmean(segment_scores) if segment_scores else None
    )
    openface_duration = _openface_duration_seconds(output_root / video.stem / "openface")
    started_at, ended_at = _resolve_window(
        video=video,
        output_root=output_root,
        explicit_start=window_started_at,
        explicit_end=window_ended_at,
    )
    finished_at = datetime.now(timezone.utc)

    if openface_duration > 0.0 and openface_duration < 60.0 and clip_count < 7:
        status = "insufficient_data"
        average_score = None
        segment_scores = []
    elif completed.returncode != 0:
        status = "failed"
        average_score = None
        segment_scores = []
    elif average_score is not None and segment_scores and clip_count >= 7:
        status = "completed"
    elif clip_count < 7 or "不足7个" in combined_output or "不足60秒" in combined_output:
        status = "insufficient_data"
        average_score = None
        segment_scores = []
    else:
        status = "failed"
        average_score = None
        segment_scores = []

    snapshot = PsychologyAssessmentSnapshot(
        assessment_id=f"psy-{uuid.uuid4().hex}",
        subject_key=subject_key,
        status=status,
        window_started_at=started_at,
        window_ended_at=ended_at,
        estimated_phq8_score=average_score,
        segment_scores=segment_scores,
        clip_count=clip_count,
        completed_at=finished_at,
    )
    LatestAssessmentStore(store_root).write(snapshot)
    if completed.returncode != 0:
        print(combined_output, file=sys.stderr)
    return snapshot


def _resolve_window(
    *,
    video: Path,
    output_root: Path,
    explicit_start: datetime | None,
    explicit_end: datetime | None,
) -> tuple[datetime, datetime]:
    if explicit_start is not None and explicit_end is not None:
        return explicit_start, explicit_end
    duration = _openface_duration_seconds(output_root / video.stem / "openface")
    ended_at = explicit_end or datetime.fromtimestamp(video.stat().st_mtime, timezone.utc)
    started_at = explicit_start or ended_at - timedelta(seconds=duration)
    return started_at, ended_at


def _openface_duration_seconds(openface_dir: Path) -> float:
    csv_path = next(iter(sorted(openface_dir.glob("*.csv"))), None)
    if csv_path is None:
        return 0.0
    maximum = 0.0
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            raw = row.get(" timestamp") or row.get("timestamp")
            try:
                maximum = max(maximum, float(raw or 0.0))
            except ValueError:
                continue
    return maximum


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("assessment window timestamps must include a timezone")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run psychology inference and publish latest JSON")
    parser.add_argument("video", type=Path)
    parser.add_argument("subject_key")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--window-started-at")
    parser.add_argument("--window-ended-at")
    parser.add_argument("--mccl-device", default="cpu")
    args = parser.parse_args()
    snapshot = run_inference(
        video=args.video,
        subject_key=args.subject_key,
        output_root=args.output_root,
        store_root=args.store_root,
        window_started_at=_parse_datetime(args.window_started_at),
        window_ended_at=_parse_datetime(args.window_ended_at),
        mccl_device=args.mccl_device,
    )
    print(snapshot.model_dump_json(indent=2))
    if snapshot.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
