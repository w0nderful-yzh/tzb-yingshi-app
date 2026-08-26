"""Capture a bounded standard live stream clip with the model Python runtime.

The stream URL is read from stdin so signed EZVIZ URLs are not exposed in the
process command line or normal logs.  This script is executed by the existing
RTMPose/BioSTGCN Python environment, which already includes OpenCV with FFmpeg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--open-timeout-ms", type=int, default=15_000)
    parser.add_argument("--read-timeout-ms", type=int, default=5_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stream_url = sys.stdin.readline().strip()
    if not stream_url:
        raise ValueError("standard stream URL was not provided on stdin")
    if args.duration <= 0:
        raise ValueError("duration must be positive")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for live stream capture") from exc

    capture = cv2.VideoCapture()
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, args.open_timeout_ms)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, args.read_timeout_ms)
    if not capture.open(stream_url, cv2.CAP_FFMPEG):
        raise RuntimeError("unable to open the EZVIZ standard live stream")

    writer = None
    frames_written = 0
    started_at = time.monotonic()
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not 1.0 <= fps <= 120.0:
            fps = 25.0

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError("the EZVIZ stream opened but returned no video frame")
        height, width = first_frame.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError("the EZVIZ stream returned an invalid frame size")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("unable to create the temporary MP4 capture")

        target_frames = max(int(round(args.duration * fps)), 90)
        writer.write(first_frame)
        frames_written = 1
        while frames_written < target_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"EZVIZ stream ended after {frames_written} captured frames"
                )
            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames_written < 90 or not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("captured clip is too short for the 90-frame fall model window")

    report = {
        "frames": frames_written,
        "fps": fps,
        "width": width,
        "height": height,
        "requested_duration_seconds": args.duration,
        "capture_elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
