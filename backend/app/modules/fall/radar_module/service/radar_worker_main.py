"""Radar worker entrypoint: run calibrated-TCN inference and publish snapshots.

Runs as a separate subprocess so torch/TI dependencies never load into the App
Backend process. Writes the raw calibrated-TCN runtime state (the radar
module's own result, not the App-facing schema) to
runtime_state/<room>_latest.json; LocalRadarSource on :8000 reads and maps it.
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from radar_module.acquisition.ti_reader import JsonlReplayAdapter, TiRadarReader
from radar_module.contracts import Room
from radar_module.inference.calibrated_tcn_live_v1 import (
    CalibratedTcnLivePredictorV1,
)


def write_snapshot(state_root: Path, room: str, result_dict: dict[str, object]) -> None:
    """Atomically write the latest runtime snapshot for a room."""
    state_root.mkdir(parents=True, exist_ok=True)
    destination = state_root / f"{room}_latest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{room}-",
        suffix=".tmp",
        dir=state_root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(result_dict, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Radar calibrated-TCN worker that publishes runtime snapshots"
    )
    parser.add_argument("--room", required=True, help="bathroom / bedroom / living_room")
    parser.add_argument("--device-id", default="iwr6843isk-01")
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-method", default="real_gaussian")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-state-dir", type=Path, default=Path("runtime_state"))
    parser.add_argument("--snapshot-interval-seconds", type=float, default=1.0)
    args = parser.parse_args()

    predictor = CalibratedTcnLivePredictorV1(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        calibration_path=args.calibration,
        calibration_method=args.calibration_method,
        device=args.device,
    )

    if args.replay_file is None:
        raise SystemExit("--replay-file is required (Phase 1 supports replay mode)")

    reader = TiRadarReader(
        source_adapter=JsonlReplayAdapter(
            args.replay_file,
            speed=args.speed,
            loop=args.loop,
        ),
        device_id=args.device_id,
        room=Room(args.room),
        max_distance_m=8.0,
    )

    state_root = args.runtime_state_dir
    last_write = 0.0
    reader.start()
    try:
        while True:
            frame = reader.read()
            if frame is None:
                if not args.loop:
                    break
                continue
            try:
                result = predictor.consume(frame)
            except ValueError:
                # Replay loop wrap-around resets frame timestamps; reset and continue.
                predictor.reset()
                continue
            if result is None:
                continue
            now = time.monotonic()
            if now - last_write >= args.snapshot_interval_seconds:
                write_snapshot(state_root, args.room, result.to_dict())
                last_write = now
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
