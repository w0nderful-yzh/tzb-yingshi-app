"""Export one verified DGUHA B0 success event to the standard replay contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from radar_module.dataset.radhar_converter import parse_radhar_text


SOURCE_FILE = "Test/5_falling_forward/radar/M_012_A5_005.txt"
OUTPUT_FILE = "dguha_b0_success_M_012_A5_005.jsonl"
CHECKPOINT_SHA256 = "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_competition_replay(root: Path) -> dict[str, object]:
    source_root = root / "data/external/dguha/raw"
    event_path = root / "data/processed/dguha_prefall_0p5_1p0_dense_v3.events.json"
    checkpoint_path = root / (
        "checkpoints/experiments_v5/tcn_hard_negative/"
        "tcn_0p5_1p0_specificity_operating_point_v1.pt"
    )
    output = root / "data/replay" / OUTPUT_FILE
    manifest_path = output.with_suffix(".manifest.json")
    if _sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen B0 checkpoint SHA256 changed")

    events = json.loads(event_path.read_text(encoding="utf-8"))
    event = next(item for item in events if item["source_file"] == SOURCE_FILE)
    frames = parse_radhar_text(
        source_root / SOURCE_FILE,
        device_id="dguha-offline-replay-b0",
    )
    descent_onset = datetime.fromisoformat(str(event["descent_onset"]))
    cutoff = descent_onset.timestamp() - 0.1
    selected = [frame for frame in frames if frame.timestamp.timestamp() <= cutoff]
    if not selected:
        raise ValueError("no pre-onset DGUHA frames were selected")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in selected:
            payload = {
                "timestamp": frame.timestamp.isoformat(),
                "device_id": "dguha-offline-replay-b0",
                "room": "bathroom",
                "points": [
                    {
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                        "velocity": point.velocity,
                    }
                    for point in frame.points
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    manifest: dict[str, object] = {
        "schema_version": "dguha_competition_replay_v1",
        "purpose": "B0 competition offline replay; not IWR6843 real-sensor evidence",
        "source_dataset": "DGUHA",
        "source_file": SOURCE_FILE,
        "event_label": "forward_fall",
        "positive_anchor": "descent_onset",
        "last_frame_margin_before_onset_seconds": 0.1,
        "frame_count": len(selected),
        "duration_seconds": (
            selected[-1].timestamp - selected[0].timestamp
        ).total_seconds(),
        "point_count": sum(len(frame.points) for frame in selected),
        "output_file": output.name,
        "output_sha256": _sha256(output),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "threshold": 0.35,
        "confirmation_windows": 3,
        "display_mode": "DGUHA Offline Replay",
        "claim_limit": (
            "Demonstrates frozen B0 on its public-data domain; it must not be "
            "presented as IWR6843 real-sensor performance."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()
    print(
        json.dumps(
            export_competition_replay(args.root.resolve()),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
