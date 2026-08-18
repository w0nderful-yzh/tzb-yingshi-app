"""Continuously record the live radar API without restarting the sensor chain."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record(url: str, output: Path, interval_seconds: float, duration_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sequence = 0
    with output.open("a", encoding="utf-8", buffering=1) as stream:
        while duration_seconds <= 0 or time.monotonic() - started < duration_seconds:
            sampled_at = _utc_now()
            row: dict[str, object] = {
                "schema_version": "radar_live_api_sample_v1",
                "sequence": sequence,
                "sampled_at": sampled_at,
            }
            try:
                request = Request(url, headers={"Accept": "application/json"})
                with urlopen(request, timeout=2.0) as response:
                    row["http_status"] = response.status
                    row["payload"] = json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # Keep recording API outages in the same timeline.
                row["error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            sequence += 1
            elapsed = time.monotonic() - started
            target = started + sequence * interval_seconds
            time.sleep(max(0.0, target - time.monotonic()))
            if sequence % 100 == 0:
                print(f"samples={sequence} elapsed={elapsed:.1f}s output={output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/radar/status")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.1)
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    record(args.url, args.output, args.interval_seconds, args.duration_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
