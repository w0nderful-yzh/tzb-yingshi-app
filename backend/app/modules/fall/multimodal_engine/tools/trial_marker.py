from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a UTC marker to a live shadow trial")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    marker = {
        "trial_id": args.trial_id,
        "event": args.event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(marker, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(marker, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
