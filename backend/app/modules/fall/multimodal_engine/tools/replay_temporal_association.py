from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.modules.fall.multimodal_engine.schemas.multimodal import MultimodalLatestResponse
from app.modules.fall.multimodal_engine.services.temporal_associated_fusion import TemporalAssociatedFusion


RISK_STATES = {"WATCH", "HIGH", "IMMINENT"}


def _episodes(states: list[str]) -> int:
    count = 0
    active = False
    for state in states:
        next_active = state in RISK_STATES
        if next_active and not active:
            count += 1
        active = next_active
    return count


def _transitions(states: list[str]) -> int:
    return sum(left != right for left, right in zip(states, states[1:]))


def replay(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    temporal = TemporalAssociatedFusion()
    states: list[str] = []
    associations: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    invalid_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(source.get("samples") or []):
        try:
            response = MultimodalLatestResponse.model_validate(sample["multimodal"])
            result = temporal.apply(response.camera, response.radar, response.fusion)
        except Exception as exc:
            invalid_samples.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        states.append(result.fusion_state)
        associations[result.target_association] += 1
        relations[result.temporal_relation] += 1
        reasons.update(result.reason_codes)
    counts = Counter(states)
    total = len(states)
    risk_count = sum(counts.get(state, 0) for state in RISK_STATES)
    four_path = dict(source.get("four_path_state_metrics") or {})
    recorded_temporal = four_path.get("temporal_associated_fusion")
    four_path["temporal_associated_fusion_replayed"] = {
        "counts": dict(counts),
        "unknown_ratio": counts.get("UNKNOWN", 0) / total if total else None,
        "watch_high_imminent_ratio": risk_count / total if total else None,
        "risk_episode_count": _episodes(states),
        "state_transitions": _transitions(states),
    }
    return {
        "schema_version": "temporal_associated_evidence_replay_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_capture": str(path.resolve()),
        "condition": source.get("condition"),
        "activity_label": source.get("activity_label"),
        "expected_risk": source.get("expected_risk"),
        "shadow_only": True,
        "affects_alerts": False,
        "model_or_threshold_changed": False,
        "sample_count": total,
        "invalid_samples": invalid_samples,
        "recorded_temporal_metrics": recorded_temporal,
        "four_path_state_metrics": four_path,
        "target_association_counts": dict(associations),
        "temporal_relation_counts": dict(relations),
        "reason_code_counts": dict(reasons),
        "guardrail": (
            "This is a deterministic Evidence replay of the shadow association logic. "
            "It does not retrain or retune Camera, Radar TCN, Fixed Fusion or thresholds."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay shadow Temporal/Associated Fusion")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
