from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


RISK_STATES = {"WATCH", "HIGH", "IMMINENT"}


def _time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    normalized = re.sub(r"(\.\d{6})\d+([+-]\d{2}:\d{2})$", r"\1\2", normalized)
    result = datetime.fromisoformat(normalized)
    if result.tzinfo is None:
        raise ValueError("timestamps must include timezone")
    return result


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def _normalize(path: str, state: str) -> str:
    if path == "camera_only":
        return {"LOW": "NORMAL", "MEDIUM": "WATCH"}.get(state, state)
    if path == "radar_only" and state == "SUPPRESSED_RECOVERY":
        return "NORMAL"
    return state


def _metrics(path: str, states: list[str]) -> dict[str, Any]:
    normalized = [_normalize(path, state) for state in states]
    counts = Counter(normalized)
    total = len(normalized)
    episodes = 0
    active = False
    for state in normalized:
        current = state in RISK_STATES
        if current and not active:
            episodes += 1
        active = current
    return {
        "sample_count": total,
        "state_counts": dict(counts),
        "watch_high_imminent_ratio": (
            sum(counts[state] for state in RISK_STATES) / total if total else None
        ),
        "unknown_ratio": counts["UNKNOWN"] / total if total else None,
        "risk_episode_count": episodes,
    }


def compare(
    fusion_log: Path,
    sync_pairs: Path,
    alignment_evidence: Path,
    *,
    max_join_ms: float,
) -> dict[str, Any]:
    pair_by_frame: dict[str, dict[str, Any]] = {}
    for pair in _rows(sync_pairs):
        frame_id = str(pair.get("camera_frame_id") or "")
        camera = pair.get("camera_evidence") or {}
        radar = pair.get("radar_evidence") or {}
        if frame_id and camera.get("source_timestamp") and radar.get("source_timestamp"):
            pair_by_frame[frame_id] = {
                "camera_time": _time(camera["source_timestamp"]),
                "radar_time": _time(radar["source_timestamp"]),
            }

    alignment_samples = []
    for alignment in _rows(alignment_evidence):
        pair = pair_by_frame.get(str(alignment.get("camera_frame_id") or ""))
        if pair is not None:
            alignment_samples.append({**pair, "alignment": alignment})

    paths: dict[str, list[str]] = {
        "camera_only": [],
        "radar_only": [],
        "fixed_fusion": [],
        "temporal_associated_recorded": [],
        "alignment_gated_temporal_counterfactual": [],
    }
    alignment_states: Counter[str] = Counter()
    joined = 0
    score_changed = 0
    for row in _rows(fusion_log):
        camera_value = row.get("camera_source_timestamp")
        radar_value = row.get("radar_source_timestamp")
        if not camera_value or not radar_value or not alignment_samples:
            continue
        camera_time = _time(str(camera_value))
        radar_time = _time(str(radar_value))
        sample = min(
            alignment_samples,
            key=lambda item: abs((item["camera_time"] - camera_time).total_seconds())
            + abs((item["radar_time"] - radar_time).total_seconds()),
        )
        camera_delta = abs((sample["camera_time"] - camera_time).total_seconds()) * 1000
        radar_delta = abs((sample["radar_time"] - radar_time).total_seconds()) * 1000
        if camera_delta > max_join_ms or radar_delta > max_join_ms:
            continue
        joined += 1
        risk = row.get("risk_state") or {}
        camera_state = str(risk.get("camera") or row.get("camera_state") or "UNKNOWN")
        radar_state = str(risk.get("radar") or row.get("radar_state") or "UNKNOWN")
        fixed_state = str(
            risk.get("fusion") or row.get("stable_fusion_state") or "UNKNOWN"
        )
        temporal = row.get("temporal_associated_fusion") or {}
        temporal_state = str(temporal.get("fusion_state") or "UNKNOWN")
        alignment_state = str(sample["alignment"].get("association_state") or "CALIBRATION_INVALID")
        alignment_states[alignment_state] += 1

        gated_state = temporal_state
        if alignment_state in {"TRACK_CONFLICT", "MULTIPLE_CANDIDATES"}:
            camera_risk = _normalize("camera_only", camera_state) in RISK_STATES
            radar_risk = _normalize("radar_only", radar_state) in RISK_STATES
            gated_state = "WATCH" if camera_risk or radar_risk else "UNKNOWN"
        elif alignment_state != "MATCHED" and temporal_state in {"HIGH", "IMMINENT"}:
            gated_state = "WATCH"

        paths["camera_only"].append(camera_state)
        paths["radar_only"].append(radar_state)
        paths["fixed_fusion"].append(fixed_state)
        paths["temporal_associated_recorded"].append(temporal_state)
        paths["alignment_gated_temporal_counterfactual"].append(gated_state)
        # Alignment is a state gate only; score equality is an explicit invariant.
        if row.get("raw_fusion_score") != temporal.get("fusion_score"):
            score_changed += 1

    return {
        "schema_version": "alignment_gated_fusion_comparison_v1",
        "source": {
            "fusion_log": str(fusion_log.resolve()),
            "sync_pairs": str(sync_pairs.resolve()),
            "alignment_evidence": str(alignment_evidence.resolve()),
        },
        "join": {
            "max_join_ms": max_join_ms,
            "alignment_sample_count": len(alignment_samples),
            "joined_fusion_row_count": joined,
        },
        "alignment_state_counts_on_joined_rows": dict(alignment_states),
        "paths": {name: _metrics(name, states) for name, states in paths.items()},
        "fixed_score_vs_temporal_readout_mismatch_count": score_changed,
        "interpretation": {
            "comparison_type": "UNLABELLED_SAME_SESSION_COUNTERFACTUAL_STATE_GATE",
            "can_measure_false_alarm_or_recall": False,
            "alignment_changes_fixed_fusion": False,
            "guardrail": "This comparison measures availability/state behavior only; it cannot establish accuracy improvement without event labels.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare alignment-gated shadow state effects")
    parser.add_argument("--fusion-log", type=Path, required=True)
    parser.add_argument("--sync-pairs", type=Path, required=True)
    parser.add_argument("--alignment-evidence", type=Path, required=True)
    parser.add_argument("--max-join-ms", type=float, default=1000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.fusion_log,
        args.sync_pairs,
        args.alignment_evidence,
        max_join_ms=args.max_join_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
