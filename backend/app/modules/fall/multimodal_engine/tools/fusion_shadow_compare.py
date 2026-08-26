from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


RISK_STATES = {"WATCH", "HIGH", "IMMINENT"}

STATE_ALIASES = {
    "camera_only": {
        "LOW": "NORMAL",
        "MEDIUM": "WATCH",
    },
    "radar_only": {
        "SUPPRESSED_RECOVERY": "NORMAL",
    },
}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    # PowerShell's round-trip format may emit seven fractional digits while
    # Python datetime accepts at most six.
    normalized = re.sub(r"(\.\d{6})\d+([+-]\d{2}:\d{2})$", r"\1\2", normalized)
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("time filters must include a timezone offset")
    return parsed


def _episodes(states: list[str]) -> int:
    count = 0
    active = False
    for state in states:
        now_active = state in RISK_STATES
        if now_active and not active:
            count += 1
        active = now_active
    return count


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "sample_count": len(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
    }


def _normalize_state(path_name: str, state: str) -> str:
    return STATE_ALIASES.get(path_name, {}).get(state, state)


def _path_metrics(
    path_name: str,
    states: list[str],
    *,
    expected_risk: str,
) -> dict[str, Any]:
    normalized = [_normalize_state(path_name, state) for state in states]
    raw_counts = Counter(states)
    counts = Counter(normalized)
    risk_count = sum(counts.get(state, 0) for state in RISK_STATES)
    total = len(normalized)
    result: dict[str, Any] = {
        "sample_count": total,
        "raw_state_counts": dict(raw_counts),
        "state_counts": dict(counts),
        "watch_high_imminent_ratio": risk_count / total if total else None,
        "unknown_ratio": counts.get("UNKNOWN", 0) / total if total else None,
        "risk_episode_count": _episodes(normalized),
    }
    if expected_risk == "NORMAL":
        result["false_alarm_proxy_episode_count"] = result["risk_episode_count"]
        result["false_alarm_proxy_note"] = (
            "Valid only because this interval was explicitly labelled NORMAL; "
            "WATCH is included as a conservative diagnostic excursion."
        )
    return result


def analyze(
    path: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    expected_risk: str,
    label: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        timestamp = _parse_time(row.get("logged_at") or row.get("timestamp"))
        if timestamp is None or (start and timestamp < start) or (end and timestamp > end):
            continue
        rows.append(row)

    paths: dict[str, list[str]] = {
        "camera_only": [],
        "radar_only": [],
        "fixed_fusion": [],
        "temporal_associated_fusion": [],
        "alignment_camera_led_radar_evidence": [],
    }
    target_associations: Counter[str] = Counter()
    temporal_relations: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    temporal_rows = 0
    sync_values: list[float] = []
    degraded_modes: Counter[str] = Counter()
    dynamic_levels: Counter[str] = Counter()
    dynamic_reasons: Counter[str] = Counter()
    dynamic_scores: list[float] = []
    short_term_scores: list[float] = []
    fall_event_statuses: Counter[str] = Counter()
    camera_quality: list[float] = []
    radar_quality: list[float] = []
    camera_latency_ms: list[float] = []
    radar_latency_ms: list[float] = []
    camera_age_ms: list[float] = []
    radar_age_ms: list[float] = []
    alignment_rows = 0
    alignment_states: Counter[str] = Counter()
    alignment_reason_codes: Counter[str] = Counter()
    alignment_sync_delta_ms: list[float] = []
    alignment_confidence: list[float] = []
    alignment_eligible_count = 0
    associated_rows = 0
    associated_evidence_states: Counter[str] = Counter()
    radar_motion_strengths: Counter[str] = Counter()
    associated_reason_codes: Counter[str] = Counter()
    associated_camera_score_mismatch_count = 0
    for row in rows:
        risk = row.get("risk_state") or {}
        paths["camera_only"].append(risk.get("camera") or row.get("camera_state") or "UNKNOWN")
        paths["radar_only"].append(risk.get("radar") or row.get("radar_state") or "UNKNOWN")
        paths["fixed_fusion"].append(
            risk.get("fusion") or row.get("stable_fusion_state") or "UNKNOWN"
        )
        temporal = row.get("temporal_associated_fusion")
        if isinstance(temporal, dict):
            temporal_rows += 1
            paths["temporal_associated_fusion"].append(
                temporal.get("fusion_state", "UNKNOWN")
            )
            target_associations[temporal.get("target_association", "UNKNOWN")] += 1
            temporal_relations[temporal.get("temporal_relation", "INSUFFICIENT_EVIDENCE")] += 1
            reason_codes.update(temporal.get("reason_codes") or [])
        else:
            paths["temporal_associated_fusion"].append("UNKNOWN")
        associated = row.get("associated_risk_augmentation")
        if isinstance(associated, dict):
            associated_rows += 1
            paths["alignment_camera_led_radar_evidence"].append(
                str(associated.get("associated_risk_state") or "UNKNOWN")
            )
            associated_evidence_states[
                str(associated.get("associated_evidence_state") or "UNKNOWN")
            ] += 1
            radar_motion_strengths[
                str(associated.get("radar_motion_evidence_strength") or "UNKNOWN")
            ] += 1
            associated_reason_codes.update(associated.get("reason_codes") or [])
            if associated.get("associated_short_term_fall_score") != associated.get(
                "base_camera_score"
            ):
                associated_camera_score_mismatch_count += 1
        else:
            paths["alignment_camera_led_radar_evidence"].append("UNKNOWN")
        sync = row.get("sync_delta_ms")
        if isinstance(sync, (int, float)):
            sync_values.append(float(sync))
        degraded_modes[str(row.get("degraded_mode") or "UNKNOWN")] += 1
        dynamic_score = row.get("dynamic_risk_score")
        if isinstance(dynamic_score, (int, float)):
            dynamic_scores.append(float(dynamic_score))
        dynamic_levels[str(row.get("dynamic_risk_level") or "UNKNOWN")] += 1
        for reason in row.get("dynamic_risk_reasons") or []:
            if isinstance(reason, dict):
                dynamic_reasons[str(reason.get("code") or "UNKNOWN")] += 1
            else:
                dynamic_reasons[str(reason)] += 1
        short_term_score = row.get("short_term_fall_score")
        if isinstance(short_term_score, (int, float)):
            short_term_scores.append(float(short_term_score))
        fall_event_statuses[str(row.get("fall_event_status") or "UNKNOWN")] += 1
        for key, target in (
            ("camera_quality", camera_quality),
            ("radar_quality", radar_quality),
            ("camera_processing_latency_ms", camera_latency_ms),
            ("radar_processing_latency_ms", radar_latency_ms),
            ("camera_evidence_age_ms", camera_age_ms),
            ("radar_evidence_age_ms", radar_age_ms),
        ):
            value = row.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))

        alignment = row.get("alignment")
        if isinstance(alignment, dict):
            alignment_rows += 1
            alignment_states[str(alignment.get("association_state") or "UNKNOWN")] += 1
            alignment_reason_codes.update(alignment.get("reason_codes") or [])
            alignment_sync = alignment.get("sync_delta_ms")
            if isinstance(alignment_sync, (int, float)):
                alignment_sync_delta_ms.append(float(alignment_sync))
            confidence = alignment.get("association_confidence")
            if isinstance(confidence, (int, float)):
                alignment_confidence.append(float(confidence))
            if alignment.get("eligible_for_temporal_association") is True:
                alignment_eligible_count += 1

    return {
        "schema_version": "fusion_shadow_system_audit_v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(path.resolve()),
        "label": label,
        "expected_risk": expected_risk,
        "time_filter": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
        "row_count": len(rows),
        "malformed_row_count": malformed,
        "temporal_extension_coverage_ratio": temporal_rows / len(rows) if rows else 0.0,
        "associated_evidence_coverage_ratio": associated_rows / len(rows) if rows else 0.0,
        "paths": {
            name: _path_metrics(name, states, expected_risk=expected_risk)
            for name, states in paths.items()
        },
        "target_association_counts": dict(target_associations),
        "temporal_relation_counts": dict(temporal_relations),
        "temporal_reason_code_counts": dict(reason_codes),
        "camera_led_associated_evidence": {
            "evidence_state_counts": dict(associated_evidence_states),
            "radar_motion_strength_counts": dict(radar_motion_strengths),
            "reason_code_counts": dict(associated_reason_codes),
            "camera_score_invariant_mismatch_count": (
                associated_camera_score_mismatch_count
            ),
            "uses_radar_tcn_score": False,
            "scope": (
                "BioSTGCN Camera-only is primary; matched TI tracking motion is "
                "non-learned shadow evidence only"
            ),
        },
        "sync_delta_ms": {
            **_distribution(sync_values),
            "available_ratio": len(sync_values) / len(rows) if rows else 0.0,
        },
        "degraded_mode_counts": dict(degraded_modes),
        "three_layer_risk": {
            "dynamic_risk": {
                "score": _distribution(dynamic_scores),
                "available_ratio": len(dynamic_scores) / len(rows) if rows else 0.0,
                "level_counts": dict(dynamic_levels),
                "reason_code_counts": dict(dynamic_reasons),
            },
            "short_term_warning": {
                "score": _distribution(short_term_scores),
                "available_ratio": len(short_term_scores) / len(rows) if rows else 0.0,
                "state_counts": dict(Counter(paths["fixed_fusion"])),
            },
            "fall_event": {
                "status_counts": dict(fall_event_statuses),
            },
        },
        "quality": {
            "camera": _distribution(camera_quality),
            "radar": _distribution(radar_quality),
        },
        "processing_latency_ms": {
            "camera": _distribution(camera_latency_ms),
            "radar": _distribution(radar_latency_ms),
        },
        "evidence_age_ms": {
            "camera": _distribution(camera_age_ms),
            "radar": _distribution(radar_age_ms),
        },
        "camera_radar_alignment": {
            "coverage_ratio": alignment_rows / len(rows) if rows else 0.0,
            "state_counts": dict(alignment_states),
            "eligible_ratio": (
                alignment_eligible_count / alignment_rows if alignment_rows else 0.0
            ),
            "sync_delta_ms": _distribution(alignment_sync_delta_ms),
            "association_confidence": _distribution(alignment_confidence),
            "reason_code_counts": dict(alignment_reason_codes),
            "scope": "shadow-only; does not alter Fixed Fusion or formal alerts",
        },
        "time_semantics": {
            "sync_delta_ms": "absolute difference between CameraEvidence and RadarEvidence source timestamps at Fusion evaluation time",
            "alignment_capture_sync": "nearest decoded Camera frame to Radar frame in the same-host monotonic clock domain; reported separately by the alignment tool",
        },
        "guardrail": (
            "Unlabelled historic rows support state/UNKNOWN/conflict auditing only. "
            "They must not be reported as measured false alarms."
        ),
        "experiment_groups": {
            "A": "BioSTGCN Camera-only",
            "B": "unchanged Fixed 0.6 Camera / 0.4 Radar TCN Fusion",
            "C": (
                "Alignment + BioSTGCN Camera result + TI tracking motion evidence "
                "state machine (no training)"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare four paths in fusion shadow JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--expected-risk", choices=("UNLABELLED", "NORMAL", "RISK"), default="UNLABELLED")
    parser.add_argument("--label", default="shadow_interval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.input,
        start=_parse_time(args.start),
        end=_parse_time(args.end),
        expected_risk=args.expected_risk,
        label=args.label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
