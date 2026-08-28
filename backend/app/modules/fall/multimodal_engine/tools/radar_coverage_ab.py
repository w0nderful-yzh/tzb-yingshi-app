from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CONDITIONS = ("WOOD_BLOCKED", "WOOD_REMOVED_OR_RADAR_REPOSITIONED")


def _get_json(url: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return (payload, None) if isinstance(payload, dict) else (None, "non-object JSON")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "p10": _quantile(values, 0.10),
        "median": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def _longest_state_run(samples: list[dict[str, Any]], state: str, interval: float) -> float:
    longest = current = 0
    for sample in samples:
        radar = sample.get("multimodal", {}).get("radar", {})
        observed = radar.get("radar_risk_state", "UNKNOWN")
        if observed == state:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * interval


def _state_transitions(states: list[str]) -> int:
    return sum(left != right for left, right in zip(states, states[1:], strict=False))


def _risk_episodes(states: list[str]) -> int:
    count = 0
    active = False
    for state in states:
        next_active = state in {"WATCH", "HIGH", "IMMINENT"}
        if next_active and not active:
            count += 1
        active = next_active
    return count


def _prediction(radar: dict[str, Any]) -> dict[str, Any]:
    return (
        radar.get("tcn_baseline")
        or radar.get("tcn_prediction")
        or radar.get("calibrated_tcn_prediction")
        or {}
    )


def capture(
    *,
    backend_url: str,
    condition: str,
    activity_label: str,
    expected_risk: str,
    duration_seconds: float,
    interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    sample_count = max(1, math.ceil(duration_seconds / interval_seconds))
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index in range(sample_count):
        radar, radar_error = _get_json(f"{backend_url}/api/radar/status", timeout_seconds)
        multimodal, fusion_error = _get_json(
            f"{backend_url}/api/multimodal/latest", timeout_seconds
        )
        samples.append(
            {
                "observed_at": datetime.now(UTC).isoformat(),
                "radar": radar or {},
                "multimodal": multimodal or {},
            }
        )
        for source, error in (("radar", radar_error), ("multimodal", fusion_error)):
            if error:
                errors.append({"sample": index, "source": source, "error": error})
        if index + 1 < sample_count:
            time.sleep(interval_seconds)

    radar_available: list[bool] = []
    point_counts: list[float] = []
    missing_ratios: list[float] = []
    scores: list[float] = []
    states: dict[str, list[str]] = {
        "camera_only": [],
        "radar_only": [],
        "camera_led_multimodal": [],
    }
    unique_radar_timestamps: set[str] = set()
    for sample in samples:
        radar_status = sample["radar"]
        multimodal = sample["multimodal"]
        radar_evidence = multimodal.get("radar") or {}
        camera_led = multimodal.get("camera_led_evidence_fusion_v2") or {}
        metrics = radar_status.get("sensor_metrics") or {}
        prediction = _prediction(radar_status)
        available = radar_evidence.get("available") is True
        radar_available.append(available)
        point = _number(metrics.get("point_count"))
        if point is None:
            point = _number((radar_evidence.get("radar_feature") or {}).get("point_count"))
        if point is not None:
            point_counts.append(point)
        missing = _number(prediction.get("missing_frame_ratio"))
        if missing is not None:
            missing_ratios.append(missing)
        score = _number(radar_evidence.get("radar_score"))
        if score is not None:
            scores.append(score)
        timestamp = radar_evidence.get("source_timestamp")
        if isinstance(timestamp, str):
            unique_radar_timestamps.add(timestamp)
        camera_evidence = multimodal.get("camera") or {}
        states["camera_only"].append(camera_evidence.get("camera_risk_state", "UNKNOWN"))
        states["radar_only"].append(radar_evidence.get("radar_risk_state", "UNKNOWN"))
        states["camera_led_multimodal"].append(camera_led.get("camera_led_state", "UNKNOWN"))

    received = len(samples)
    availability_ratio = sum(radar_available) / received if received else 0.0
    state_metrics: dict[str, Any] = {}
    for name, observed in states.items():
        counts = Counter(observed)
        risk_count = sum(counts.get(item, 0) for item in ("WATCH", "HIGH", "IMMINENT"))
        state_metrics[name] = {
            "counts": dict(counts),
            "unknown_ratio": counts.get("UNKNOWN", 0) / received if received else 1.0,
            "watch_high_imminent_ratio": risk_count / received if received else 0.0,
            "state_transitions": _state_transitions(observed),
            "risk_episode_count": _risk_episodes(observed),
        }
    normal_false_alarm_proxy = (
        {name: metrics["watch_high_imminent_ratio"] for name, metrics in state_metrics.items()}
        if expected_risk == "NORMAL"
        else None
    )
    return {
        "schema_version": "radar_coverage_ab_capture_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "data_source": "REAL_CAMERA_RADAR",
        "shadow_only": True,
        "condition": condition,
        "activity_label": activity_label,
        "expected_risk": expected_risk,
        "protocol_note": (
            "Keep camera, models, thresholds, action sequence and distance "
            "unchanged between A/B conditions."
        ),
        "duration_seconds_requested": duration_seconds,
        "interval_seconds": interval_seconds,
        "sample_count": received,
        "errors": errors,
        "coverage": {
            "radar_availability_ratio": availability_ratio,
            "radar_unavailable_ratio": 1.0 - availability_ratio,
            "unique_radar_evidence_windows": len(unique_radar_timestamps),
            "evidence_update_hz": len(unique_radar_timestamps)
            / max(duration_seconds, interval_seconds),
            "point_count": _summary(point_counts),
            "point_count_zero_ratio": (
                sum(value <= 0 for value in point_counts) / len(point_counts)
                if point_counts
                else None
            ),
            "missing_frame_ratio": _summary(missing_ratios),
            "radar_unknown_longest_seconds": _longest_state_run(
                samples, "UNKNOWN", interval_seconds
            ),
            "radar_score": _summary(scores),
            "radar_score_available_ratio": len(scores) / received if received else 0.0,
            "radar_state_transitions": _state_transitions(states["radar_only"]),
        },
        "room_path_state_metrics": state_metrics,
        "normal_activity_false_alarm_proxy": normal_false_alarm_proxy,
        "samples": samples,
    }


def compare(blocked: dict[str, Any], clear: dict[str, Any]) -> dict[str, Any]:
    def value(report: dict[str, Any], *path: str) -> float | None:
        current: Any = report
        for key in path:
            current = current.get(key, {}) if isinstance(current, dict) else None
        return _number(current)

    metrics = {
        "radar_availability_ratio": ("coverage", "radar_availability_ratio"),
        "point_count_median": ("coverage", "point_count", "median"),
        "point_count_p95": ("coverage", "point_count", "p95"),
        "missing_frame_ratio_mean": ("coverage", "missing_frame_ratio", "mean"),
        "radar_unknown_longest_seconds": ("coverage", "radar_unknown_longest_seconds"),
        "radar_score_std": ("coverage", "radar_score", "std"),
        "radar_state_transitions": ("coverage", "radar_state_transitions"),
    }
    comparison: dict[str, Any] = {}
    for name, path in metrics.items():
        left = value(blocked, *path)
        right = value(clear, *path)
        comparison[name] = {
            "wood_blocked": left,
            "wood_removed_or_radar_repositioned": right,
            "delta_clear_minus_blocked": (
                right - left if left is not None and right is not None else None
            ),
        }
    return {
        "schema_version": "radar_coverage_ab_comparison_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "REAL_CAMERA_RADAR",
        "shadow_only": True,
        "valid_protocol_match": (
            blocked.get("activity_label") == clear.get("activity_label")
            and blocked.get("expected_risk") == clear.get("expected_risk")
        ),
        "comparison": comparison,
        "room_path_state_metrics": {
            "wood_blocked": blocked.get("room_path_state_metrics", {}),
            "wood_removed_or_radar_repositioned": clear.get("room_path_state_metrics", {}),
        },
        "interpretation_guardrail": (
            "Coverage differences diagnose placement/occlusion only; they do not validate "
            "a model improvement or radar lead time."
        ),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _console_summary(payload: dict[str, Any], output: Path) -> dict[str, Any]:
    if payload.get("schema_version") == "radar_coverage_ab_capture_v1":
        return {
            "schema_version": payload["schema_version"],
            "condition": payload["condition"],
            "sample_count": payload["sample_count"],
            "errors": payload["errors"],
            "coverage": payload["coverage"],
            "room_path_state_metrics": payload["room_path_state_metrics"],
            "output": str(output.resolve()),
        }
    return {**payload, "output": str(output.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only living-room Radar coverage A/B")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    capture_parser.add_argument("--condition", choices=CONDITIONS, required=True)
    capture_parser.add_argument("--activity-label", required=True)
    capture_parser.add_argument("--expected-risk", choices=("NORMAL", "RISK"), default="NORMAL")
    capture_parser.add_argument("--duration", type=float, default=60.0)
    capture_parser.add_argument("--interval", type=float, default=0.5)
    capture_parser.add_argument("--timeout", type=float, default=2.0)
    capture_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--blocked", type=Path, required=True)
    compare_parser.add_argument("--clear", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        if args.duration <= 0 or args.interval <= 0 or args.timeout <= 0:
            parser.error("duration, interval and timeout must be positive")
        payload = capture(
            backend_url=args.backend_url.rstrip("/"),
            condition=args.condition,
            activity_label=args.activity_label,
            expected_risk=args.expected_risk,
            duration_seconds=args.duration,
            interval_seconds=args.interval,
            timeout_seconds=args.timeout,
        )
    else:
        blocked = json.loads(args.blocked.read_text(encoding="utf-8"))
        clear = json.loads(args.clear.read_text(encoding="utf-8"))
        payload = compare(blocked, clear)
    _write(args.output, payload)
    print(json.dumps(_console_summary(payload, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
