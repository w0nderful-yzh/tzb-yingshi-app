from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _get_json(url: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            return None, "response is not a JSON object"
        return payload, None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def run(
    *,
    backend_url: str,
    radar_url: str,
    expected_room: str,
    sample_count: int,
    interval_seconds: float,
    timeout_seconds: float,
    shadow_path: Path,
) -> dict[str, Any]:
    before_size = shadow_path.stat().st_size if shadow_path.is_file() else 0
    radar_health, radar_health_error = _get_json(f"{radar_url}/health", timeout_seconds)
    camera_samples: list[dict[str, Any]] = []
    radar_samples: list[dict[str, Any]] = []
    fusion_samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index in range(sample_count):
        camera, camera_error = _get_json(
            f"{backend_url}/api/fall-live/status", timeout_seconds
        )
        radar, radar_error = _get_json(f"{backend_url}/api/radar/status", timeout_seconds)
        fusion, fusion_error = _get_json(
            f"{backend_url}/api/multimodal/latest", timeout_seconds
        )
        if camera is not None:
            camera_samples.append(camera)
        if radar is not None:
            radar_samples.append(radar)
        if fusion is not None:
            fusion_samples.append(fusion)
        for source, error in (
            ("camera", camera_error),
            ("radar", radar_error),
            ("fusion", fusion_error),
        ):
            if error:
                errors.append({"sample": str(index), "source": source, "error": error})
        if index + 1 < sample_count:
            time.sleep(interval_seconds)

    after_size = shadow_path.stat().st_size if shadow_path.is_file() else 0
    latest_camera = camera_samples[-1] if camera_samples else {}
    latest_radar = radar_samples[-1] if radar_samples else {}
    latest_fusion = fusion_samples[-1] if fusion_samples else {}
    latest_camera_evidence = latest_fusion.get("camera", {})
    latest_radar_evidence = latest_fusion.get("radar", {})
    latest_fusion_result = latest_fusion.get("fusion", {})
    latest_sensor_metrics = latest_radar.get("sensor_metrics") or {}

    checks = {
        "radar_service_reachable": radar_health is not None,
        "radar_connected": bool(radar_health and radar_health.get("radar_connected")),
        "radar_model_loaded": bool(radar_health and radar_health.get("model_loaded")),
        "radar_source_real": bool(
            radar_health and radar_health.get("source_mode") == "REAL"
        ),
        "backend_camera_reachable": bool(camera_samples),
        "camera_running": latest_camera.get("state") == "RUNNING",
        "camera_input_ready": bool(
            latest_camera.get("input_state") == "READY"
            and latest_camera.get("training_input_ready")
        ),
        "camera_score_available": _finite(latest_camera.get("risk_score")),
        "backend_radar_reachable": bool(radar_samples),
        "radar_online": latest_radar.get("online") is True,
        "radar_room_living_room": latest_radar.get("room") == expected_room,
        "radar_point_cloud_active": bool(
            _finite(latest_sensor_metrics.get("point_count"))
            and latest_sensor_metrics["point_count"] > 0
        ),
        "radar_frame_rate_active": bool(
            _finite(latest_sensor_metrics.get("frame_rate_hz"))
            and latest_sensor_metrics["frame_rate_hz"] > 0
        ),
        "radar_score_available": _finite(latest_radar_evidence.get("radar_score")),
        "fusion_reachable": bool(fusion_samples),
        "real_data_source": latest_fusion.get("data_source") == "REAL_CAMERA_RADAR",
        "camera_evidence_available": latest_camera_evidence.get("available") is True,
        "radar_evidence_available": latest_radar_evidence.get("available") is True,
        "fusion_score_available": _finite(latest_fusion_result.get("fusion_score")),
        "sync_within_tolerance": bool(
            _finite(latest_fusion_result.get("sync_delta_ms"))
            and latest_fusion_result["sync_delta_ms"]
            <= latest_fusion.get("timing", {}).get("tolerance_ms", 2000.0)
        ),
        "shadow_log_growing": after_size > before_size,
    }
    required = tuple(checks)
    return {
        "schema_version": "real_camera_radar_smoke_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "REAL_CAMERA_RADAR",
        "shadow_only": True,
        "expected_room": expected_room,
        "sample_count_requested": sample_count,
        "sample_count_received": {
            "camera": len(camera_samples),
            "radar": len(radar_samples),
            "fusion": len(fusion_samples),
        },
        "checks": checks,
        "passed": all(checks[name] for name in required),
        "errors": errors,
        "latest": {
            "camera_score": latest_camera_evidence.get("camera_score"),
            "camera_risk_state": latest_camera_evidence.get("camera_risk_state"),
            "radar_score": latest_radar_evidence.get("radar_score"),
            "radar_risk_state": latest_radar_evidence.get("radar_risk_state"),
            "radar_point_count": latest_sensor_metrics.get("point_count"),
            "radar_frame_rate_hz": latest_sensor_metrics.get("frame_rate_hz"),
            "fusion_score": latest_fusion_result.get("fusion_score"),
            "fusion_risk_state": latest_fusion_result.get("fusion_risk_state"),
            "sync_delta_ms": latest_fusion_result.get("sync_delta_ms"),
            "degraded_mode": latest_fusion_result.get("degraded_mode"),
        },
        "shadow_log": {
            "path": str(shadow_path.resolve()),
            "bytes_before": before_size,
            "bytes_after": after_size,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only real Camera/Radar/Fusion smoke test")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--radar-url", default="http://127.0.0.1:8010")
    parser.add_argument("--expected-room", default="living_room")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--shadow-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runtime" / "fusion_shadow.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runtime"
        / "real_multimodal_smoke_latest.json",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.interval < 0 or args.timeout <= 0:
        parser.error("samples/interval/timeout are invalid")
    report = run(
        backend_url=args.backend_url.rstrip("/"),
        radar_url=args.radar_url.rstrip("/"),
        expected_room=args.expected_room,
        sample_count=args.samples,
        interval_seconds=args.interval,
        timeout_seconds=args.timeout,
        shadow_path=args.shadow_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
