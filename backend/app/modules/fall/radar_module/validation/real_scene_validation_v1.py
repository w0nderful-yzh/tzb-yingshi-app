from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

from radar_module.acquisition.ti_reader import (
    JsonlReplayAdapter,
    RadarSourceAdapter,
    TiOfficialOutputAdapter,
    TiRadarReader,
)
from radar_module.contracts import RadarFrame, Room, SourceMode
from radar_module.inference.tcn_live_v1 import RadarTcnLivePredictorV1
from radar_module.validation.iwr6843_stability_v1 import (
    FrameStabilitySampleV1,
    Iwr6843StabilityReportV1,
    analyze_iwr6843_stability,
)


ACTION_CATEGORIES = ("normal", "high_risk", "controlled_fall")
DEFAULT_CHECKPOINT_SHA256 = (
    "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef"
)
DEFAULT_EXPECTED_FRAME_RATE_HZ = 1000.0 / 55.0


@dataclass(frozen=True, slots=True)
class RealSceneExperimentReportV1:
    schema_version: str
    session_id: str
    action_category: str
    action_name: str
    source_mode: str
    room: str
    frame_count: int
    prediction_count: int
    valid_prediction_count: int
    unknown_prediction_count: int
    post_warmup_unknown_count: int
    unknown_reason_counts: dict[str, int]
    maximum_risk_score: float | None
    maximum_score_timestamp: str | None
    imminent_triggered: bool
    first_imminent_timestamp: str | None
    event_triggered: bool
    action_onset_seconds: float | None
    impact_seconds: float | None
    lead_time_seconds: float | None
    lead_time_reference: str | None
    lead_time_status: str
    checkpoint_sha256: str
    model_version: str
    threshold: float
    confirmation_windows: int
    shadow_only: bool
    formal_alerts_enabled: bool
    stability: dict[str, object]
    unknown_diagnosis: dict[str, object]
    score_curve_csv: str
    score_curve_svg: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DiagnosticRecordingAdapterV1:
    """Observe decoded metadata, then pass the same mapping through unchanged."""

    def __init__(self, delegate: RadarSourceAdapter) -> None:
        self.delegate = delegate
        self.samples: list[FrameStabilitySampleV1] = []
        self.last_decoded: Mapping[str, Any] | None = None

    @property
    def source_mode(self) -> SourceMode:
        return self.delegate.source_mode

    def start(self) -> None:
        self.samples.clear()
        self.last_decoded = None
        self.delegate.start()

    def read_decoded(self) -> Mapping[str, Any] | None:
        payload = self.delegate.read_decoded()
        self.last_decoded = payload
        if payload is not None:
            points = payload.get("points", ())
            point_count = len(points) if isinstance(points, Sequence) else 0
            self.samples.append(
                FrameStabilitySampleV1(
                    timestamp=_parse_timestamp(payload.get("timestamp")),
                    point_count=point_count,
                    ti_frame_number=_optional_int(payload.get("ti_frame_number")),
                    ti_parser_error=_optional_int(payload.get("ti_parser_error")),
                )
            )
        return payload

    def stop(self) -> None:
        self.delegate.stop()


def run_validation_session(
    source_adapter: RadarSourceAdapter,
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    output_directory: str | Path,
    session_id: str,
    action_category: str,
    action_name: str,
    room: Room | str = Room.BATHROOM,
    device_id: str = "iwr6843isk-01",
    duration_seconds: float | None = None,
    expected_frame_rate_hz: float = DEFAULT_EXPECTED_FRAME_RATE_HZ,
    confirmation_windows: int = 3,
    action_onset_seconds: float | None = None,
    impact_seconds: float | None = None,
    torch_device: str = "cpu",
) -> RealSceneExperimentReportV1:
    _validate_session_metadata(
        session_id=session_id,
        action_category=action_category,
        action_name=action_name,
        duration_seconds=duration_seconds,
        action_onset_seconds=action_onset_seconds,
        impact_seconds=impact_seconds,
    )
    output_dir = Path(output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = RadarTcnLivePredictorV1(
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        confirmation_windows=confirmation_windows,
        device=torch_device,
    )
    diagnostic_adapter = DiagnosticRecordingAdapterV1(source_adapter)
    reader = TiRadarReader(
        diagnostic_adapter,
        device_id=device_id,
        room=Room(room),
    )
    records: list[dict[str, object]] = []
    try:
        reader.start()
        started_monotonic = time.monotonic()
        while True:
            if (
                duration_seconds is not None
                and time.monotonic() - started_monotonic >= duration_seconds
            ):
                break
            frame = reader.read()
            if frame is None:
                delegate = diagnostic_adapter.delegate
                if isinstance(delegate, JsonlReplayAdapter) and delegate.finished:
                    break
                continue
            prediction = predictor.consume(frame)
            decoded = diagnostic_adapter.last_decoded or {}
            records.append(
                _session_record(
                    frame,
                    prediction.to_dict() if prediction is not None else None,
                    decoded,
                )
            )
    except BaseException as exc:
        _write_json(
            output_dir / "capture_error.json",
            {
                "schema_version": "radar_real_scene_capture_error_v1",
                "session_id": session_id,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "captured_frame_count": len(records),
                "checkpoint_sha256": predictor.checkpoint_sha256,
                "model_version": predictor.model_version,
                "threshold": predictor.threshold,
                "shadow_only": True,
                "formal_alerts_enabled": False,
            },
        )
        if records:
            _write_jsonl(output_dir / "partial_session.jsonl", records)
        raise
    finally:
        reader.stop()

    session_path = output_dir / "session.jsonl"
    _write_jsonl(session_path, records)
    manifest = {
        "schema_version": "radar_real_scene_session_v1",
        "session_id": session_id,
        "action_category": action_category,
        "action_name": action_name,
        "room": Room(room).value,
        "device_id": device_id,
        "source_mode": source_adapter.source_mode.value,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "session_file": session_path.name,
        "expected_frame_rate_hz": expected_frame_rate_hz,
        "action_onset_seconds": action_onset_seconds,
        "impact_seconds": impact_seconds,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "confirmation_windows": confirmation_windows,
        "shadow_only": True,
        "formal_alerts_enabled": False,
    }
    _write_json(output_dir / "session_manifest.json", manifest)
    return generate_experiment_report(
        output_dir,
        records=records,
        manifest=manifest,
        stability_samples=diagnostic_adapter.samples,
    )


def regenerate_experiment_report(
    session_directory: str | Path,
    *,
    action_onset_seconds: float | None = None,
    impact_seconds: float | None = None,
) -> RealSceneExperimentReportV1:
    output_dir = Path(session_directory).resolve()
    manifest_path = output_dir / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if action_onset_seconds is not None:
        manifest["action_onset_seconds"] = action_onset_seconds
    if impact_seconds is not None:
        manifest["impact_seconds"] = impact_seconds
    _validate_optional_seconds(
        manifest.get("action_onset_seconds"), "action_onset_seconds"
    )
    _validate_optional_seconds(manifest.get("impact_seconds"), "impact_seconds")
    _write_json(manifest_path, manifest)
    records = _read_jsonl(output_dir / str(manifest["session_file"]))
    samples = [_sample_from_record(record) for record in records]
    return generate_experiment_report(
        output_dir,
        records=records,
        manifest=manifest,
        stability_samples=samples,
    )


def generate_experiment_report(
    output_directory: str | Path,
    *,
    records: Sequence[dict[str, object]],
    manifest: Mapping[str, object],
    stability_samples: Sequence[FrameStabilitySampleV1],
) -> RealSceneExperimentReportV1:
    output_dir = Path(output_directory).resolve()
    stability = analyze_iwr6843_stability(
        stability_samples,
        expected_frame_rate_hz=float(manifest["expected_frame_rate_hz"]),
    )
    predictions = [
        prediction
        for record in records
        if isinstance((prediction := record.get("tcn_prediction")), dict)
    ]
    valid_predictions = [
        prediction for prediction in predictions if bool(prediction["score_valid"])
    ]
    valid_scores = [float(prediction["pre_fall_score"]) for prediction in valid_predictions]
    maximum_prediction = (
        max(valid_predictions, key=lambda item: float(item["pre_fall_score"]))
        if valid_predictions
        else None
    )
    imminent_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("risk_state") == "IMMINENT"
    ]
    first_imminent = imminent_predictions[0] if imminent_predictions else None
    first_valid_index = next(
        (
            index
            for index, prediction in enumerate(predictions)
            if bool(prediction["score_valid"])
        ),
        len(predictions),
    )
    post_warmup_unknown = [
        prediction
        for prediction in predictions[first_valid_index + 1 :]
        if not bool(prediction["score_valid"])
    ]
    unknown_reasons = Counter(
        str(prediction.get("unknown_reason") or "UNSPECIFIED")
        for prediction in predictions
        if not bool(prediction["score_valid"])
    )
    first_timestamp = (
        _parse_timestamp(records[0]["timestamp"]) if records else None
    )
    lead_time, lead_reference, lead_status = _lead_time(
        first_timestamp=first_timestamp,
        first_imminent=first_imminent,
        impact_seconds=_optional_float(manifest.get("impact_seconds")),
    )
    score_csv = output_dir / "score_curve.csv"
    score_svg = output_dir / "score_curve.svg"
    _write_score_csv(score_csv, records, first_timestamp)
    _write_score_svg(
        score_svg,
        records,
        first_timestamp=first_timestamp,
        threshold=float(manifest["threshold"]),
        action_name=str(manifest["action_name"]),
        action_onset_seconds=_optional_float(manifest.get("action_onset_seconds")),
        impact_seconds=_optional_float(manifest.get("impact_seconds")),
    )
    diagnosis = _diagnose_unknown(
        stability,
        post_warmup_unknown_predictions=post_warmup_unknown,
        unknown_reasons=dict(unknown_reasons),
    )
    report = RealSceneExperimentReportV1(
        schema_version="radar_real_scene_experiment_report_v1",
        session_id=str(manifest["session_id"]),
        action_category=str(manifest["action_category"]),
        action_name=str(manifest["action_name"]),
        source_mode=str(manifest["source_mode"]),
        room=str(manifest["room"]),
        frame_count=len(records),
        prediction_count=len(predictions),
        valid_prediction_count=len(valid_predictions),
        unknown_prediction_count=len(predictions) - len(valid_predictions),
        post_warmup_unknown_count=len(post_warmup_unknown),
        unknown_reason_counts=dict(unknown_reasons),
        maximum_risk_score=(max(valid_scores) if valid_scores else None),
        maximum_score_timestamp=(
            str(maximum_prediction["timestamp"])
            if maximum_prediction is not None
            else None
        ),
        imminent_triggered=bool(imminent_predictions),
        first_imminent_timestamp=(
            str(first_imminent["timestamp"]) if first_imminent else None
        ),
        event_triggered=any(
            bool(prediction.get("event_triggered")) for prediction in predictions
        ),
        action_onset_seconds=_optional_float(manifest.get("action_onset_seconds")),
        impact_seconds=_optional_float(manifest.get("impact_seconds")),
        lead_time_seconds=lead_time,
        lead_time_reference=lead_reference,
        lead_time_status=lead_status,
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        model_version=str(manifest["model_version"]),
        threshold=float(manifest["threshold"]),
        confirmation_windows=int(manifest["confirmation_windows"]),
        shadow_only=True,
        formal_alerts_enabled=False,
        stability=stability.to_dict(),
        unknown_diagnosis=diagnosis,
        score_curve_csv=score_csv.name,
        score_curve_svg=score_svg.name,
    )
    _write_json(output_dir / "stability_report.json", stability.to_dict())
    _write_json(output_dir / "experiment_report.json", report.to_dict())
    return report


def summarize_experiment_root(
    experiment_root: str | Path,
) -> dict[str, object]:
    root = Path(experiment_root).resolve()
    report_paths = sorted(root.glob("*/experiment_report.json"))
    if not report_paths:
        raise ValueError(f"no experiment reports found under {root}")
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    checkpoint_hashes = {str(report["checkpoint_sha256"]) for report in reports}
    thresholds = {float(report["threshold"]) for report in reports}
    confirmation_windows = {
        int(report["confirmation_windows"]) for report in reports
    }
    if len(checkpoint_hashes) != 1:
        raise ValueError("cannot summarize sessions with different checkpoints")
    if len(thresholds) != 1:
        raise ValueError("cannot summarize sessions with different thresholds")
    if len(confirmation_windows) != 1:
        raise ValueError(
            "cannot summarize sessions with different confirmation windows"
        )

    rows: list[dict[str, object]] = []
    for report, path in zip(reports, report_paths, strict=True):
        stability = report["stability"]
        prediction_count = int(report["prediction_count"])
        valid_prediction_count = int(report["valid_prediction_count"])
        post_warmup_unknown_count = int(report["post_warmup_unknown_count"])
        post_warmup_count = valid_prediction_count + post_warmup_unknown_count
        rows.append(
            {
                "session_id": report["session_id"],
                "action_category": report["action_category"],
                "action_name": report["action_name"],
                "maximum_risk_score": report["maximum_risk_score"],
                "imminent_triggered": bool(report["imminent_triggered"]),
                "lead_time_seconds": report["lead_time_seconds"],
                "lead_time_status": report["lead_time_status"],
                "valid_prediction_count": valid_prediction_count,
                "unknown_prediction_count": int(report["unknown_prediction_count"]),
                "post_warmup_unknown_count": post_warmup_unknown_count,
                "all_output_unknown_rate": (
                    int(report["unknown_prediction_count"]) / prediction_count
                    if prediction_count
                    else 0.0
                ),
                "post_warmup_unknown_rate": (
                    post_warmup_unknown_count / post_warmup_count
                    if post_warmup_count
                    else 0.0
                ),
                "observed_frame_rate_hz": stability["observed_frame_rate_hz"],
                "missing_frame_rate": stability["missing_frame_rate"],
                "critical_gap_count": stability["critical_gap_count"],
                "report_file": str(path.relative_to(root)),
            }
        )

    category_summary: dict[str, dict[str, object]] = {}
    for category in ACTION_CATEGORIES:
        members = [row for row in rows if row["action_category"] == category]
        if not members:
            continue
        maximum_scores = [
            float(row["maximum_risk_score"])
            for row in members
            if row["maximum_risk_score"] is not None
        ]
        lead_times = [
            float(row["lead_time_seconds"])
            for row in members
            if row["lead_time_seconds"] is not None
        ]
        total_predictions = sum(
            int(row["valid_prediction_count"])
            + int(row["unknown_prediction_count"])
            for row in members
        )
        total_unknown = sum(
            int(row["unknown_prediction_count"]) for row in members
        )
        total_post_warmup_unknown = sum(
            int(row["post_warmup_unknown_count"]) for row in members
        )
        total_post_warmup = sum(
            int(row["valid_prediction_count"])
            + int(row["post_warmup_unknown_count"])
            for row in members
        )
        category_summary[category] = {
            "session_count": len(members),
            "imminent_trigger_count": sum(
                bool(row["imminent_triggered"]) for row in members
            ),
            "maximum_score": max(maximum_scores) if maximum_scores else None,
            "mean_session_maximum_score": (
                statistics.mean(maximum_scores) if maximum_scores else None
            ),
            "median_lead_time_seconds": (
                statistics.median(lead_times) if lead_times else None
            ),
            "all_output_unknown_rate": (
                total_unknown / total_predictions if total_predictions else 0.0
            ),
            "post_warmup_unknown_rate": (
                total_post_warmup_unknown / total_post_warmup
                if total_post_warmup
                else 0.0
            ),
        }
    summary: dict[str, object] = {
        "schema_version": "radar_real_scene_suite_summary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root),
        "session_count": len(rows),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "threshold": next(iter(thresholds)),
        "confirmation_windows": next(iter(confirmation_windows)),
        "shadow_only": True,
        "formal_alerts_enabled": False,
        "category_summary": category_summary,
        "sessions": rows,
    }
    _write_json(root / "experiment_suite_summary.json", summary)
    csv_path = root / "experiment_suite_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _session_record(
    frame: RadarFrame,
    prediction: dict[str, object] | None,
    decoded: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "timestamp": frame.timestamp.isoformat(),
        "device_id": frame.device_id,
        "room": frame.room.value,
        "source_mode": frame.source_mode.value,
        "points": [
            {
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "velocity": point.velocity,
                **({"snr": point.snr} if point.snr is not None else {}),
                **(
                    {"track_id": point.track_id}
                    if point.track_id is not None
                    else {}
                ),
            }
            for point in frame.points
        ],
        "accepted_point_count": len(frame.points),
        "raw_point_count": len(decoded.get("points", ())),
        "ti_frame_number": _optional_int(decoded.get("ti_frame_number")),
        "ti_parser_error": _optional_int(decoded.get("ti_parser_error")),
        "tcn_prediction": prediction,
    }


def _diagnose_unknown(
    stability: Iwr6843StabilityReportV1,
    *,
    post_warmup_unknown_predictions: Sequence[Mapping[str, object]],
    unknown_reasons: dict[str, int],
) -> dict[str, object]:
    post_warmup_unknown_count = len(post_warmup_unknown_predictions)
    unresolved_gap_values = [
        float(prediction.get("longest_unresolved_gap_seconds", 0.0))
        for prediction in post_warmup_unknown_predictions
    ]
    missing_ratio_values = [
        float(prediction.get("missing_frame_ratio", 0.0))
        for prediction in post_warmup_unknown_predictions
    ]
    unresolved_point_gap_count = sum(
        value > 0.25 for value in unresolved_gap_values
    )
    alignment_missing_count = sum(value > 0.20 for value in missing_ratio_values)
    causes: list[str] = []
    if stability.parser_error_frame_count:
        causes.append("ti_parser_errors_observed")
    if (stability.exact_missing_frame_count or 0) > 0:
        causes.append("ti_frame_number_gaps_observed")
    if stability.critical_gap_count:
        if stability.exact_missing_frame_count == 0:
            causes.append("host_decode_delivery_gaps_with_sequential_ti_frames")
        else:
            causes.append("decoded_stream_intervals_exceed_quality_contract")
    if (
        post_warmup_unknown_count
        and unresolved_point_gap_count == post_warmup_unknown_count
        and stability.exact_missing_frame_count == 0
        and stability.critical_gap_count == 0
    ):
        causes.append("consecutive_empty_or_filtered_point_frames")
    elif unresolved_point_gap_count:
        causes.append("unresolved_point_feature_gaps")
    if alignment_missing_count:
        causes.append("feature_grid_alignment_missing_frames")
    if post_warmup_unknown_count and not causes:
        causes.append("feature_coverage_or_point_availability_requires_review")
    if not post_warmup_unknown_count:
        causes.append("no_post_warmup_unknown_observed")
    return {
        "post_warmup_unknown_count": post_warmup_unknown_count,
        "unknown_reason_counts": unknown_reasons,
        "unknown_with_unresolved_point_gap_count": unresolved_point_gap_count,
        "unknown_with_alignment_missing_count": alignment_missing_count,
        "maximum_unresolved_point_gap_seconds": (
            max(unresolved_gap_values) if unresolved_gap_values else 0.0
        ),
        "maximum_missing_frame_ratio": (
            max(missing_ratio_values) if missing_ratio_values else 0.0
        ),
        "likely_causes": causes,
        "quality_limit_changed": False,
        "recommended_next_check": (
            "compare TI frame-number continuity with host timestamp gaps before "
            "changing any feature-quality limit"
        ),
    }


def _lead_time(
    *,
    first_timestamp: datetime | None,
    first_imminent: Mapping[str, object] | None,
    impact_seconds: float | None,
) -> tuple[float | None, str | None, str]:
    if impact_seconds is None:
        return None, None, "impact_not_annotated"
    if first_timestamp is None:
        return None, "manual_impact", "empty_session"
    if first_imminent is None:
        return None, "manual_impact", "imminent_not_triggered"
    impact_timestamp = first_timestamp + timedelta(seconds=impact_seconds)
    lead = (
        impact_timestamp - _parse_timestamp(first_imminent["timestamp"])
    ).total_seconds()
    return lead, "manual_impact", (
        "triggered_before_impact" if lead >= 0.0 else "triggered_after_impact"
    )


def _write_score_csv(
    path: Path,
    records: Sequence[dict[str, object]],
    first_timestamp: datetime | None,
) -> None:
    lines = [
        "relative_seconds,timestamp,score,score_valid,risk_state,data_quality,unknown_reason"
    ]
    for record in records:
        prediction = record.get("tcn_prediction")
        if not isinstance(prediction, dict):
            continue
        relative = (
            (_parse_timestamp(record["timestamp"]) - first_timestamp).total_seconds()
            if first_timestamp is not None
            else 0.0
        )
        score = (
            str(float(prediction["pre_fall_score"]))
            if bool(prediction["score_valid"])
            else ""
        )
        lines.append(
            ",".join(
                [
                    f"{relative:.6f}",
                    str(prediction["timestamp"]),
                    score,
                    str(bool(prediction["score_valid"])).lower(),
                    str(prediction["risk_state"]),
                    str(prediction["data_quality"]),
                    str(prediction.get("unknown_reason") or ""),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_score_svg(
    path: Path,
    records: Sequence[dict[str, object]],
    *,
    first_timestamp: datetime | None,
    threshold: float,
    action_name: str,
    action_onset_seconds: float | None,
    impact_seconds: float | None,
) -> None:
    width, height = 1000, 420
    left, right, top, bottom = 70, 25, 45, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    points: list[tuple[float, float | None]] = []
    for record in records:
        prediction = record.get("tcn_prediction")
        if not isinstance(prediction, dict) or first_timestamp is None:
            continue
        relative = (
            _parse_timestamp(record["timestamp"]) - first_timestamp
        ).total_seconds()
        score = (
            float(prediction["pre_fall_score"])
            if bool(prediction["score_valid"])
            else None
        )
        points.append((relative, score))
    maximum_time = max((value[0] for value in points), default=1.0)
    maximum_time = max(maximum_time, impact_seconds or 0.0, 1.0)

    def x_pos(seconds: float) -> float:
        return left + plot_width * seconds / maximum_time

    def y_pos(score: float) -> float:
        return top + plot_height * (1.0 - score)

    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="17">TCN score — {escape(action_name)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#333"/>',
    ]
    for tick in range(6):
        score = tick / 5
        y = y_pos(score)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e6e6e6"/>'
        )
        elements.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="11">{score:.1f}</text>'
        )
    threshold_y = y_pos(threshold)
    elements.append(
        f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{left+plot_width}" y2="{threshold_y:.1f}" stroke="#d62728" stroke-dasharray="7 5"/>'
    )
    elements.append(
        f'<text x="{left+plot_width-4}" y="{threshold_y-6:.1f}" text-anchor="end" fill="#d62728" font-size="11">threshold {threshold:.2f}</text>'
    )
    segments: list[list[str]] = [[]]
    for seconds, score in points:
        if score is None:
            x = x_pos(seconds)
            elements.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}" stroke="#bdbdbd" opacity="0.45"/>'
            )
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(f"{x_pos(seconds):.1f},{y_pos(score):.1f}")
    for segment in segments:
        if segment:
            elements.append(
                f'<polyline points="{" ".join(segment)}" fill="none" stroke="#1565c0" stroke-width="2"/>'
            )
    for seconds, color, label in (
        (action_onset_seconds, "#ff9800", "action onset"),
        (impact_seconds, "#7b1fa2", "impact"),
    ):
        if seconds is not None:
            x = x_pos(seconds)
            elements.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}" stroke="{color}" stroke-width="2"/>'
            )
            elements.append(
                f'<text x="{x+4:.1f}" y="{top+14}" fill="{color}" font-size="11">{label}</text>'
            )
    elements.extend(
        [
            f'<text x="{width/2}" y="{height-13}" text-anchor="middle" font-size="12">seconds from first decoded frame</text>',
            f'<text x="17" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90 17 {height/2})">pre-fall score</text>',
            '<text x="75" y="405" font-size="10" fill="#777">grey vertical marks = UNKNOWN</text>',
        ]
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>",
        encoding="utf-8",
    )


def _sample_from_record(record: Mapping[str, object]) -> FrameStabilitySampleV1:
    return FrameStabilitySampleV1(
        timestamp=_parse_timestamp(record["timestamp"]),
        point_count=int(record.get("raw_point_count", len(record.get("points", [])))),
        ti_frame_number=_optional_int(record.get("ti_frame_number")),
        ti_parser_error=_optional_int(record.get("ti_parser_error")),
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        raise ValueError("decoded timestamp is missing or unsupported")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("decoded timestamp must include timezone")
    return parsed


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _validate_optional_seconds(value: object, name: str) -> None:
    if value is None:
        return
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_session_metadata(
    *,
    session_id: str,
    action_category: str,
    action_name: str,
    duration_seconds: float | None,
    action_onset_seconds: float | None,
    impact_seconds: float | None,
) -> None:
    if not session_id.strip() or any(character in session_id for character in "\\/:"):
        raise ValueError("session_id must be non-empty and path-safe")
    if action_category not in ACTION_CATEGORIES:
        raise ValueError(f"unsupported action_category: {action_category}")
    if not action_name.strip():
        raise ValueError("action_name must not be blank")
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds <= 0.0
    ):
        raise ValueError("duration_seconds must be finite and positive")
    _validate_optional_seconds(action_onset_seconds, "action_onset_seconds")
    _validate_optional_seconds(impact_seconds, "impact_seconds")


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"env file does not exist: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _default_checkpoint() -> Path:
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / "checkpoints"
        / "experiments_v5"
        / "tcn_hard_negative"
        / "tcn_0p5_1p0_specificity_operating_point_v1.pt"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record frozen-TCN real-scene validation sessions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--source", choices=("real", "replay"), required=True)
    capture.add_argument("--replay")
    capture.add_argument("--env-file", default=".env")
    capture.add_argument(
        "--reuse-existing-config",
        action="store_true",
        help=(
            "diagnostic-only TI 'sensorStart 0' restart after a successful "
            "full-config run; do not use for repeated action sessions"
        ),
    )
    capture.add_argument("--duration-seconds", type=float)
    capture.add_argument("--session-id", required=True)
    capture.add_argument("--action-category", choices=ACTION_CATEGORIES, required=True)
    capture.add_argument("--action-name", required=True)
    capture.add_argument("--action-onset-seconds", type=float)
    capture.add_argument("--impact-seconds", type=float)
    capture.add_argument("--room", choices=tuple(room.value for room in Room), default="bathroom")
    capture.add_argument("--device-id", default="iwr6843isk-01")
    capture.add_argument("--expected-frame-rate-hz", type=float)
    capture.add_argument("--checkpoint", type=Path, default=_default_checkpoint())
    capture.add_argument("--sha256", default=DEFAULT_CHECKPOINT_SHA256)
    capture.add_argument("--confirmation-windows", type=int, default=3)
    capture.add_argument("--output-root", type=Path, default=Path("reports/real_scene_validation_v1"))

    report = subparsers.add_parser("report")
    report.add_argument("--session-dir", type=Path, required=True)
    report.add_argument("--action-onset-seconds", type=float)
    report.add_argument("--impact-seconds", type=float)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--experiment-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "summary":
        summary = summarize_experiment_root(args.experiment_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        report = regenerate_experiment_report(
            args.session_dir,
            action_onset_seconds=args.action_onset_seconds,
            impact_seconds=args.impact_seconds,
        )
    else:
        if args.source == "real":
            if args.duration_seconds is None:
                raise SystemExit("--duration-seconds is required for real capture")
            _load_env_file(Path(args.env_file).resolve())
            command_json = os.getenv("TI_OFFICIAL_OUTPUT_COMMAND_JSON", "")
            if not command_json:
                raise SystemExit("TI_OFFICIAL_OUTPUT_COMMAND_JSON is missing")
            command = json.loads(command_json)
            if not isinstance(command, list) or not all(
                isinstance(item, str) and item for item in command
            ):
                raise SystemExit("TI_OFFICIAL_OUTPUT_COMMAND_JSON must be a string array")
            if args.reuse_existing_config and "--reuse-existing-config" not in command:
                command.append("--reuse-existing-config")
            cwd = os.getenv("TI_OFFICIAL_OUTPUT_CWD", "").strip() or None
            source: RadarSourceAdapter = TiOfficialOutputAdapter(
                command=command,
                cwd=cwd,
            )
            expected_rate = (
                args.expected_frame_rate_hz or DEFAULT_EXPECTED_FRAME_RATE_HZ
            )
        else:
            if args.reuse_existing_config:
                raise SystemExit("--reuse-existing-config is valid only for real capture")
            if not args.replay:
                raise SystemExit("--replay is required for replay capture")
            source = JsonlReplayAdapter(args.replay, speed=100_000.0)
            expected_rate = args.expected_frame_rate_hz or 10.0
        output_dir = args.output_root.resolve() / args.session_id
        report = run_validation_session(
            source,
            checkpoint_path=args.checkpoint,
            expected_checkpoint_sha256=args.sha256,
            output_directory=output_dir,
            session_id=args.session_id,
            action_category=args.action_category,
            action_name=args.action_name,
            room=args.room,
            device_id=args.device_id,
            duration_seconds=args.duration_seconds,
            expected_frame_rate_hz=expected_rate,
            confirmation_windows=args.confirmation_windows,
            action_onset_seconds=args.action_onset_seconds,
            impact_seconds=args.impact_seconds,
        )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
