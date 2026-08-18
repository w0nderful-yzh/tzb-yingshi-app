from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from radar_module.acquisition.ti_reader import (
    JsonlReplayAdapter,
    RadarSourceAdapter,
    TiOfficialOutputAdapter,
    TiRadarReader,
)
from radar_module.contracts import Room
from radar_module.inference.tcn_live_v1 import RadarTcnLivePredictorV1
from radar_module.validation.iwr6843_stability_v1 import (
    FrameStabilitySampleV1,
    analyze_iwr6843_stability,
)
from radar_module.validation.real_scene_validation_v1 import (
    ACTION_CATEGORIES,
    DEFAULT_CHECKPOINT_SHA256,
    DEFAULT_EXPECTED_FRAME_RATE_HZ,
    DiagnosticRecordingAdapterV1,
    _default_checkpoint,
    _load_env_file,
    _session_record,
    _write_json,
    _write_jsonl,
    generate_experiment_report,
)


@dataclass(frozen=True, slots=True)
class ValidationPhaseV1:
    phase_id: str
    action_category: str
    action_name: str
    duration_seconds: float
    instruction: str = ""
    cue_beeps: int = 1
    action_onset_seconds: float | None = None
    impact_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContinuousValidationReportV1:
    schema_version: str
    session_id: str
    source_mode: str
    room: str
    frame_count: int
    phase_count: int
    total_planned_duration_seconds: float
    checkpoint_sha256: str
    model_version: str
    threshold: float
    confirmation_windows: int
    shadow_only: bool
    formal_alerts_enabled: bool
    stability: dict[str, object]
    phases: list[dict[str, object]]
    session_file: str
    phase_events_file: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PhaseCallback = Callable[[ValidationPhaseV1, int], None]


def load_phase_plan(path: str | Path) -> list[ValidationPhaseV1]:
    plan_path = Path(path).resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "radar_validation_phase_plan_v1":
        raise ValueError("unsupported phase-plan schema_version")
    raw_phases = payload.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("phase plan must contain a non-empty phases array")
    phases: list[ValidationPhaseV1] = []
    for raw_phase in raw_phases:
        if not isinstance(raw_phase, Mapping):
            raise ValueError("each phase must be an object")
        phases.append(
            ValidationPhaseV1(
                phase_id=str(raw_phase.get("phase_id", "")),
                action_category=str(raw_phase.get("action_category", "")),
                action_name=str(raw_phase.get("action_name", "")),
                duration_seconds=float(raw_phase.get("duration_seconds", 0.0)),
                instruction=str(raw_phase.get("instruction", "")),
                cue_beeps=int(raw_phase.get("cue_beeps", 1)),
                action_onset_seconds=_optional_float(
                    raw_phase.get("action_onset_seconds")
                ),
                impact_seconds=_optional_float(raw_phase.get("impact_seconds")),
            )
        )
    _validate_phases(phases)
    return phases


def run_continuous_validation(
    source_adapter: RadarSourceAdapter,
    *,
    phases: Sequence[ValidationPhaseV1],
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    output_directory: str | Path,
    session_id: str,
    room: Room | str = Room.BATHROOM,
    device_id: str = "iwr6843isk-01",
    expected_frame_rate_hz: float = DEFAULT_EXPECTED_FRAME_RATE_HZ,
    confirmation_windows: int = 3,
    torch_device: str = "cpu",
    phase_callback: PhaseCallback | None = None,
) -> ContinuousValidationReportV1:
    _validate_session_id(session_id)
    _validate_phases(phases)
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
    phase_records: dict[str, list[dict[str, object]]] = {
        phase.phase_id: [] for phase in phases
    }
    phase_samples: dict[str, list[FrameStabilitySampleV1]] = {
        phase.phase_id: [] for phase in phases
    }
    all_records: list[dict[str, object]] = []
    all_samples: list[FrameStabilitySampleV1] = []
    phase_events_path = output_dir / "phase_events.jsonl"
    phase_events_path.write_text("", encoding="utf-8")
    total_duration = sum(phase.duration_seconds for phase in phases)
    cumulative_ends: list[float] = []
    cumulative = 0.0
    for phase in phases:
        cumulative += phase.duration_seconds
        cumulative_ends.append(cumulative)

    first_frame_time: datetime | None = None
    phase_index = 0
    announced_phase = -1
    try:
        reader.start()
        while phase_index < len(phases):
            frame = reader.read()
            if frame is None:
                if (
                    isinstance(source_adapter, JsonlReplayAdapter)
                    and source_adapter.finished
                ):
                    break
                continue
            if first_frame_time is None:
                first_frame_time = frame.timestamp
            elapsed = (frame.timestamp - first_frame_time).total_seconds()
            while (
                phase_index < len(phases)
                and elapsed >= cumulative_ends[phase_index]
            ):
                phase_index += 1
            if phase_index >= len(phases) or elapsed >= total_duration:
                break
            phase = phases[phase_index]
            if announced_phase != phase_index:
                announced_phase = phase_index
                _append_jsonl(
                    phase_events_path,
                    {
                        "event": "PHASE_START",
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "frame_timestamp": frame.timestamp.isoformat(),
                        "phase_index": phase_index,
                        **phase.to_dict(),
                    },
                )
                if phase_callback is not None:
                    phase_callback(phase, phase_index)
            prediction = predictor.consume(frame)
            decoded = diagnostic_adapter.last_decoded or {}
            record = _session_record(
                frame,
                prediction.to_dict() if prediction is not None else None,
                decoded,
            )
            record.update(
                {
                    "phase_id": phase.phase_id,
                    "phase_index": phase_index,
                    "phase_action_category": phase.action_category,
                    "phase_action_name": phase.action_name,
                }
            )
            all_records.append(record)
            all_samples.append(diagnostic_adapter.samples[-1])
            phase_records[phase.phase_id].append(record)
            phase_samples[phase.phase_id].append(diagnostic_adapter.samples[-1])
    except BaseException as exc:
        _write_json(
            output_dir / "capture_error.json",
            {
                "schema_version": "radar_continuous_capture_error_v1",
                "session_id": session_id,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "captured_frame_count": len(all_records),
                "current_phase_index": phase_index,
                "checkpoint_sha256": predictor.checkpoint_sha256,
                "model_version": predictor.model_version,
                "threshold": predictor.threshold,
                "shadow_only": True,
                "formal_alerts_enabled": False,
            },
        )
        if all_records:
            _write_jsonl(output_dir / "partial_session.jsonl", all_records)
        raise
    finally:
        reader.stop()

    session_path = output_dir / "session.jsonl"
    _write_jsonl(session_path, all_records)
    phase_summaries: list[dict[str, object]] = []
    for index, phase in enumerate(phases):
        records = phase_records[phase.phase_id]
        phase_dir = output_dir / "phases" / phase.phase_id
        phase_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(phase_dir / "session.jsonl", records)
        phase_manifest: dict[str, object] = {
            "schema_version": "radar_real_scene_session_v1",
            "session_id": f"{session_id}__{phase.phase_id}",
            "action_category": phase.action_category,
            "action_name": phase.action_name,
            "room": Room(room).value,
            "device_id": device_id,
            "source_mode": source_adapter.source_mode.value,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "session_file": "session.jsonl",
            "expected_frame_rate_hz": expected_frame_rate_hz,
            "action_onset_seconds": phase.action_onset_seconds,
            "impact_seconds": phase.impact_seconds,
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "model_version": predictor.model_version,
            "threshold": predictor.threshold,
            "confirmation_windows": confirmation_windows,
            "shadow_only": True,
            "formal_alerts_enabled": False,
            "continuous_parent_session_id": session_id,
            "phase_id": phase.phase_id,
            "phase_index": index,
        }
        _write_json(phase_dir / "session_manifest.json", phase_manifest)
        phase_report = generate_experiment_report(
            phase_dir,
            records=records,
            manifest=phase_manifest,
            stability_samples=phase_samples[phase.phase_id],
        )
        phase_summaries.append(
            {
                **phase.to_dict(),
                "frame_count": phase_report.frame_count,
                "valid_prediction_count": phase_report.valid_prediction_count,
                "unknown_prediction_count": phase_report.unknown_prediction_count,
                "post_warmup_unknown_count": (
                    phase_report.post_warmup_unknown_count
                ),
                "maximum_risk_score": phase_report.maximum_risk_score,
                "imminent_triggered": phase_report.imminent_triggered,
                "event_triggered": phase_report.event_triggered,
                "lead_time_seconds": phase_report.lead_time_seconds,
                "report_file": str(
                    (phase_dir / "experiment_report.json").relative_to(output_dir)
                ),
            }
        )

    stability = analyze_iwr6843_stability(
        all_samples,
        expected_frame_rate_hz=expected_frame_rate_hz,
    )
    manifest: dict[str, object] = {
        "schema_version": "radar_continuous_session_v1",
        "session_id": session_id,
        "room": Room(room).value,
        "device_id": device_id,
        "source_mode": source_adapter.source_mode.value,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "session_file": session_path.name,
        "expected_frame_rate_hz": expected_frame_rate_hz,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "confirmation_windows": confirmation_windows,
        "shadow_only": True,
        "formal_alerts_enabled": False,
        "phases": [phase.to_dict() for phase in phases],
        "phase_events_file": phase_events_path.name,
    }
    _write_json(output_dir / "session_manifest.json", manifest)
    _write_json(output_dir / "stability_report.json", stability.to_dict())
    report = ContinuousValidationReportV1(
        schema_version="radar_continuous_validation_report_v1",
        session_id=session_id,
        source_mode=source_adapter.source_mode.value,
        room=Room(room).value,
        frame_count=len(all_records),
        phase_count=len(phases),
        total_planned_duration_seconds=total_duration,
        checkpoint_sha256=predictor.checkpoint_sha256,
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        confirmation_windows=confirmation_windows,
        shadow_only=True,
        formal_alerts_enabled=False,
        stability=stability.to_dict(),
        phases=phase_summaries,
        session_file=session_path.name,
        phase_events_file=phase_events_path.name,
    )
    _write_json(output_dir / "sequence_report.json", report.to_dict())
    return report


def _validate_session_id(session_id: str) -> None:
    if not session_id.strip() or any(character in session_id for character in "\\/:"):
        raise ValueError("session_id must be non-empty and path-safe")


def _validate_phases(phases: Sequence[ValidationPhaseV1]) -> None:
    if not phases:
        raise ValueError("at least one validation phase is required")
    seen: set[str] = set()
    for phase in phases:
        _validate_session_id(phase.phase_id)
        if phase.phase_id in seen:
            raise ValueError(f"duplicate phase_id: {phase.phase_id}")
        seen.add(phase.phase_id)
        if phase.action_category not in ACTION_CATEGORIES:
            raise ValueError(
                f"unsupported action_category: {phase.action_category}"
            )
        if not phase.action_name.strip():
            raise ValueError("phase action_name must not be blank")
        if not math.isfinite(phase.duration_seconds) or phase.duration_seconds <= 0:
            raise ValueError("phase duration_seconds must be finite and positive")
        if not 0 <= phase.cue_beeps <= 5:
            raise ValueError("phase cue_beeps must be between 0 and 5")
        for name, value in (
            ("action_onset_seconds", phase.action_onset_seconds),
            ("impact_seconds", phase.impact_seconds),
        ):
            if value is not None and (
                not math.isfinite(value) or value < 0 or value > phase.duration_seconds
            ):
                raise ValueError(
                    f"phase {name} must fall within phase duration"
                )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def _announce_phase(phase: ValidationPhaseV1, index: int) -> None:
    print(
        json.dumps(
            {
                "event": "PHASE_START",
                "phase_index": index,
                **phase.to_dict(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    _emit_audible_cue(phase.cue_beeps)


def _emit_audible_cue(count: int) -> None:
    if count <= 0:
        return
    try:
        import winsound
    except ImportError:
        return
    frequency_hz = 1100 if count >= 3 else 880
    for index in range(count):
        winsound.Beep(frequency_hz, 180)
        if index + 1 < count:
            time.sleep(0.12)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one radar start across multiple labelled action phases."
    )
    parser.add_argument("--source", choices=("real", "replay"), required=True)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--replay")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--room",
        choices=tuple(room.value for room in Room),
        default=Room.BATHROOM.value,
    )
    parser.add_argument("--device-id", default="iwr6843isk-01")
    parser.add_argument("--expected-frame-rate-hz", type=float)
    parser.add_argument("--checkpoint", type=Path, default=_default_checkpoint())
    parser.add_argument("--sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--confirmation-windows", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/continuous_scene_validation_v1"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    phases = load_phase_plan(args.phase_plan)
    if args.source == "real":
        _load_env_file(Path(args.env_file).resolve())
        command_json = os.getenv("TI_OFFICIAL_OUTPUT_COMMAND_JSON", "")
        if not command_json:
            raise SystemExit("TI_OFFICIAL_OUTPUT_COMMAND_JSON is missing")
        command = json.loads(command_json)
        if not isinstance(command, list) or not all(
            isinstance(item, str) and item for item in command
        ):
            raise SystemExit("TI_OFFICIAL_OUTPUT_COMMAND_JSON must be a string array")
        if "--reuse-existing-config" in command:
            raise SystemExit(
                "continuous validation requires a cold-start full-config command"
            )
        cwd = os.getenv("TI_OFFICIAL_OUTPUT_CWD", "").strip() or None
        source: RadarSourceAdapter = TiOfficialOutputAdapter(command=command, cwd=cwd)
        expected_rate = args.expected_frame_rate_hz or DEFAULT_EXPECTED_FRAME_RATE_HZ
    else:
        if not args.replay:
            raise SystemExit("--replay is required for replay mode")
        source = JsonlReplayAdapter(args.replay, speed=100_000.0)
        expected_rate = args.expected_frame_rate_hz or 10.0
    report = run_continuous_validation(
        source,
        phases=phases,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.sha256,
        output_directory=args.output_root.resolve() / args.session_id,
        session_id=args.session_id,
        room=args.room,
        device_id=args.device_id,
        expected_frame_rate_hz=expected_rate,
        confirmation_windows=args.confirmation_windows,
        phase_callback=_announce_phase,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
