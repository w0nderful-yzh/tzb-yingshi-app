from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_module.dataset.dguha_research_v2 import (
    DGUHA_FALL_ACTION,
    DGUHA_SPLIT_BY_SUBJECT,
)
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.model.radar_lstm import RadarLSTM
from radar_module.model.point_prediction_v1 import POINT_PREDICTION_MODEL_VERSION
from radar_module.model.point_iwr6843_adaptation_v1 import (
    FEATURE_NAMES as ADAPTATION_FEATURE_NAMES,
    PREDICTION_VERSION as ADAPTATION_PREDICTION_VERSION,
    SEQUENCE_VERSION as ADAPTATION_SEQUENCE_VERSION,
)
from radar_module.model.pointnet_formal_prediction_v1 import (
    FORMAL_MODEL_VERSION,
)
from radar_module.model.point_temporal import (
    PointTemporalEncoder,
    PointTemporalPredictionHead,
)
from radar_module.model.research_training_v2 import (
    RESEARCH_MODEL_MODE,
    RESEARCH_MODEL_VERSION,
)
from radar_module.model.temporal_models_v3 import (
    EXPERIMENT_MODEL_VERSION,
    MULTIHORIZON_MODEL_VERSION,
    MULTITASK_MODEL_VERSION,
    MultiHorizonTemporalModel,
    SharedMultiTaskTemporalModel,
    TemporalBinaryModel,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
    WINDOW_SIZE_V2,
)
from radar_module.preprocess.relative_temporal_features_v3 import (
    FEATURE_NAMES_V3,
    FEATURE_VERSION_V3,
    transform_v2_values_to_v3,
)
from radar_module.preprocess.hybrid_temporal_features_v4 import (
    FEATURE_NAMES_V4,
    FEATURE_VERSION_V4,
    transform_v2_values_to_v4,
)
from radar_module.preprocess.pointcloud_sequence import (
    POINT_FEATURE_NAMES,
    POINT_SEQUENCE_VERSION,
    PointCloudSequenceBuilder,
)


def evaluate_dguha_events(
    data_root: str | Path,
    events_path: str | Path,
    checkpoint_path: str | Path,
    report_path: str | Path,
    *,
    split: str = "test",
    confirmation_windows: int = 3,
    step_seconds: float = 0.1,
    minimum_lead_seconds: float = 0.1,
    maximum_lead_seconds: float = 0.6,
    evaluation_anchor: str = "descent_onset",
    minimum_pre_descent_margin_seconds: float = 0.1,
    early_negative_minimum_lead_seconds: float = 1.2,
    decision_threshold_override: float | None = None,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if confirmation_windows < 2 or step_seconds <= 0:
        raise ValueError("invalid confirmation settings")
    if early_negative_minimum_lead_seconds <= maximum_lead_seconds:
        raise ValueError("early-negative boundary must be beyond the prediction corridor")
    if evaluation_anchor not in {"descent_onset", "near_floor_level_reached"}:
        raise ValueError("unsupported evaluation anchor")
    source_root = Path(data_root).resolve()
    event_file = Path(events_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    destination = Path(report_path).resolve()
    if not source_root.is_dir() or not event_file.is_file():
        raise FileNotFoundError("DGUHA data root and events file must exist")
    if not checkpoint_file.is_file():
        raise FileNotFoundError("research checkpoint does not exist")

    events = json.loads(event_file.read_text(encoding="utf-8"))
    event_by_source = {str(event["source_file"]): event for event in events}
    checkpoint = _load_checkpoint(checkpoint_file)
    model = _build_prefall_model(checkpoint)
    use_relative_features_v3 = checkpoint.get("feature_version") == FEATURE_VERSION_V3
    use_hybrid_features_v4 = checkpoint.get("feature_version") == FEATURE_VERSION_V4
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    threshold = _resolve_evaluation_threshold(
        checkpoint,
        decision_threshold_override,
    )
    sweep_thresholds = sorted(
        set(
            [threshold]
            + [0.005, 0.01, 0.02, 0.03, 0.04]
            + np.linspace(0.01, 0.99, 99, dtype=np.float64).tolist()
        )
    )
    sweep = {
        value: {
            "corridor_detected": 0,
            "any_pre_onset_detected": 0,
            "normal_runs": 0,
            "normal_recordings": 0,
            "normal_high_windows": 0,
            "normal_confirmed_active_windows": 0,
            "normal_total_windows": 0,
            "same_recording_early_high_windows": 0,
            "same_recording_early_total_windows": 0,
        }
        for value in sweep_thresholds
    }
    point_input = checkpoint["model_version"] in {
        POINT_PREDICTION_MODEL_VERSION,
        ADAPTATION_PREDICTION_VERSION,
        FORMAL_MODEL_VERSION,
    }
    extractor = RadarTemporalFeatureExtractorV2()
    point_builder = PointCloudSequenceBuilder()
    required_history = (
        (point_builder.time_steps - 1) / point_builder.sample_rate_hz
        if point_input
        else (extractor.window_size - 1) / extractor.target_sample_rate_hz
    )

    fall_results: list[dict[str, Any]] = []
    normal_results: list[dict[str, Any]] = []
    score_groups: dict[str, list[float]] = defaultdict(list)
    radar_files = sorted(source_root.glob("*/*/radar/*.txt"))
    for radar_path in radar_files:
        relative = radar_path.relative_to(source_root).as_posix()
        subject = _subject_id(radar_path.name)
        if DGUHA_SPLIT_BY_SUBJECT[subject] != split:
            continue
        action = radar_path.parent.parent.name
        event = event_by_source.get(relative)
        if action == DGUHA_FALL_ACTION and (
            event is None or not bool(event["eligible_for_prediction_windows"])
            or event.get(evaluation_anchor) is None
        ):
            continue

        frames = parse_radhar_text(
            radar_path, device_id=f"dguha-event-eval-{radar_path.stem}"
        )
        radar_start = frames[0].timestamp
        duration = (frames[-1].timestamp - radar_start).total_seconds()
        if action == DGUHA_FALL_ACTION:
            onset = datetime.fromisoformat(str(event["descent_onset"]))
            anchor = datetime.fromisoformat(str(event[evaluation_anchor]))
            maximum_end = (
                (onset - radar_start).total_seconds()
                - minimum_pre_descent_margin_seconds
            )
        else:
            onset = None
            anchor = None
            maximum_end = duration
        endpoints = np.arange(
            required_history,
            maximum_end + step_seconds * 0.25,
            step_seconds,
            dtype=np.float64,
        )
        if point_input:
            times, scores, skipped = _score_point_recording(
                frames,
                endpoints,
                radar_start,
                point_builder,
                model,
                mean,
                std,
            )
        else:
            times, scores, skipped = _score_recording(
                frames,
                endpoints,
                radar_start,
                extractor,
                model,
                mean,
                std,
                use_relative_features_v3=use_relative_features_v3,
                use_hybrid_features_v4=use_hybrid_features_v4,
            )
        high = scores >= threshold
        run_ends = _confirmed_run_end_indices(
            high, times, confirmation_windows, step_seconds
        )
        confirmed_active = _confirmed_active_mask(
            high, times, confirmation_windows, step_seconds
        )
        if action != DGUHA_FALL_ACTION:
            score_groups["normal"].extend(scores.tolist())
            normal_results.append(
                {
                    "source_file": relative,
                    "subject_id": subject,
                    "action": action,
                    "duration_seconds": duration,
                    "evaluated_window_count": len(scores),
                    "skipped_window_count": skipped,
                    "above_threshold_window_count": int(high.sum()),
                    "confirmed_run_count": len(run_ends),
                    "confirmed_active_window_count": int(confirmed_active.sum()),
                }
            )
            for sweep_threshold in sweep_thresholds:
                sweep_runs = _confirmed_run_end_indices(
                    scores >= sweep_threshold,
                    times,
                    confirmation_windows,
                    step_seconds,
                )
                sweep[sweep_threshold]["normal_runs"] += len(sweep_runs)
                sweep_high = scores >= sweep_threshold
                sweep[sweep_threshold]["normal_high_windows"] += int(
                    sweep_high.sum()
                )
                sweep[sweep_threshold]["normal_total_windows"] += len(scores)
                sweep[sweep_threshold]["normal_confirmed_active_windows"] += int(
                    _confirmed_active_mask(
                        sweep_high,
                        times,
                        confirmation_windows,
                        step_seconds,
                    ).sum()
                )
                sweep[sweep_threshold]["normal_recordings"] += int(
                    bool(sweep_runs)
                )
            continue

        onset_seconds = (onset - radar_start).total_seconds()
        anchor_seconds = (anchor - radar_start).total_seconds()
        leads = anchor_seconds - times
        onset_leads = onset_seconds - times
        early_negative = onset_leads >= early_negative_minimum_lead_seconds - 1e-6
        early_negative_scores = scores[early_negative]
        corridor = (
            (leads >= minimum_lead_seconds - 1e-6)
            & (leads <= maximum_lead_seconds + 1e-6)
        )
        corridor_run_ends = _confirmed_run_end_indices(
            high[corridor], times[corridor], confirmation_windows, step_seconds
        )
        corridor_scores = scores[corridor]
        for sweep_threshold in sweep_thresholds:
            sweep[sweep_threshold]["same_recording_early_high_windows"] += int(
                np.sum(early_negative_scores >= sweep_threshold)
            )
            sweep[sweep_threshold]["same_recording_early_total_windows"] += len(
                early_negative_scores
            )
            sweep[sweep_threshold]["any_pre_onset_detected"] += int(
                bool(
                    _confirmed_run_end_indices(
                        scores >= sweep_threshold,
                        times,
                        confirmation_windows,
                        step_seconds,
                    )
                )
            )
            sweep[sweep_threshold]["corridor_detected"] += int(
                bool(
                    _confirmed_run_end_indices(
                        corridor_scores >= sweep_threshold,
                        times[corridor],
                        confirmation_windows,
                        step_seconds,
                    )
                )
            )
        score_groups["prediction_corridor"].extend(corridor_scores.tolist())
        confirmation_leads = [float(leads[index]) for index in run_ends]
        corridor_confirmation_leads = [
            float(leads[np.flatnonzero(corridor)[index]])
            for index in corridor_run_ends
        ]
        fall_results.append(
            {
                "source_file": relative,
                "subject_id": subject,
                "onset_seconds": onset_seconds,
                "anchor_seconds": anchor_seconds,
                "evaluated_pre_onset_window_count": len(scores),
                "prediction_corridor_window_count": len(corridor_scores),
                "skipped_window_count": skipped,
                "any_pre_onset_confirmed": bool(run_ends),
                "prediction_corridor_confirmed": bool(corridor_run_ends),
                "earliest_pre_onset_confirmation_lead_seconds": (
                    max(confirmation_leads) if confirmation_leads else None
                ),
                "earliest_corridor_confirmation_lead_seconds": (
                    max(corridor_confirmation_leads)
                    if corridor_confirmation_leads
                    else None
                ),
                "corridor_score_max": (
                    float(corridor_scores.max()) if len(corridor_scores) else None
                ),
                "same_recording_early_negative_window_count": int(
                    len(early_negative_scores)
                ),
                "same_recording_early_negative_high_window_count": int(
                    np.sum(early_negative_scores >= threshold)
                ),
            }
        )

    normal_duration = sum(item["duration_seconds"] for item in normal_results)
    normal_runs = sum(item["confirmed_run_count"] for item in normal_results)
    normal_windows = sum(item["evaluated_window_count"] for item in normal_results)
    normal_high_windows = sum(
        item["above_threshold_window_count"] for item in normal_results
    )
    normal_active_windows = sum(
        item["confirmed_active_window_count"] for item in normal_results
    )
    corridor_detected = [
        item for item in fall_results if item["prediction_corridor_confirmed"]
    ]
    any_pre_onset = [
        item for item in fall_results if item["any_pre_onset_confirmed"]
    ]
    lead_values = [
        item["earliest_corridor_confirmation_lead_seconds"]
        for item in corridor_detected
    ]
    early_negative_windows = sum(
        item["same_recording_early_negative_window_count"] for item in fall_results
    )
    early_negative_high_windows = sum(
        item["same_recording_early_negative_high_window_count"] for item in fall_results
    )
    payload: dict[str, Any] = {
        "data_root": str(source_root),
        "events_file": str(event_file),
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "split": split,
        "threshold": threshold,
        "checkpoint_threshold": (
            float(checkpoint["decision_threshold"])
            if "decision_threshold" in checkpoint
            else None
        ),
        "threshold_source": (
            "checkpoint_validation_window"
            if decision_threshold_override is None
            else "external_locked_calibration"
        ),
        "confirmation_windows": confirmation_windows,
        "step_seconds": step_seconds,
        "prediction_lead_interval_seconds": [
            minimum_lead_seconds,
            maximum_lead_seconds,
        ],
        "evaluation_anchor": evaluation_anchor,
        "minimum_pre_descent_margin_seconds": minimum_pre_descent_margin_seconds,
        "same_recording_early_negative_minimum_lead_seconds": (
            early_negative_minimum_lead_seconds
        ),
        "eligible_fall_recording_count": len(fall_results),
        "prediction_corridor_detected_event_count": len(corridor_detected),
        "prediction_corridor_event_recall": (
            len(corridor_detected) / len(fall_results) if fall_results else 0.0
        ),
        "any_pre_onset_detected_event_count": len(any_pre_onset),
        "any_pre_onset_event_recall": (
            len(any_pre_onset) / len(fall_results) if fall_results else 0.0
        ),
        "corridor_confirmation_lead_seconds": _describe(lead_values),
        "same_recording_early_negative_window_count": early_negative_windows,
        "same_recording_early_negative_high_window_count": early_negative_high_windows,
        "same_recording_early_negative_false_positive_rate": (
            early_negative_high_windows / early_negative_windows
            if early_negative_windows
            else 0.0
        ),
        "normal_recording_count": len(normal_results),
        "normal_duration_seconds": normal_duration,
        "normal_confirmed_run_count": normal_runs,
        "normal_confirmed_runs_per_hour": (
            normal_runs * 3600.0 / normal_duration if normal_duration else 0.0
        ),
        "normal_above_threshold_window_fraction": (
            normal_high_windows / normal_windows if normal_windows else 0.0
        ),
        "normal_confirmed_active_window_fraction": (
            normal_active_windows / normal_windows if normal_windows else 0.0
        ),
        "normal_confirmed_active_seconds_per_hour": (
            normal_active_windows * step_seconds * 3600.0 / normal_duration
            if normal_duration
            else 0.0
        ),
        "normal_recordings_with_confirmed_run": sum(
            item["confirmed_run_count"] > 0 for item in normal_results
        ),
        "normal_score_distribution": _describe(score_groups["normal"]),
        "prediction_corridor_score_distribution": _describe(
            score_groups["prediction_corridor"]
        ),
        "threshold_sweep": [
            {
                "threshold": float(sweep_threshold),
                "prediction_corridor_detected_event_count": int(
                    sweep[sweep_threshold]["corridor_detected"]
                ),
                "prediction_corridor_event_recall": (
                    sweep[sweep_threshold]["corridor_detected"] / len(fall_results)
                    if fall_results
                    else 0.0
                ),
                "any_pre_onset_event_recall": (
                    sweep[sweep_threshold]["any_pre_onset_detected"]
                    / len(fall_results)
                    if fall_results
                    else 0.0
                ),
                "normal_confirmed_run_count": int(
                    sweep[sweep_threshold]["normal_runs"]
                ),
                "normal_confirmed_runs_per_hour": (
                    sweep[sweep_threshold]["normal_runs"] * 3600.0 / normal_duration
                    if normal_duration
                    else 0.0
                ),
                "normal_recordings_with_confirmed_run": int(
                    sweep[sweep_threshold]["normal_recordings"]
                ),
                "normal_above_threshold_window_fraction": (
                    sweep[sweep_threshold]["normal_high_windows"]
                    / sweep[sweep_threshold]["normal_total_windows"]
                    if sweep[sweep_threshold]["normal_total_windows"]
                    else 0.0
                ),
                "normal_confirmed_active_window_fraction": (
                    sweep[sweep_threshold]["normal_confirmed_active_windows"]
                    / sweep[sweep_threshold]["normal_total_windows"]
                    if sweep[sweep_threshold]["normal_total_windows"]
                    else 0.0
                ),
                "normal_confirmed_active_seconds_per_hour": (
                    sweep[sweep_threshold]["normal_confirmed_active_windows"]
                    * step_seconds
                    * 3600.0
                    / normal_duration
                    if normal_duration
                    else 0.0
                ),
                "same_recording_early_negative_false_positive_rate": (
                    sweep[sweep_threshold]["same_recording_early_high_windows"]
                    / sweep[sweep_threshold]["same_recording_early_total_windows"]
                    if sweep[sweep_threshold]["same_recording_early_total_windows"]
                    else 0.0
                ),
            }
            for sweep_threshold in sweep_thresholds
        ],
        "fall_results": fall_results,
        "normal_results": normal_results,
        "shadow_only": True,
        "deployment_eligible": False,
        "interpretation_limit": (
            "DGUHA contains staged forward falls by young healthy subjects. "
            "Event timing is skeleton-derived and is not a clinical loss-of-balance label."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _score_recording(
    frames,
    endpoints: np.ndarray,
    radar_start: datetime,
    extractor: RadarTemporalFeatureExtractorV2,
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    use_relative_features_v3: bool = False,
    use_hybrid_features_v4: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    frame_epochs = np.asarray(
        [frame.timestamp.timestamp() for frame in frames], dtype=np.float64
    )
    values: list[np.ndarray] = []
    times: list[float] = []
    skipped = 0
    for end_seconds in endpoints:
        end_timestamp = radar_start + timedelta(seconds=float(end_seconds))
        left = bisect.bisect_left(
            frame_epochs,
            (end_timestamp - timedelta(seconds=extractor.history_seconds + 0.06)).timestamp(),
        )
        right = bisect.bisect_right(
            frame_epochs, (end_timestamp + timedelta(seconds=0.06)).timestamp()
        )
        if left >= right:
            skipped += 1
            continue
        window = extractor.transform(frames[left:right], end_timestamp=end_timestamp)
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            skipped += 1
            continue
        values.append(np.asarray(window.values, dtype=np.float32))
        times.append(float(end_seconds))
    if not values:
        return np.asarray([]), np.asarray([]), skipped
    feature_tensor = np.stack(values).astype(np.float32, copy=False)
    if use_relative_features_v3:
        feature_tensor = transform_v2_values_to_v3(feature_tensor)
    elif use_hybrid_features_v4:
        feature_tensor = transform_v2_values_to_v4(feature_tensor)
    normalized = (feature_tensor - mean[None, None, :]) / std[None, None, :]
    with torch.inference_mode():
        scores = torch.sigmoid(
            model(torch.from_numpy(normalized.astype(np.float32, copy=False)))
        ).numpy()
    return np.asarray(times, dtype=np.float64), scores.astype(np.float64), skipped


def _score_point_recording(
    frames,
    endpoints: np.ndarray,
    radar_start: datetime,
    builder: PointCloudSequenceBuilder,
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    frame_epochs = np.asarray(
        [frame.timestamp.timestamp() for frame in frames], dtype=np.float64
    )
    values: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    times: list[float] = []
    skipped = 0
    for end_seconds in endpoints:
        end_timestamp = radar_start + timedelta(seconds=float(end_seconds))
        left = bisect.bisect_left(
            frame_epochs,
            (end_timestamp - timedelta(seconds=builder.history_seconds + 0.08)).timestamp(),
        )
        right = bisect.bisect_right(
            frame_epochs, (end_timestamp + timedelta(seconds=0.08)).timestamp()
        )
        if left >= right:
            skipped += 1
            continue
        sequence = builder.transform(frames[left:right], end_timestamp=end_timestamp)
        if int(sequence.frame_mask.sum()) < max(2, builder.time_steps // 2):
            skipped += 1
            continue
        values.append(sequence.values)
        point_masks.append(sequence.point_mask)
        frame_masks.append(sequence.frame_mask)
        times.append(float(end_seconds))
    if not values:
        return np.asarray([]), np.asarray([]), skipped
    raw = np.stack(values).astype(np.float32, copy=False)
    point_mask = np.stack(point_masks).astype(np.bool_, copy=False)
    frame_mask = np.stack(frame_masks).astype(np.bool_, copy=False)
    if len(mean) == len(ADAPTATION_FEATURE_NAMES):
        raw5 = raw[..., :5].copy()
        raw5[..., [0, 1]] = raw5[..., [1, 0]]
        normalized = (raw5 - mean[None, None, None, :]) / std[None, None, None, :]
        # DGUHA ROS point exports have no calibrated SNR.
        normalized[..., 4] = 0.0
    else:
        normalized = (raw - mean[None, None, None, :]) / std[None, None, None, :]
        normalized[..., 4] = np.where(
            raw[..., 5] > 0.5, normalized[..., 4], 0.0
        )
        normalized[..., 5] = raw[..., 5]
    normalized *= point_mask[..., None]
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 256):
            logits = model(
                torch.from_numpy(normalized[start : start + 256]),
                torch.from_numpy(point_mask[start : start + 256]),
                torch.from_numpy(frame_mask[start : start + 256]),
            ).squeeze(-1)
            scores.append(torch.sigmoid(logits).numpy())
    return (
        np.asarray(times, dtype=np.float64),
        np.concatenate(scores).astype(np.float64, copy=False),
        skipped,
    )


def _confirmed_run_end_indices(
    high: np.ndarray,
    times: np.ndarray,
    confirmation_windows: int,
    step_seconds: float,
) -> list[int]:
    run_length = 0
    latched = False
    ends: list[int] = []
    previous_time: float | None = None
    for index, (is_high, timestamp) in enumerate(zip(high, times)):
        contiguous = (
            previous_time is not None
            and timestamp - previous_time <= step_seconds * 1.5 + 1e-9
        )
        if not is_high or (previous_time is not None and not contiguous):
            run_length = 0
            latched = False
        if is_high:
            run_length += 1
            if run_length >= confirmation_windows and not latched:
                ends.append(index)
                latched = True
        previous_time = float(timestamp)
    return ends


def _resolve_evaluation_threshold(
    checkpoint: dict[str, Any],
    override: float | None,
) -> float:
    if override is None and "decision_threshold" not in checkpoint:
        raise ValueError("this checkpoint requires an externally locked evaluation threshold")
    threshold = float(checkpoint["decision_threshold"] if override is None else override)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("evaluation threshold must be finite and between 0 and 1")
    return threshold


def _confirmed_active_mask(
    high: np.ndarray,
    times: np.ndarray,
    confirmation_windows: int,
    step_seconds: float,
) -> np.ndarray:
    active = np.zeros(len(high), dtype=np.bool_)
    run_length = 0
    confirmed = False
    previous_time: float | None = None
    for index, (is_high, timestamp) in enumerate(zip(high, times)):
        contiguous = (
            previous_time is not None
            and timestamp - previous_time <= step_seconds * 1.5 + 1e-9
        )
        if not is_high or (previous_time is not None and not contiguous):
            run_length = 0
            confirmed = False
        if is_high:
            run_length += 1
            if run_length >= confirmation_windows:
                confirmed = True
            if confirmed:
                active[index] = True
        previous_time = float(timestamp)
    return active


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") == FORMAL_MODEL_VERSION:
        if checkpoint.get("model_role") != "pointnet_gru_short_horizon_radar_evidence":
            raise ValueError("checkpoint is not formal PointNet-GRU radar evidence")
        if checkpoint.get("sequence_version") != ADAPTATION_SEQUENCE_VERSION:
            raise ValueError("formal PointNet sequence version is incompatible")
        if tuple(checkpoint.get("feature_names", ())) != ADAPTATION_FEATURE_NAMES:
            raise ValueError("formal PointNet feature names/order are incompatible")
        if int(checkpoint.get("input_size", -1)) != len(ADAPTATION_FEATURE_NAMES):
            raise ValueError("formal PointNet input size is incompatible")
        if int(checkpoint.get("time_steps", -1)) != PointCloudSequenceBuilder().time_steps:
            raise ValueError("formal PointNet time steps are incompatible")
        if int(checkpoint.get("max_points", -1)) != PointCloudSequenceBuilder().max_points:
            raise ValueError("formal PointNet maximum points are incompatible")
        if bool(checkpoint.get("iwr_fall_recordings_used_as_prediction_positive", True)):
            raise ValueError("formal PointNet violated the IWR label constraint")
        if bool(checkpoint.get("deployment_eligible", True)):
            raise ValueError("evaluation requires a non-deployable checkpoint")
        return checkpoint
    if checkpoint.get("model_version") == ADAPTATION_PREDICTION_VERSION:
        if checkpoint.get("model_role") != "weak_supervision_prefall_prediction_research":
            raise ValueError("checkpoint is not PointNet adaptation research")
        if checkpoint.get("sequence_version") != ADAPTATION_SEQUENCE_VERSION:
            raise ValueError("adaptation sequence version is incompatible")
        if tuple(checkpoint.get("feature_names", ())) != ADAPTATION_FEATURE_NAMES:
            raise ValueError("adaptation feature names/order are incompatible")
        if int(checkpoint.get("input_size", -1)) != len(ADAPTATION_FEATURE_NAMES):
            raise ValueError("adaptation input size is incompatible")
        if int(checkpoint.get("time_steps", -1)) != PointCloudSequenceBuilder().time_steps:
            raise ValueError("adaptation time steps are incompatible")
        if int(checkpoint.get("max_points", -1)) != PointCloudSequenceBuilder().max_points:
            raise ValueError("adaptation maximum points are incompatible")
        if bool(checkpoint.get("deployment_eligible", True)):
            raise ValueError("evaluation requires a non-deployable checkpoint")
        return checkpoint
    if checkpoint.get("model_version") == POINT_PREDICTION_MODEL_VERSION:
        if checkpoint.get("model_role") != "weak_supervision_prefall_prediction":
            raise ValueError("checkpoint is not point-cloud pre-fall research")
        if checkpoint.get("sequence_version") != POINT_SEQUENCE_VERSION:
            raise ValueError("point checkpoint sequence version is incompatible")
        if tuple(checkpoint.get("feature_names", ())) != POINT_FEATURE_NAMES:
            raise ValueError("point checkpoint feature names/order are incompatible")
        if int(checkpoint.get("time_steps", -1)) != PointCloudSequenceBuilder().time_steps:
            raise ValueError("point checkpoint time steps are incompatible")
        if int(checkpoint.get("max_points", -1)) != PointCloudSequenceBuilder().max_points:
            raise ValueError("point checkpoint maximum points are incompatible")
        if bool(checkpoint.get("deployment_eligible", True)):
            raise ValueError("evaluation requires a non-deployable checkpoint")
        return checkpoint
    if checkpoint.get("model_version") not in {
        RESEARCH_MODEL_VERSION,
        EXPERIMENT_MODEL_VERSION,
        MULTITASK_MODEL_VERSION,
        MULTIHORIZON_MODEL_VERSION,
    }:
        raise ValueError("unsupported research checkpoint")
    if checkpoint.get("model_mode") != RESEARCH_MODEL_MODE:
        raise ValueError("checkpoint is not weak-supervision research")
    feature_version = checkpoint.get("feature_version")
    expected_names = {
        FEATURE_VERSION_V2: FEATURE_NAMES_V2,
        FEATURE_VERSION_V3: FEATURE_NAMES_V3,
        FEATURE_VERSION_V4: FEATURE_NAMES_V4,
    }.get(feature_version)
    if expected_names is None:
        raise ValueError("checkpoint feature version is incompatible")
    if tuple(checkpoint.get("feature_names", ())) != expected_names:
        raise ValueError("checkpoint feature names/order are incompatible")
    if int(checkpoint.get("window_size", -1)) != WINDOW_SIZE_V2:
        raise ValueError("checkpoint window size is incompatible")
    if bool(checkpoint.get("deployment_eligible", True)):
        raise ValueError("evaluation requires a non-deployable checkpoint")
    return checkpoint


def _build_prefall_model(checkpoint: dict[str, Any]) -> torch.nn.Module:
    version = str(checkpoint["model_version"])
    if version in {
        POINT_PREDICTION_MODEL_VERSION,
        ADAPTATION_PREDICTION_VERSION,
        FORMAL_MODEL_VERSION,
    }:
        input_size = (
            len(ADAPTATION_FEATURE_NAMES)
            if version in {ADAPTATION_PREDICTION_VERSION, FORMAL_MODEL_VERSION}
            else len(POINT_FEATURE_NAMES)
        )
        encoder = PointTemporalEncoder(
            input_size=input_size,
            frame_hidden_size=int(checkpoint["frame_hidden_size"]),
            temporal_hidden_size=int(checkpoint["temporal_hidden_size"]),
        )
        point_model = PointTemporalPredictionHead(encoder, horizon_count=1)
        point_model.load_state_dict(checkpoint["state_dict"], strict=True)
        point_model.eval()
        return point_model
    hidden_size = int(checkpoint["hidden_size"])
    input_size = len(tuple(checkpoint["feature_names"]))
    if version == RESEARCH_MODEL_VERSION:
        model: torch.nn.Module = RadarLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
        )
    elif version == EXPERIMENT_MODEL_VERSION:
        model = TemporalBinaryModel(
            architecture=str(checkpoint["model_architecture"]),
            input_size=input_size,
            hidden_size=hidden_size,
        )
    elif version == MULTITASK_MODEL_VERSION:
        model = SharedMultiTaskTemporalModel(
            architecture=str(checkpoint["model_architecture"]),
            input_size=input_size,
            hidden_size=hidden_size,
            action_class_count=int(checkpoint["action_class_count"]),
        )
    else:
        model = MultiHorizonTemporalModel(
            architecture=str(checkpoint["model_architecture"]),
            input_size=input_size,
            hidden_size=hidden_size,
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _subject_id(file_name: str) -> str:
    match = re.fullmatch(r"([FM]_\d{3})_A\d+_\d+\.txt", file_name)
    if match is None or match.group(1) not in DGUHA_SPLIT_BY_SUBJECT:
        raise ValueError(f"unexpected DGUHA file name: {file_name}")
    return match.group(1)


def _describe(values) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(array),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DGUHA recordings event-wise.")
    parser.add_argument("--data-directory", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--confirmation-windows", type=int, default=3)
    parser.add_argument(
        "--evaluation-anchor",
        choices=("descent_onset", "near_floor_level_reached"),
        default="descent_onset",
    )
    parser.add_argument("--minimum-lead-seconds", type=float, default=0.1)
    parser.add_argument("--maximum-lead-seconds", type=float, default=0.6)
    parser.add_argument("--minimum-pre-descent-margin-seconds", type=float, default=0.1)
    parser.add_argument("--decision-threshold-override", type=float)
    args = parser.parse_args()
    result = evaluate_dguha_events(
        args.data_directory,
        args.events,
        args.checkpoint,
        args.report,
        split=args.split,
        confirmation_windows=args.confirmation_windows,
        evaluation_anchor=args.evaluation_anchor,
        minimum_lead_seconds=args.minimum_lead_seconds,
        maximum_lead_seconds=args.maximum_lead_seconds,
        minimum_pre_descent_margin_seconds=(
            args.minimum_pre_descent_margin_seconds
        ),
        decision_threshold_override=args.decision_threshold_override,
    )
    summary_keys = (
        "split",
        "threshold",
        "eligible_fall_recording_count",
        "prediction_corridor_detected_event_count",
        "prediction_corridor_event_recall",
        "any_pre_onset_detected_event_count",
        "any_pre_onset_event_recall",
        "corridor_confirmation_lead_seconds",
        "normal_recording_count",
        "normal_duration_seconds",
        "normal_confirmed_run_count",
        "normal_confirmed_runs_per_hour",
        "normal_above_threshold_window_fraction",
        "normal_confirmed_active_window_fraction",
        "normal_confirmed_active_seconds_per_hour",
        "normal_recordings_with_confirmed_run",
        "normal_score_distribution",
        "prediction_corridor_score_distribution",
    )
    print(
        json.dumps(
            {key: result[key] for key in summary_keys},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
