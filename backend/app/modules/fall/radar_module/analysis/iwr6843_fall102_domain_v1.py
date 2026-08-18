from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from radar_module.acquisition.ti_reader import JsonlReplayAdapter, TiRadarReader
from radar_module.contracts import RadarFrame, Room
from radar_module.dataset.iwr6843_fall_v1 import parse_iwr6843_fall_csv
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


ANALYSIS_VERSION = "iwr6843_fall102_domain_v1"
CONTINUOUS_FEATURE_COUNT = 16


def analyze_iwr6843_fall102_domain(
    *,
    dguha_dataset_path: str | Path,
    iwr_dataset_path: str | Path,
    checkpoint_path: str | Path,
    iwr_gathered_data_path: str | Path,
    dguha_raw_path: str | Path,
    output_dir: str | Path,
    live_replay_path: str | Path | None = None,
    dguha_raw_sample_files: int = 84,
) -> dict[str, Any]:
    dguha_path = Path(dguha_dataset_path).resolve()
    iwr_path = Path(iwr_dataset_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    iwr_raw_root = _resolve_iwr_root(Path(iwr_gathered_data_path).resolve())
    dguha_raw_root = Path(dguha_raw_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    dguha = _load_dataset(dguha_path)
    iwr = _load_dataset(iwr_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    _validate_contract(dguha, iwr, checkpoint)

    dguha_features = np.asarray(dguha["features"], dtype=np.float32)
    dguha_labels = np.asarray(dguha["labels"], dtype=np.int8)
    dguha_split = np.asarray(dguha["split"]).astype(str)
    iwr_features = np.asarray(iwr["features"], dtype=np.float32)
    iwr_labels = np.asarray(iwr["labels"], dtype=np.int8)
    iwr_actions = np.asarray(iwr["action"]).astype(str)

    train_mask = dguha_split == "train"
    validation_mask = dguha_split == "validation"
    reference = dguha_features[train_mask]
    checkpoint_mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    checkpoint_std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    model = _load_model(checkpoint)

    groups = {
        "dguha_train_all": reference,
        "dguha_train_prefall_positive": dguha_features[train_mask & (dguha_labels == 1)],
        "dguha_train_negative": dguha_features[train_mask & (dguha_labels == 0)],
        "dguha_validation_prefall_positive": dguha_features[
            validation_mask & (dguha_labels == 1)
        ],
        "dguha_validation_negative": dguha_features[
            validation_mask & (dguha_labels == 0)
        ],
        "iwr6843_fall102_fall_sequence": iwr_features[iwr_labels == 1],
        "iwr6843_fall102_nonfall": iwr_features[iwr_labels == 0],
    }
    feature_rows = _feature_rows(groups, checkpoint_mean, checkpoint_std)
    _write_csv(destination / "feature_summary.csv", feature_rows)

    dguha_point_rows = _sample_dguha_raw_points(
        dguha_raw_root, maximum_files=dguha_raw_sample_files
    )
    iwr_point_rows = _all_iwr_raw_points(iwr_raw_root)
    raw_rows = _raw_point_summary_rows(dguha_point_rows + iwr_point_rows)
    _write_csv(destination / "raw_point_summary.csv", raw_rows)

    native_scores = {
        name: _scores(model, values, checkpoint_mean, checkpoint_std)
        for name, values in groups.items()
    }
    score_summary = {
        name: _describe(values) for name, values in native_scores.items()
    }
    score_summary["iwr6843_fall102_recording_auroc"] = _auroc(
        iwr_labels, _scores(model, iwr_features, checkpoint_mean, checkpoint_std)
    )

    live_replay_summary = None
    if live_replay_path is not None:
        live_replay = Path(live_replay_path).resolve()
        live_features, live_quality_counts = _load_live_replay_features(live_replay)
        live_native_scores = _scores(
            model, live_features, checkpoint_mean, checkpoint_std
        )
        live_counterfactuals, live_parameters = _counterfactual_features(
            live_features, reference
        )
        live_replay_summary = {
            "file": _display_path(live_replay),
            "sha256": _sha256(live_replay),
            "valid_window_count": int(len(live_features)),
            "quality_counts": live_quality_counts,
            "native_score": _describe(live_native_scores),
            "counterfactual_parameters": live_parameters,
            "counterfactual_scores": {
                name: _describe(
                    _scores(model, values, checkpoint_mean, checkpoint_std)
                )
                for name, values in live_counterfactuals.items()
            },
            "normalized_fraction_above_5sigma": float(
                np.mean(
                    np.abs(
                        (live_features - checkpoint_mean[None, None, :])
                        / checkpoint_std[None, None, :]
                    )
                    > 5.0
                )
            ),
            "distance_from_dguha_training": _distribution_distance(
                reference, live_features, checkpoint_std
            ),
            "distance_from_fall102": _distribution_distance(
                iwr_features, live_features, checkpoint_std
            ),
            "primary_feature_window_means": {
                FEATURE_NAMES_V2[index]: float(np.mean(live_features[..., index]))
                for index in (0, 4, 5, 6, 7, 8, 9, 10, 13)
            },
        }

    counterfactuals, transform_parameters = _counterfactual_features(
        iwr_features, reference
    )
    counterfactual_rows: list[dict[str, Any]] = []
    counterfactual_summary: dict[str, Any] = {}
    for name, values in counterfactuals.items():
        scores = _scores(model, values, checkpoint_mean, checkpoint_std)
        fall_scores = scores[iwr_labels == 1]
        nonfall_scores = scores[iwr_labels == 0]
        summary = {
            "all": _describe(scores),
            "fall": _describe(fall_scores),
            "nonfall": _describe(nonfall_scores),
            "recording_auroc": _auroc(iwr_labels, scores),
            "fall_above_threshold_fraction": float(
                np.mean(fall_scores >= float(checkpoint["decision_threshold"]))
            ),
            "nonfall_above_threshold_fraction": float(
                np.mean(nonfall_scores >= float(checkpoint["decision_threshold"]))
            ),
            "normalized_abs_median": float(
                np.median(
                    np.abs(
                        (values - checkpoint_mean[None, None, :])
                        / checkpoint_std[None, None, :]
                    )
                )
            ),
            "normalized_fraction_above_5sigma": float(
                np.mean(
                    np.abs(
                        (values - checkpoint_mean[None, None, :])
                        / checkpoint_std[None, None, :]
                    )
                    > 5.0
                )
            ),
        }
        counterfactual_summary[name] = summary
        counterfactual_rows.append(
            {
                "transform": name,
                "fall_median": summary["fall"]["median"],
                "fall_p95": summary["fall"]["p95"],
                "fall_max": summary["fall"]["max"],
                "nonfall_median": summary["nonfall"]["median"],
                "nonfall_p95": summary["nonfall"]["p95"],
                "nonfall_max": summary["nonfall"]["max"],
                "recording_auroc": summary["recording_auroc"],
                "fall_above_threshold_fraction": summary[
                    "fall_above_threshold_fraction"
                ],
                "nonfall_above_threshold_fraction": summary[
                    "nonfall_above_threshold_fraction"
                ],
                "normalized_abs_median": summary["normalized_abs_median"],
                "normalized_fraction_above_5sigma": summary[
                    "normalized_fraction_above_5sigma"
                ],
            }
        )
    _write_csv(destination / "counterfactual_scores.csv", counterfactual_rows)

    per_action = {}
    all_native_iwr_scores = _scores(
        model, iwr_features, checkpoint_mean, checkpoint_std
    )
    for action in sorted(set(iwr_actions)):
        per_action[action] = _describe(all_native_iwr_scores[iwr_actions == action])

    distribution_distance = _distribution_distance(
        reference, iwr_features, checkpoint_std
    )
    _plot_feature_shift(feature_rows, destination / "feature_shift.png")
    _plot_counterfactual_scores(
        counterfactuals,
        iwr_labels,
        model,
        checkpoint_mean,
        checkpoint_std,
        float(checkpoint["decision_threshold"]),
        destination / "counterfactual_scores.png",
    )

    decision = _make_decision(counterfactual_summary)
    report: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "scope": {
            "training_performed": False,
            "checkpoint_modified": False,
            "model_modified": False,
            "live_inference_modified": False,
            "fall102_label_scope": "recording-level fall sequence; no onset or impact time",
        },
        "sources": {
            "dguha_dataset": _display_path(dguha_path),
            "dguha_dataset_sha256": _sha256(dguha_path),
            "iwr6843_fall102_dataset": _display_path(iwr_path),
            "iwr6843_fall102_dataset_sha256": _sha256(iwr_path),
            "checkpoint": _display_path(checkpoint_file),
            "checkpoint_sha256": _sha256(checkpoint_file),
        },
        "contract": {
            "feature_version": str(dguha["feature_version"].item()),
            "feature_names": list(FEATURE_NAMES_V2),
            "window_shape": [20, len(FEATURE_NAMES_V2)],
            "threshold": float(checkpoint["decision_threshold"]),
            "model_uses_raw_x_or_y": False,
            "model_uses_z_and_euclidean_range": True,
        },
        "sample_counts": {name: int(len(values)) for name, values in groups.items()},
        "raw_point_audit": {
            "dguha_sampled_file_count": len(
                {row["file"] for row in dguha_point_rows}
            ),
            "iwr6843_file_count": len({row["file"] for row in iwr_point_rows}),
            "summary_file": _display_path(destination / "raw_point_summary.csv"),
        },
        "native_score_summary": score_summary,
        "native_score_by_iwr_action": per_action,
        "distribution_distance": distribution_distance,
        "counterfactual_parameters": transform_parameters,
        "counterfactual_score_summary": counterfactual_summary,
        "live_replay_diagnostic": live_replay_summary,
        "decision": decision,
        "limitations": [
            "Fall-102 has no descent-onset or impact timestamps, so it cannot measure lead time.",
            "Fall-102 terminal windows mix pre-fall, descent and post-impact states.",
            "DGUHA raw-coordinate statistics are a deterministic sample; the exact model-input comparison uses all processed windows.",
            "Domain-alignment counterfactuals are diagnostics only and are not deployable preprocessing.",
        ],
    }
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "ANALYSIS_REPORT.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return report


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        return {name: np.asarray(dataset[name]) for name in dataset.files}


def _validate_contract(
    dguha: dict[str, np.ndarray],
    iwr: dict[str, np.ndarray],
    checkpoint: dict[str, Any],
) -> None:
    expected = tuple(FEATURE_NAMES_V2)
    for name, dataset in (("DGUHA", dguha), ("Fall-102", iwr)):
        if tuple(str(value) for value in dataset["feature_names"]) != expected:
            raise ValueError(f"{name} feature order is incompatible")
        if np.asarray(dataset["features"]).shape[1:] != (20, len(expected)):
            raise ValueError(f"{name} feature shape is incompatible")
    if tuple(checkpoint["feature_names"]) != expected:
        raise ValueError("checkpoint feature order is incompatible")


def _load_model(checkpoint: dict[str, Any]) -> TemporalBinaryModel:
    model = TemporalBinaryModel(
        architecture=str(checkpoint["model_architecture"]),
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _scores(
    model: TemporalBinaryModel,
    raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = ((raw - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 1024):
            logits = model(torch.from_numpy(normalized[start : start + 1024]))
            chunks.append(torch.sigmoid(logits).numpy().astype(np.float64))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def _counterfactual_features(
    iwr: np.ndarray, reference: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    native = np.asarray(iwr, dtype=np.float32)
    reference_flat = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    iwr_flat = native.reshape(-1, native.shape[-1]).astype(np.float64)

    z_scale = _safe_scale(iwr_flat[:, 0], reference_flat[:, 0])
    z_offset = float(np.mean(reference_flat[:, 0]) - z_scale * np.mean(iwr_flat[:, 0]))
    velocity_scale = _magnitude_scale(
        iwr_flat[:, [6, 7, 8]], reference_flat[:, [6, 7, 8]]
    )
    point_count_scale = _positive_median_scale(iwr_flat[:, 5], reference_flat[:, 5])
    range_scale = _positive_median_scale(iwr_flat[:, 9], reference_flat[:, 9])

    z_aligned = _apply_z_affine(native, z_scale, z_offset)
    z_flipped = _flip_z(native)
    z_flipped_scale = _safe_scale(
        z_flipped[..., 0].reshape(-1), reference_flat[:, 0]
    )
    z_flipped_offset = float(
        np.mean(reference_flat[:, 0])
        - z_flipped_scale * np.mean(z_flipped[..., 0])
    )
    z_flipped_aligned = _apply_z_affine(
        z_flipped, z_flipped_scale, z_flipped_offset
    )
    velocity_aligned = _apply_group_scales(
        native, velocity_scale=velocity_scale
    )
    count_aligned = _apply_group_scales(
        native, point_count_scale=point_count_scale
    )
    physical_aligned = _apply_group_scales(
        z_aligned,
        velocity_scale=velocity_scale,
        point_count_scale=point_count_scale,
        range_scale=range_scale,
    )

    featurewise = native.astype(np.float64, copy=True)
    for index in range(CONTINUOUS_FEATURE_COUNT):
        source_values = iwr_flat[:, index]
        target_values = reference_flat[:, index]
        scale = _safe_scale(source_values, target_values)
        offset = float(np.mean(target_values) - scale * np.mean(source_values))
        featurewise[..., index] = featurewise[..., index] * scale + offset

    return (
        {
            "native": native,
            "z_sign_flip": z_flipped,
            "z_scale_offset": z_aligned,
            "z_sign_flip_then_scale_offset": z_flipped_aligned,
            "velocity_scale_only": velocity_aligned,
            "point_count_scale_only": count_aligned,
            "physical_group_alignment": physical_aligned,
            "featurewise_affine_upper_bound": featurewise.astype(np.float32),
        },
        {
            "z_scale": z_scale,
            "z_offset_m": z_offset,
            "z_flipped_scale": z_flipped_scale,
            "z_flipped_offset_m": z_flipped_offset,
            "velocity_scale": velocity_scale,
            "point_count_scale": point_count_scale,
            "range_scale": range_scale,
            "alignment_reference": "all DGUHA training windows; Fall-102 labels not used",
        },
    )


def _apply_z_affine(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    result[..., 0:4] = result[..., 0:4] * scale + offset
    result[..., 4] *= abs(scale)
    result[..., 11:15] *= scale
    result[..., 15] *= abs(scale)
    return result.astype(np.float32)


def _flip_z(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    old_p10 = values[..., 1].astype(np.float64)
    old_p90 = values[..., 3].astype(np.float64)
    result[..., 0] *= -1.0
    result[..., 1] = -old_p90
    result[..., 2] *= -1.0
    result[..., 3] = -old_p10
    result[..., 11:15] *= -1.0
    return result.astype(np.float32)


def _apply_group_scales(
    values: np.ndarray,
    *,
    velocity_scale: float = 1.0,
    point_count_scale: float = 1.0,
    range_scale: float = 1.0,
) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    result[..., 5] *= point_count_scale
    result[..., 6:9] *= velocity_scale
    result[..., 9:11] *= range_scale
    return result.astype(np.float32)


def _safe_scale(source: np.ndarray, target: np.ndarray) -> float:
    source_std = float(np.std(source))
    target_std = float(np.std(target))
    return target_std / source_std if source_std > 1e-9 else 1.0


def _magnitude_scale(source: np.ndarray, target: np.ndarray) -> float:
    source_value = float(np.quantile(np.abs(source), 0.75))
    target_value = float(np.quantile(np.abs(target), 0.75))
    return target_value / source_value if source_value > 1e-9 else 1.0


def _positive_median_scale(source: np.ndarray, target: np.ndarray) -> float:
    source_positive = source[source > 0]
    target_positive = target[target > 0]
    if not len(source_positive) or not len(target_positive):
        return 1.0
    denominator = float(np.median(source_positive))
    return float(np.median(target_positive)) / denominator if denominator > 1e-9 else 1.0


def _feature_rows(
    groups: dict[str, np.ndarray], checkpoint_mean: np.ndarray, checkpoint_std: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, values in groups.items():
        flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
        normalized = (flat - checkpoint_mean[None, :]) / checkpoint_std[None, :]
        for index, feature in enumerate(FEATURE_NAMES_V2):
            column = flat[:, index]
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "count": int(len(column)),
                    "mean": float(np.mean(column)),
                    "std": float(np.std(column)),
                    "p05": float(np.quantile(column, 0.05)),
                    "median": float(np.median(column)),
                    "p95": float(np.quantile(column, 0.95)),
                    "checkpoint_z_mean": float(np.mean(normalized[:, index])),
                    "checkpoint_z_abs_median": float(
                        np.median(np.abs(normalized[:, index]))
                    ),
                    "fraction_above_5sigma": float(
                        np.mean(np.abs(normalized[:, index]) > 5.0)
                    ),
                }
            )
    return rows


def _distribution_distance(
    reference: np.ndarray, iwr: np.ndarray, checkpoint_std: np.ndarray
) -> dict[str, Any]:
    source = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    target = iwr.reshape(-1, iwr.shape[-1]).astype(np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    standardized = np.abs(target_mean - source_mean) / checkpoint_std
    order = np.argsort(standardized)[::-1]
    return {
        "mean_absolute_standardized_mean_shift": float(np.mean(standardized)),
        "continuous_feature_mean_absolute_shift": float(
            np.mean(standardized[:CONTINUOUS_FEATURE_COUNT])
        ),
        "top_shifted_features": [
            {"feature": FEATURE_NAMES_V2[index], "absolute_shift_sigma": float(standardized[index])}
            for index in order[:10]
        ],
    }


def _all_iwr_raw_points(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.csv")):
        frames, _ = parse_iwr6843_fall_csv(path)
        label = "fall" if path.parent.name == "Fall" else "nonfall"
        for frame in frames:
            for point in frame.points:
                rows.append(
                    {
                        "dataset": "iwr6843_fall102",
                        "class": label,
                        "file": path.as_posix(),
                        "frame": frame.timestamp.isoformat(),
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                        "velocity": point.velocity,
                    }
                )
    return rows


def _load_live_replay_features(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    adapter = JsonlReplayAdapter(path, speed=100_000.0, loop=False)
    reader = TiRadarReader(
        adapter, device_id="fall102-domain-live-replay", room=Room.BATHROOM
    )
    extractor = RadarTemporalFeatureExtractorV2()
    frames: deque[RadarFrame] = deque()
    windows: list[np.ndarray] = []
    qualities: Counter[str] = Counter()
    last_evaluation: datetime | None = None
    reader.start()
    try:
        while not adapter.finished:
            frame = reader.read()
            if frame is None:
                continue
            frames.append(frame)
            while frames and (
                frame.timestamp - frames[0].timestamp
            ).total_seconds() > 2.2:
                frames.popleft()
            if last_evaluation is not None and (
                frame.timestamp - last_evaluation
            ).total_seconds() < 0.095:
                continue
            last_evaluation = frame.timestamp
            if (frame.timestamp - frames[0].timestamp).total_seconds() < 1.9:
                qualities["WARMUP"] += 1
                continue
            window = extractor.transform(tuple(frames), end_timestamp=frame.timestamp)
            qualities[window.data_quality.value] += 1
            if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
                continue
            windows.append(np.asarray(window.values, dtype=np.float32))
    finally:
        reader.stop()
    if not windows:
        raise ValueError("live replay produced no valid feature windows")
    return np.stack(windows), dict(qualities)


def _sample_dguha_raw_points(root: Path, *, maximum_files: int) -> list[dict[str, Any]]:
    files = sorted(path for path in root.rglob("*.txt") if path.parent.name == "radar")
    if len(files) > maximum_files:
        indices = np.linspace(0, len(files) - 1, maximum_files, dtype=np.int64)
        files = [files[int(index)] for index in indices]
    rows: list[dict[str, Any]] = []
    for path in files:
        frames = parse_radhar_text(path)
        action = path.parent.parent.name
        label = "fall" if action == "5_falling_forward" else "nonfall"
        for frame in frames:
            for point in frame.points:
                rows.append(
                    {
                        "dataset": "dguha_raw_sample",
                        "class": label,
                        "file": path.as_posix(),
                        "frame": frame.timestamp.isoformat(),
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                        "velocity": point.velocity,
                    }
                )
    return rows


def _raw_point_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["class"]) for row in rows})
    for dataset, label in keys:
        selected = [row for row in rows if row["dataset"] == dataset and row["class"] == label]
        for field in ("x", "y", "z", "velocity"):
            values = np.asarray([row[field] for row in selected], dtype=np.float64)
            output.append(
                {
                    "dataset": dataset,
                    "class": label,
                    "metric": field,
                    **_describe(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            )
        frame_counts: dict[tuple[str, str], int] = {}
        for row in selected:
            key = (row["file"], row["frame"])
            frame_counts[key] = frame_counts.get(key, 0) + 1
        output.append(
            {
                "dataset": dataset,
                "class": label,
                "metric": "point_count_per_frame",
                **_describe(np.asarray(list(frame_counts.values()), dtype=np.float64)),
                "mean": float(np.mean(list(frame_counts.values()))),
                "std": float(np.std(list(frame_counts.values()))),
            }
        )
    return output


def _resolve_iwr_root(path: Path) -> Path:
    candidate = path / "GatheredData"
    root = candidate if candidate.is_dir() else path
    if not (root / "Fall").is_dir() or not (root / "Not").is_dir():
        raise FileNotFoundError("Fall-102 GatheredData/Fall and Not were not found")
    return root


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _describe(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _plot_feature_shift(rows: list[dict[str, Any]], path: Path) -> None:
    groups = ("dguha_train_all", "iwr6843_fall102_fall_sequence", "iwr6843_fall102_nonfall")
    lookup = {(row["group"], row["feature"]): row for row in rows}
    features = FEATURE_NAMES_V2[:CONTINUOUS_FEATURE_COUNT]
    x = np.arange(len(features))
    width = 0.25
    fig, axis = plt.subplots(figsize=(14, 5.5))
    for offset, group in zip((-width, 0.0, width), groups):
        values = [lookup[(group, feature)]["checkpoint_z_mean"] for feature in features]
        axis.bar(x + offset, values, width=width, label=group)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(features, rotation=55, ha="right")
    axis.set_ylabel("Mean in B0 checkpoint-standardized units")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_counterfactual_scores(
    counterfactuals: dict[str, np.ndarray],
    labels: np.ndarray,
    model: TemporalBinaryModel,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    names = list(counterfactuals)
    fall = []
    nonfall = []
    for name in names:
        scores = _scores(model, counterfactuals[name], mean, std)
        fall.append(float(np.median(scores[labels == 1])))
        nonfall.append(float(np.median(scores[labels == 0])))
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(x, fall, "o-", label="Fall-102 fall sequence median")
    axis.plot(x, nonfall, "o-", label="Fall-102 nonfall median")
    axis.axhline(threshold, color="red", linestyle="--", label=f"threshold={threshold:g}")
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.set_ylabel("B0 score (symlog)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_decision(counterfactual: dict[str, Any]) -> dict[str, str]:
    native = counterfactual["native"]
    physical = counterfactual["physical_group_alignment"]
    upper = counterfactual["featurewise_affine_upper_bound"]
    best_physical_median = max(
        float(counterfactual[name]["fall"]["median"])
        for name in (
            "z_sign_flip",
            "z_scale_offset",
            "z_sign_flip_then_scale_offset",
            "velocity_scale_only",
            "point_count_scale_only",
            "physical_group_alignment",
        )
    )
    best_ranking = max(
        float(summary["recording_auroc"])
        for summary in counterfactual.values()
    )
    native_median = float(native["fall"]["median"])
    if best_physical_median > native_median * 1.5 or best_ranking > float(
        native["recording_auroc"]
    ) + 0.1:
        primary = "coordinate_and_scale_contribute_but_do_not_explain_score_collapse_alone"
    else:
        primary = "coordinate_and_marginal_scale_are_not_major_score_collapse_factors"
    return {
        "primary": primary,
        "interpretation": (
            "Use Fall-102 only as an action-sequence/domain diagnostic. Retaining DGUHA as the "
            "public pre-fall source remains scientifically possible, but same hardware does not "
            "establish input compatibility. The remaining gap includes joint temporal-pattern and "
            "label-semantic mismatch; Fall-102 cannot validate advance prediction."
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    native = report["counterfactual_score_summary"]["native"]
    lines = [
        "# IWR6843 Fall-102 与 DGUHA 输入域分析",
        "",
        f"分析版本：`{report['analysis_version']}`。本轮未训练、未修改 checkpoint、模型或实时链路。",
        "",
        "## 直接结论",
        "",
        f"判断：`{report['decision']['primary']}`。",
        "",
        "坐标与尺度会影响排序和分值，但不足以单独解释低分；剩余差异主要指向联合时序模式与标签语义不一致。DGUHA仍可作为公开跌前弱监督来源保留，但不能因硬件相近就视为与Fall-102或当前实时链路同域。",
        "",
        "## 原始 B0 结果",
        "",
        f"- Fall-102 fall-sequence：median={native['fall']['median']:.6g}，p95={native['fall']['p95']:.6g}，max={native['fall']['max']:.6g}",
        f"- Fall-102 nonfall：median={native['nonfall']['median']:.6g}，p95={native['nonfall']['p95']:.6g}，max={native['nonfall']['max']:.6g}",
        f"- 动作级 AUROC：{native['recording_auroc']:.3f}",
        "",
        "## 无训练反事实",
        "",
        "| 变换 | fall median | fall max | nonfall median | AUROC | fall>=0.35 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["counterfactual_score_summary"].items():
        lines.append(
            f"| {name} | {summary['fall']['median']:.6g} | {summary['fall']['max']:.6g} | "
            f"{summary['nonfall']['median']:.6g} | {summary['recording_auroc']:.3f} | "
            f"{summary['fall_above_threshold_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "`featurewise_affine_upper_bound` 使用无标签的全域边缘均值/方差对齐，只是诊断上界，不能直接接入实时链路。",
            "",
            "## 标签限制",
            "",
            "Fall-102 每段只有 fall/nonfall 标签，没有下降起点或撞击时刻；终端2秒同时混入跌前、下降和落地后状态。因此它可以检验同硬件数据是否进入相似特征域，但不能验证0.5–1.0秒提前预测。",
            "",
        ]
    )
    live = report.get("live_replay_diagnostic")
    if live is not None:
        top_live_shifts = live["distance_from_dguha_training"][
            "top_shifted_features"
        ][:5]
        shift_text = "、".join(
            f"{item['feature']}={item['absolute_shift_sigma']:.2f}σ"
            for item in top_live_shifts
        )
        lines.extend(
            [
                "## 真实 IWR6843 回放核验",
                "",
                f"- 有效窗口：{live['valid_window_count']}",
                f"- 原始score：median={live['native_score']['median']:.6g}，p95={live['native_score']['p95']:.6g}，max={live['native_score']['max']:.6g}",
                f"- 超过5σ的特征占比：{live['normalized_fraction_above_5sigma']:.3%}",
                f"- 相对DGUHA均值偏移最大的特征：{shift_text}",
                "",
                "该回放的反事实结果记录在 `report.json`。它用于判断当前实时低分是否可由简单坐标/尺度变换恢复，不作为模型性能评估。",
                "",
            ]
        )
    lines.extend(
        [
            "## 产物",
            "",
            "- `feature_summary.csv`：完整20×19输入特征统计",
            "- `raw_point_summary.csv`：原始坐标、Doppler和点数统计",
            "- `counterfactual_scores.csv`：坐标/尺度反事实",
            "- `feature_shift.png`、`counterfactual_scores.png`：可视化",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Fall-102 and DGUHA input domains")
    parser.add_argument("--dguha-dataset", required=True, type=Path)
    parser.add_argument("--iwr-dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--iwr-raw", required=True, type=Path)
    parser.add_argument("--dguha-raw", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--live-replay", type=Path)
    parser.add_argument("--dguha-raw-sample-files", type=int, default=84)
    args = parser.parse_args()
    report = analyze_iwr6843_fall102_domain(
        dguha_dataset_path=args.dguha_dataset,
        iwr_dataset_path=args.iwr_dataset,
        checkpoint_path=args.checkpoint,
        iwr_gathered_data_path=args.iwr_raw,
        dguha_raw_path=args.dguha_raw,
        output_dir=args.output_dir,
        live_replay_path=args.live_replay,
        dguha_raw_sample_files=args.dguha_raw_sample_files,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
