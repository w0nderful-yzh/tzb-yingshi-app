from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from radar_module.analysis.dguha_motion_evolution_v1 import (
    _normal_motion_anchor,
    _recording_series,
    _subject_id,
)
from radar_module.dataset.dguha_research_v2 import (
    DGUHA_SPLIT_BY_SUBJECT,
    parse_dguha_kinect,
)
from radar_module.dataset.radhar_converter import parse_radhar_text
from radar_module.model.point_temporal import (
    PointTemporalEncoder,
    PointTemporalPredictionHead,
    PointTemporalPretrainingModel,
)
from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.pointcloud_sequence import PointCloudSequenceBuilder
from radar_module.preprocess.temporal_features_v2 import (
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


ANALYSIS_VERSION = "embedding_label_feasibility_v1"
TIME_GRID = np.round(np.arange(-3.0, 0.001, 0.1), 1)
DEVELOPMENT_SPLITS = frozenset(("train", "validation"))


@dataclass(slots=True)
class EncoderBundle:
    name: str
    family: str
    dimension: int
    payload: dict[str, Any]
    model: torch.nn.Module
    has_head: bool


@dataclass(slots=True)
class EncodedRows:
    metadata: pd.DataFrame
    embeddings: np.ndarray
    head_score: np.ndarray


def run_analysis(
    *,
    dguha_root: str | Path,
    events_file: str | Path,
    point_dataset: str | Path,
    tcn_dataset: str | Path,
    point_pretrain_checkpoint: str | Path,
    point_finetuned_checkpoint: str | Path,
    tcn_checkpoint: str | Path,
    mmfall_dataset: str | Path | None,
    output_directory: str | Path,
) -> dict[str, Any]:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    root = Path(dguha_root).resolve()
    event_path = Path(events_file).resolve()
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    events = json.loads(event_path.read_text(encoding="utf-8"))
    development_events = [
        event
        for event in events
        if bool(event["eligible_for_prediction_windows"])
        and str(event["project_split"]) in DEVELOPMENT_SPLITS
    ]
    heldout_events = [
        event
        for event in events
        if bool(event["eligible_for_prediction_windows"])
        and str(event["project_split"]) == "test"
    ]
    point_pretrained = _load_point_pretrained(point_pretrain_checkpoint)
    point_finetuned = _load_point_finetuned(point_finetuned_checkpoint)
    tcn = _load_tcn(tcn_checkpoint)
    bundles = (point_pretrained, point_finetuned, tcn)

    raw_metadata, point_values, point_masks, frame_masks, tcn_values = _build_raw_trajectories(
        root, development_events
    )
    raw_sets = {
        point_pretrained.name: _encode_point_rows(
            point_pretrained, raw_metadata, point_values, point_masks, frame_masks
        ),
        point_finetuned.name: _encode_point_rows(
            point_finetuned, raw_metadata, point_values, point_masks, frame_masks
        ),
        tcn.name: _encode_tcn_rows(tcn, raw_metadata, tcn_values),
    }

    normal_sets = {
        point_pretrained.name: _point_normal_reference(point_pretrained, point_dataset),
        point_finetuned.name: _point_normal_reference(point_finetuned, point_dataset),
        tcn.name: _tcn_normal_reference(tcn, tcn_dataset),
    }

    geometry_rows: list[pd.DataFrame] = []
    separability_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    geometry_contracts: dict[str, dict[str, np.ndarray]] = {}
    for bundle in bundles:
        geometry, separability, summary, contract = _analyze_encoder_geometry(
            bundle,
            raw_sets[bundle.name],
            normal_sets[bundle.name],
        )
        geometry_rows.append(geometry)
        separability_rows.extend(separability)
        trajectory_rows.append(summary)
        geometry_contracts[bundle.name] = contract

    geometry_table = pd.concat(geometry_rows, ignore_index=True)
    separability_table = pd.DataFrame(separability_rows)
    trajectory_table = pd.DataFrame(trajectory_rows)
    anchor_table, anchor_summary = _audit_dguha_anchors(root, development_events)
    other_dataset_table = _other_dataset_label_audit()

    mmfall_table = pd.DataFrame()
    mmfall_summary: dict[str, Any] = {"available": False}
    if mmfall_dataset is not None and Path(mmfall_dataset).is_file():
        mmfall_table, mmfall_summary = _analyze_mmfall(
            tcn,
            mmfall_dataset,
            geometry_contracts[tcn.name],
        )

    geometry_table.to_csv(destination / "embedding_geometry_timeseries.csv", index=False)
    separability_table.to_csv(destination / "embedding_separability_by_time.csv", index=False)
    trajectory_table.to_csv(destination / "embedding_trajectory_summary.csv", index=False)
    anchor_table.to_csv(destination / "dguha_anchor_sensitivity.csv", index=False)
    other_dataset_table.to_csv(destination / "other_dataset_label_audit.csv", index=False)
    if len(mmfall_table):
        mmfall_table.to_csv(destination / "mmfall_external_embedding_statistics.csv", index=False)
    _save_embedding_npz(destination, raw_sets)

    _plot_embedding_geometry(geometry_table, separability_table, destination)
    _plot_event_heatmaps(geometry_table, destination)
    _plot_anchor_sensitivity(anchor_table, destination)
    if len(mmfall_table):
        _plot_mmfall(mmfall_table, destination)

    verdict = _verdict(trajectory_table, separability_table, anchor_summary, mmfall_summary)
    report: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "development_policy": "train and validation only; test embeddings not extracted",
        "eligible_development_fall_events": len(development_events),
        "heldout_test_fall_events_not_analyzed": len(heldout_events),
        "time_grid_seconds_relative_to_current_descent_onset": [
            float(TIME_GRID[0]), float(TIME_GRID[-1]), 0.1
        ],
        "encoder_contracts": {
            bundle.name: {
                "family": bundle.family,
                "embedding_dimension": bundle.dimension,
                "checkpoint_model_version": str(bundle.payload.get("model_version")),
                "positive_anchor": bundle.payload.get("positive_anchor"),
                "prediction_horizon_seconds": bundle.payload.get("prediction_horizon_seconds"),
                "fall_prediction_head_available": bundle.has_head,
                "deployment_eligible": bool(bundle.payload.get("deployment_eligible", False)),
            }
            for bundle in bundles
        },
        "important_geometry_warning": (
            "Fine-tuned PointNet and TCN embeddings were already supervised by the current "
            "descent_onset labels. Their separability is not independent evidence that the "
            "underlying data contain a clinically meaningful pre-fall state."
        ),
        "point_pretrain_control": (
            "The activity-pretrained PointNet-GRU encoder has never seen fall-prediction labels. "
            "However, the reported target-projection axis is still defined from DGUHA train labels, "
            "so validation separation is a frozen-representation probe, not unsupervised discovery."
        ),
        "trajectory_summary": trajectory_table.to_dict(orient="records"),
        "dguha_anchor_summary": anchor_summary,
        "mmfall_external_summary": mmfall_summary,
        "verdict": verdict,
        "model_training_performed": False,
        "model_parameters_modified": False,
        "test_split_inspected": False,
        "deployment_eligible": False,
    }
    (destination / "embedding_label_feasibility_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return report


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint: {path}")
    return payload


def _load_point_pretrained(path: str | Path) -> EncoderBundle:
    payload = _load_checkpoint(path)
    model = PointTemporalPretrainingModel(
        class_count=len(payload["class_names"]),
        frame_hidden_size=int(payload["frame_hidden_size"]),
        temporal_hidden_size=int(payload["temporal_hidden_size"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return EncoderBundle(
        "pointnet_gru_activity_pretrained",
        "pointnet_gru",
        int(payload["temporal_hidden_size"]),
        payload,
        model.eval(),
        False,
    )


def _load_point_finetuned(path: str | Path) -> EncoderBundle:
    payload = _load_checkpoint(path)
    encoder = PointTemporalEncoder(
        frame_hidden_size=int(payload["frame_hidden_size"]),
        temporal_hidden_size=int(payload["temporal_hidden_size"]),
    )
    model = PointTemporalPredictionHead(encoder, horizon_count=1)
    model.load_state_dict(payload["state_dict"], strict=True)
    return EncoderBundle(
        "pointnet_gru_prefall_finetuned",
        "pointnet_gru",
        int(payload["temporal_hidden_size"]),
        payload,
        model.eval(),
        True,
    )


def _load_tcn(path: str | Path) -> EncoderBundle:
    payload = _load_checkpoint(path)
    model = TemporalBinaryModel(
        architecture=str(payload["model_architecture"]),
        input_size=int(payload["input_size"]),
        hidden_size=int(payload["hidden_size"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return EncoderBundle(
        "causal_tcn_prefall",
        "causal_tcn",
        int(payload["hidden_size"]),
        payload,
        model.eval(),
        True,
    )


def _build_raw_trajectories(
    root: Path, events: list[dict[str, Any]]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_builder = PointCloudSequenceBuilder()
    feature_extractor = RadarTemporalFeatureExtractorV2()
    metadata: list[dict[str, Any]] = []
    point_values: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    tcn_values: list[np.ndarray] = []

    records: list[tuple[Path, datetime, str, str, str]] = []
    for event in events:
        records.append(
            (
                root / Path(str(event["source_file"])),
                datetime.fromisoformat(str(event["descent_onset"])),
                str(event["project_split"]),
                str(event["subject_id"]),
                "fall",
            )
        )
    for radar_path in sorted(root.glob("*/3_Sit_down_and_stand_up/radar/*.txt")):
        subject = _subject_id(radar_path.name)
        split = DGUHA_SPLIT_BY_SUBJECT[subject]
        if split not in DEVELOPMENT_SPLITS:
            continue
        frames = parse_radhar_text(radar_path, device_id=f"embed-anchor-{radar_path.stem}")
        anchor_seconds = _normal_motion_anchor(
            _recording_series(frames), "3_Sit_down_and_stand_up"
        )
        records.append(
            (
                radar_path,
                frames[0].timestamp + timedelta(seconds=anchor_seconds),
                split,
                subject,
                "normal_sit",
            )
        )

    for record_index, (radar_path, anchor, split, subject, group) in enumerate(records, 1):
        frames = parse_radhar_text(radar_path, device_id=f"embed-{radar_path.stem}")
        relative_path = radar_path.relative_to(root).as_posix()
        for relative_seconds in TIME_GRID:
            end = anchor + timedelta(seconds=float(relative_seconds))
            if end < frames[0].timestamp or end > frames[-1].timestamp:
                continue
            try:
                point_sequence = point_builder.transform(frames, end_timestamp=end)
                temporal_window = feature_extractor.transform(frames, end_timestamp=end)
            except ValueError:
                continue
            if int(point_sequence.frame_mask.sum()) < point_builder.time_steps // 2:
                continue
            if temporal_window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
                continue
            metadata.append(
                {
                    "recording_id": relative_path,
                    "split": split,
                    "subject_id": subject,
                    "group": group,
                    "relative_seconds": float(relative_seconds),
                    "point_observed_frames": int(point_sequence.frame_mask.sum()),
                    "tcn_data_quality": temporal_window.data_quality.value,
                }
            )
            point_values.append(point_sequence.values)
            point_masks.append(point_sequence.point_mask)
            frame_masks.append(point_sequence.frame_mask)
            tcn_values.append(temporal_window.values)
        if record_index % 20 == 0 or record_index == len(records):
            print(f"trajectory extraction {record_index}/{len(records)}")
    return (
        pd.DataFrame(metadata),
        np.stack(point_values).astype(np.float32),
        np.stack(point_masks).astype(np.bool_),
        np.stack(frame_masks).astype(np.bool_),
        np.stack(tcn_values).astype(np.float32),
    )


def _normalize_points(raw: np.ndarray, mask: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    std = np.asarray(payload["normalization_std"], dtype=np.float32)
    result = (raw - mean[None, None, None, :]) / std[None, None, None, :]
    result[..., 4] = np.where(raw[..., 5] > 0.5, result[..., 4], 0.0)
    result[..., 5] = raw[..., 5]
    result *= mask[..., None]
    return result.astype(np.float32, copy=False)


def _encode_point_rows(
    bundle: EncoderBundle,
    metadata: pd.DataFrame,
    points: np.ndarray,
    point_mask: np.ndarray,
    frame_mask: np.ndarray,
    batch_size: int = 256,
) -> EncodedRows:
    normalized = _normalize_points(points, point_mask, bundle.payload)
    outputs: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    encoder = bundle.model.encoder
    with torch.inference_mode():
        for start in range(0, len(points), batch_size):
            end = start + batch_size
            values = torch.from_numpy(normalized[start:end])
            pm = torch.from_numpy(point_mask[start:end])
            fm = torch.from_numpy(frame_mask[start:end])
            embedding = encoder(values, pm, fm)
            outputs.append(embedding.numpy())
            if bundle.has_head:
                scores.append(torch.sigmoid(bundle.model.output(embedding).squeeze(-1)).numpy())
    score = np.concatenate(scores) if scores else np.full(len(points), np.nan)
    return EncodedRows(metadata.copy(), np.concatenate(outputs), score)


def _encode_tcn_rows(
    bundle: EncoderBundle,
    metadata: pd.DataFrame,
    features: np.ndarray,
    batch_size: int = 1024,
) -> EncodedRows:
    mean = np.asarray(bundle.payload["normalization_mean"], dtype=np.float32)
    std = np.asarray(bundle.payload["normalization_std"], dtype=np.float32)
    normalized = ((features - mean[None, None]) / std[None, None]).astype(np.float32)
    outputs: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            values = torch.from_numpy(normalized[start : start + batch_size])
            embedding = bundle.model.encoder(values)
            outputs.append(embedding.numpy())
            scores.append(torch.sigmoid(bundle.model.output(embedding).squeeze(-1)).numpy())
    return EncodedRows(metadata.copy(), np.concatenate(outputs), np.concatenate(scores))


def _point_normal_reference(bundle: EncoderBundle, dataset: str | Path) -> EncodedRows:
    with np.load(dataset, allow_pickle=False) as arrays:
        keep = arrays["label_source"] == "dguha_recording_activity_label"
        metadata = pd.DataFrame(
            {
                "recording_id": arrays["source_files"][keep],
                "split": arrays["split"][keep],
                "group": "normal_activity",
                "action": [_action_from_source(value) for value in arrays["source_files"][keep]],
            }
        )
        return _encode_point_rows(
            bundle,
            metadata,
            arrays["points"][keep],
            arrays["point_mask"][keep],
            arrays["frame_mask"][keep],
        )


def _tcn_normal_reference(bundle: EncoderBundle, dataset: str | Path) -> EncodedRows:
    with np.load(dataset, allow_pickle=False) as arrays:
        keep = arrays["label_source"] == "dguha_recording_activity_label"
        metadata = pd.DataFrame(
            {
                "recording_id": arrays["source_files"][keep],
                "split": arrays["split"][keep],
                "group": "normal_activity",
                "action": arrays["action"][keep],
            }
        )
        return _encode_tcn_rows(bundle, metadata, arrays["features"][keep])


def _analyze_encoder_geometry(
    bundle: EncoderBundle,
    trajectories: EncodedRows,
    normal: EncodedRows,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    train_normal = normal.metadata["split"].to_numpy() == "train"
    mean = normal.embeddings[train_normal].mean(axis=0)
    std = normal.embeddings[train_normal].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    trajectory_z = (trajectories.embeddings - mean) / std
    normal_z = (normal.embeddings - mean) / std

    train_target = (
        (trajectories.metadata["split"].to_numpy() == "train")
        & (trajectories.metadata["group"].to_numpy() == "fall")
        & trajectories.metadata["relative_seconds"].between(-1.0, -0.5).to_numpy()
    )
    target_frame = trajectories.metadata.loc[train_target, ["recording_id"]].copy()
    target_frame["row"] = np.flatnonzero(train_target)
    event_centroids = [
        trajectory_z[group["row"].to_numpy()].mean(axis=0)
        for _, group in target_frame.groupby("recording_id")
    ]
    target_centroid = np.mean(event_centroids, axis=0)
    norm = float(np.linalg.norm(target_centroid))
    direction = target_centroid / max(norm, 1e-12)

    geometry = trajectories.metadata.copy()
    geometry["encoder"] = bundle.name
    geometry["normal_centroid_distance"] = np.linalg.norm(trajectory_z, axis=1) / math.sqrt(bundle.dimension)
    geometry["target_projection"] = trajectory_z @ direction
    geometry["head_score"] = trajectories.head_score
    geometry["distance_from_t_minus_3"] = np.nan
    geometry["step_distance"] = np.nan
    for _, group in geometry.groupby("recording_id"):
        order = group.sort_values("relative_seconds").index.to_numpy()
        vectors = trajectory_z[order]
        times = geometry.loc[order, "relative_seconds"].to_numpy()
        baseline = np.flatnonzero(np.isclose(times, -3.0))
        if len(baseline):
            geometry.loc[order, "distance_from_t_minus_3"] = (
                np.linalg.norm(vectors - vectors[baseline[0]], axis=1) / math.sqrt(bundle.dimension)
            )
        if len(order) > 1:
            gaps = np.diff(times)
            steps = np.linalg.norm(np.diff(vectors, axis=0), axis=1) / math.sqrt(bundle.dimension)
            steps[~np.isclose(gaps, 0.1, atol=1e-5)] = np.nan
            geometry.loc[order[1:], "step_distance"] = steps

    normal_metrics = normal.metadata.copy()
    normal_metrics["projection"] = normal_z @ direction
    normal_metrics["distance"] = np.linalg.norm(normal_z, axis=1) / math.sqrt(bundle.dimension)
    normal_metrics["head_score"] = normal.head_score
    validation_normal = normal_metrics[normal_metrics["split"] == "validation"]
    normal_recording = validation_normal.groupby(["recording_id", "action"], as_index=False)[
        ["projection", "distance", "head_score"]
    ].median()

    separability: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260809)
    for time in TIME_GRID:
        fall = geometry[
            (geometry["split"] == "validation")
            & (geometry["group"] == "fall")
            & np.isclose(geometry["relative_seconds"], time)
        ]
        for comparator, normal_rows in (
            ("all_normal", normal_recording),
            (
                "normal_sit",
                normal_recording[
                    normal_recording["action"].astype(str).str.contains(
                        "Sit_down|SIT_STAND", case=False, regex=True
                    )
                ],
            ),
        ):
            for metric, fall_column, normal_column in (
                ("target_projection", "target_projection", "projection"),
                ("normal_centroid_distance", "normal_centroid_distance", "distance"),
                ("existing_head_score", "head_score", "head_score"),
            ):
                positive = fall[fall_column].dropna().to_numpy(dtype=float)
                negative = normal_rows[normal_column].dropna().to_numpy(dtype=float)
                if len(positive) < 3 or len(negative) < 3:
                    continue
                auc = _auc(positive, negative)
                low, high = _bootstrap_auc(positive, negative, rng)
                separability.append(
                    {
                        "encoder": bundle.name,
                        "relative_seconds": float(time),
                        "comparator": comparator,
                        "metric": metric,
                        "fall_event_count": len(positive),
                        "normal_recording_count": len(negative),
                        "auroc": auc,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                    }
                )

    fall_development = geometry[geometry["group"] == "fall"]
    correlations: list[float] = []
    for _, group in fall_development.groupby("recording_id"):
        if len(group) >= 6:
            rho = spearmanr(group["relative_seconds"], group["target_projection"]).statistic
            if np.isfinite(rho):
                correlations.append(float(rho))
    early = _earliest_stable_separation(separability, bundle.name)
    available = {
        str(float(time)): int(
            (
                (fall_development["split"] == "validation")
                & np.isclose(fall_development["relative_seconds"], time)
            ).sum()
        )
        for time in (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0)
    }
    summary = {
        "encoder": bundle.name,
        "embedding_dimension": bundle.dimension,
        "fall_label_supervised_encoder": bundle.name != "pointnet_gru_activity_pretrained",
        "validation_event_availability": json.dumps(available),
        "median_event_projection_spearman": float(np.median(correlations)) if correlations else math.nan,
        "same_positive_direction_fraction": float(np.mean(np.asarray(correlations) > 0)) if correlations else math.nan,
        "stable_forward_evolution": bool(
            correlations
            and float(np.median(correlations)) >= 0.30
            and float(np.mean(np.asarray(correlations) > 0)) >= 0.65
        ),
        "earliest_stable_validation_separation_seconds": early,
        "stable_separation_definition": (
            "static cross-recording separation: AUROC>=0.70 and lower bootstrap CI>0.50 "
            "for 3 consecutive 0.1 s bins; this is not by itself temporal evolution"
        ),
    }
    return geometry, separability, summary, {"mean": mean, "std": std, "direction": direction}


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative))
    ranks = pd.Series(values).rank(method="average").to_numpy()
    rank_sum = ranks[: len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def _bootstrap_auc(
    positive: np.ndarray, negative: np.ndarray, rng: np.random.Generator, repeats: int = 500
) -> tuple[float, float]:
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        values[index] = _auc(
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _earliest_stable_separation(rows: list[dict[str, Any]], encoder: str) -> float | None:
    table = pd.DataFrame(rows)
    if not len(table):
        return None
    selected = table[
        (table["encoder"] == encoder)
        & (table["comparator"] == "all_normal")
        & (table["metric"] == "target_projection")
        & (table["relative_seconds"] <= -0.1)
    ].sort_values("relative_seconds")
    stable = (selected["auroc"] >= 0.70) & (selected["bootstrap_ci_low"] > 0.50)
    flags = stable.to_numpy()
    times = selected["relative_seconds"].to_numpy()
    for index in range(max(0, len(flags) - 2)):
        if flags[index : index + 3].all() and np.allclose(np.diff(times[index : index + 3]), 0.1):
            return float(times[index])
    return None


def _audit_dguha_anchors(
    root: Path, events: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        radar_path = root / Path(str(event["source_file"]))
        kinect_path = radar_path.parent.parent / "kinect" / radar_path.name
        frames = parse_dguha_kinect(kinect_path)
        timestamps = np.asarray([frame.timestamp.timestamp() for frame in frames])
        relative = timestamps - timestamps[0]
        height = np.asarray([np.median(frame.points_mm[:, 2]) / 1000.0 for frame in frames])
        baseline_mask = relative <= 0.75
        final_mask = relative >= max(relative[-1] - 3.0, 0.0)
        baseline = float(np.median(height[baseline_mask]))
        final = float(np.median(height[final_mask]))
        drop = abs(final - baseline)
        direction = 1.0 if final >= baseline else -1.0
        progress = direction * (height - baseline)
        smooth = np.convolve(np.pad(progress, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid")
        speed = np.gradient(smooth, timestamps)
        baseline_progress = smooth[baseline_mask]
        median = float(np.median(baseline_progress))
        mad = float(np.median(np.abs(baseline_progress - median)))
        departure_threshold = max(0.02, 3.0 * 1.4826 * mad)
        current_threshold = max(0.08, 0.10 * drop)
        current_time = datetime.fromisoformat(str(event["descent_onset"])).timestamp()
        candidate_indices = {
            "baseline_departure_robust": _first_sustained(smooth >= departure_threshold, 3),
            "fixed_5cm_departure": _first_sustained(smooth >= 0.05, 3),
            "slow_descent_0p10mps": _first_sustained((smooth >= departure_threshold) & (speed >= 0.10), 3),
            "current_displacement_rule_recomputed": _first_sustained(smooth >= current_threshold, 3),
            "global_speed_crossing_0p25mps": _first_sustained(speed >= 0.25, 2),
        }
        row: dict[str, Any] = {
            "recording_id": str(event["source_file"]),
            "split": str(event["project_split"]),
            "subject_id": str(event["subject_id"]),
            "vertical_drop_m": drop,
            "current_displacement_threshold_m": current_threshold,
            "baseline_departure_threshold_m": departure_threshold,
            "current_to_near_floor_seconds": (
                datetime.fromisoformat(str(event["near_floor_level_reached"])).timestamp()
                - current_time
            ),
            "stored_rapid_descent_relative_to_current_seconds": (
                datetime.fromisoformat(str(event["rapid_descent_onset"])).timestamp()
                - current_time
            ),
        }
        for name, index in candidate_indices.items():
            row[f"{name}_relative_to_current_seconds"] = (
                float(timestamps[index] - current_time) if index is not None else math.nan
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    columns = [column for column in table if column.endswith("relative_to_current_seconds")]
    summary = {
        "current_rule": (
            "first 3 skeleton frames whose smoothed whole-body median-height displacement "
            "exceeds max(0.08 m, 10% of the eventual full drop)"
        ),
        "future_information_in_label_derivation": (
            "The 10% threshold and movement direction use the final 3 seconds of the recording. "
            "This is permissible for offline pseudo-label creation but is not an online-observable onset."
        ),
        "candidate_offsets_seconds": {
            column: _describe(table[column].dropna().to_numpy()) for column in columns
        },
        "semantic_limit": (
            "Skeleton displacement cannot identify loss of balance or distinguish an intentional "
            "descent from an involuntary fall transition."
        ),
    }
    return table, summary


def _first_sustained(mask: np.ndarray, count: int) -> int | None:
    values = np.asarray(mask, dtype=bool)
    if len(values) < count:
        return None
    hits = np.convolve(values.astype(np.int8), np.ones(count, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(hits == count)
    return int(indices[0]) if len(indices) else None


def _analyze_mmfall(
    bundle: EncoderBundle,
    dataset: str | Path,
    contract: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with np.load(dataset, allow_pickle=False) as arrays:
        metadata = pd.DataFrame(
            {
                "split": arrays["split"],
                "source_file": arrays["source_files"],
                "source_category": arrays["source_categories"],
                "anchor_frame": arrays["anchor_frames"],
                "seconds_to_anchor": arrays["seconds_to_anchor"],
                "label": arrays["labels"],
            }
        )
        encoded = _encode_tcn_rows(bundle, metadata, arrays["features"])
    z = (encoded.embeddings - contract["mean"]) / contract["std"]
    metadata["target_projection"] = z @ contract["direction"]
    metadata["head_score"] = encoded.head_score
    validation_normal = metadata[
        (metadata["split"] == "validation") & (metadata["label"] == 0)
    ].groupby("source_file", as_index=False)[["target_projection", "head_score"]].median()
    rows: list[dict[str, Any]] = []
    positive = metadata[(metadata["split"] == "validation") & (metadata["label"] == 1)].copy()
    positive["relative_seconds"] = -positive["seconds_to_anchor"]
    for time, group in positive.groupby(positive["relative_seconds"].round(1)):
        for metric in ("target_projection", "head_score"):
            pos = group[metric].to_numpy(dtype=float)
            neg = validation_normal[metric].to_numpy(dtype=float)
            if len(pos) >= 3 and len(neg) >= 3:
                rows.append(
                    {
                        "relative_seconds": float(time),
                        "metric": metric,
                        "positive_window_count": len(pos),
                        "normal_recording_count": len(neg),
                        "auroc": _auc(pos, neg),
                        "median_positive": float(np.median(pos)),
                        "median_normal": float(np.median(neg)),
                    }
                )
    event_rhos: list[float] = []
    positive["event_id"] = positive["source_file"].astype(str) + "#" + positive["anchor_frame"].astype(str)
    for _, group in positive.groupby("event_id"):
        if group["relative_seconds"].nunique() >= 5:
            rho = spearmanr(group["relative_seconds"], group["target_projection"]).statistic
            if np.isfinite(rho):
                event_rhos.append(float(rho))
    summary = {
        "available": True,
        "encoder": bundle.name,
        "validation_positive_anchor_count": int(positive["event_id"].nunique()),
        "lead_range_seconds": [0.2, 1.5],
        "median_event_projection_spearman": float(np.median(event_rhos)) if event_rhos else math.nan,
        "same_positive_direction_fraction": float(np.mean(np.asarray(event_rhos) > 0)) if event_rhos else math.nan,
        "label_warning": "official fall-motion anchor, not verified loss-of-balance or impact",
        "split_warning": "no subject identity; the fixed split is recording/category based",
    }
    return pd.DataFrame(rows), summary


def _other_dataset_label_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "DGUHA",
                "fall_positive": True,
                "temporal_anchor": "skeleton-derived displacement onset",
                "loss_of_balance_or_impact_verified": False,
                "identity_split": "project subject-disjoint",
                "prediction_feasibility": "weak pseudo-label only",
            },
            {
                "dataset": "mmFall DS2",
                "fall_positive": True,
                "temporal_anchor": "official fall-motion frame",
                "loss_of_balance_or_impact_verified": False,
                "identity_split": "participant identity unavailable",
                "prediction_feasibility": "supplementary 0.2-1.5 s anchor analysis only",
            },
            {
                "dataset": "IWR6843 fall-102",
                "fall_positive": True,
                "temporal_anchor": "recording-level label only",
                "loss_of_balance_or_impact_verified": False,
                "identity_split": "3-subject disjoint",
                "prediction_feasibility": "fall-sequence classification only",
            },
            {
                "dataset": "mmRadPose",
                "fall_positive": False,
                "temporal_anchor": "activity label only",
                "loss_of_balance_or_impact_verified": False,
                "identity_split": "12-subject disjoint",
                "prediction_feasibility": "normal representation pretraining only",
            },
            {
                "dataset": "RadHAR",
                "fall_positive": False,
                "temporal_anchor": "activity recording",
                "loss_of_balance_or_impact_verified": False,
                "identity_split": "not established",
                "prediction_feasibility": "squat/jump hard negatives only",
            },
            {
                "dataset": "mmWave OCPID PointCloudData",
                "fall_positive": False,
                "temporal_anchor": "walking condition",
                "loss_of_balance_or_impact_verified": False,
                "identity_split": "9 subjects",
                "prediction_feasibility": "walking normal manifold only",
            },
        ]
    )


def _plot_embedding_geometry(
    geometry: pd.DataFrame, separability: pd.DataFrame, destination: Path
) -> None:
    _configure_matplotlib()
    encoders = list(geometry["encoder"].drop_duplicates())
    fig, axes = plt.subplots(
        len(encoders), 3, figsize=(183 / 25.4, 178 / 25.4), sharex="col"
    )
    colors = {"fall": "#a9433e", "normal_sit": "#3f70ad"}
    for row_index, encoder in enumerate(encoders):
        subset = geometry[geometry["encoder"] == encoder]
        for group, label in (("fall", "Fall"), ("normal_sit", "Sit/stand")):
            _median_iqr(
                axes[row_index, 0], subset[subset["group"] == group],
                "distance_from_t_minus_3", colors[group], label,
            )
            _median_iqr(
                axes[row_index, 1], subset[subset["group"] == group],
                "target_projection", colors[group], label,
            )
        auc = separability[
            (separability["encoder"] == encoder)
            & (separability["comparator"] == "all_normal")
        ]
        for metric, color, label in (
            ("target_projection", "#76528b", "Centroid projection"),
            ("normal_centroid_distance", "#5b8e62", "Normal distance"),
            ("existing_head_score", "#d08a3d", "Existing head"),
        ):
            selected = auc[auc["metric"] == metric]
            if len(selected):
                axes[row_index, 2].plot(
                    selected["relative_seconds"], selected["auroc"],
                    color=color, lw=1.2, label=label,
                )
                axes[row_index, 2].fill_between(
                    selected["relative_seconds"], selected["bootstrap_ci_low"],
                    selected["bootstrap_ci_high"], color=color, alpha=0.13, lw=0,
                )
        axes[row_index, 2].axhline(0.5, color="#777777", lw=0.7, ls=":")
        for ax in axes[row_index]:
            ax.axvline(0.0, color="#222222", lw=0.7, ls="--")
            ax.axvspan(-1.0, -0.5, color="#e2b967", alpha=0.12, lw=0)
            ax.grid(axis="y", color="#dddddd", lw=0.4)
            ax.set_xlim(-3.0, 0.0)
        row_labels = {
            "pointnet_gru_activity_pretrained": "Activity-pretrained\nPointNet–GRU",
            "pointnet_gru_prefall_finetuned": "Pre-fall fine-tuned\nPointNet–GRU",
            "causal_tcn_prefall": "Pre-fall\ncausal TCN",
        }
        axes[row_index, 0].set_ylabel(row_labels.get(encoder, encoder), fontsize=6)
    axes[0, 0].set_title("Distance from embedding at t = −3 s")
    axes[0, 1].set_title("Projection toward train fall-target centroid")
    axes[0, 2].set_title("Validation separation from normal recordings")
    axes[-1, 0].set_xlabel("Seconds to current descent onset")
    axes[-1, 1].set_xlabel("Seconds to current descent onset")
    axes[-1, 2].set_xlabel("Seconds to current descent onset")
    axes[0, 0].legend(loc="upper left", fontsize=6)
    axes[0, 2].legend(loc="lower right", fontsize=5.5)
    fig.suptitle("Frozen-encoder geometry before the current DGUHA descent onset", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    _save_figure(fig, destination / "figure1_embedding_geometry")
    plt.close(fig)


def _plot_event_heatmaps(geometry: pd.DataFrame, destination: Path) -> None:
    _configure_matplotlib()
    encoders = list(geometry["encoder"].drop_duplicates())
    fig, axes = plt.subplots(len(encoders), 1, figsize=(183 / 25.4, 130 / 25.4), sharex=True)
    image = None
    for ax, encoder in zip(np.atleast_1d(axes), encoders):
        selected = geometry[
            (geometry["encoder"] == encoder)
            & (geometry["group"] == "fall")
            & (geometry["split"] == "validation")
        ]
        events = sorted(selected["recording_id"].unique())
        matrix = np.full((len(events), len(TIME_GRID)), np.nan)
        for event_index, event in enumerate(events):
            event_rows = selected[selected["recording_id"] == event]
            for _, row in event_rows.iterrows():
                time_index = int(np.argmin(np.abs(TIME_GRID - row["relative_seconds"])))
                matrix[event_index, time_index] = row["target_projection"]
        scale = np.nanquantile(np.abs(matrix), 0.95)
        image = ax.imshow(
            np.clip(matrix / max(scale, 1e-6), -1, 1), cmap="coolwarm", vmin=-1, vmax=1,
            aspect="auto", interpolation="nearest",
            extent=(TIME_GRID[0], TIME_GRID[-1], len(events), 0),
        )
        ax.axvline(0.0, color="black", lw=0.7, ls="--")
        ax.set_ylabel(encoder.replace("_", "\n"), fontsize=6)
    axes[-1].set_xlabel("Seconds to current descent onset")
    fig.subplots_adjust(left=0.20, right=0.84, top=0.93, bottom=0.10, hspace=0.20)
    if image is not None:
        cax = fig.add_axes((0.875, 0.24, 0.018, 0.52))
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label("Per-encoder scaled target projection", fontsize=6)
    fig.suptitle("Validation-event embedding trajectories (missing cells = insufficient history)")
    _save_figure(fig, destination / "figure2_validation_event_embedding_heatmaps")
    plt.close(fig)


def _plot_anchor_sensitivity(table: pd.DataFrame, destination: Path) -> None:
    _configure_matplotlib()
    columns = [
        "baseline_departure_robust_relative_to_current_seconds",
        "fixed_5cm_departure_relative_to_current_seconds",
        "slow_descent_0p10mps_relative_to_current_seconds",
        "stored_rapid_descent_relative_to_current_seconds",
    ]
    labels = [
        "Robust baseline\ndeparture",
        "Fixed 5 cm\ndeparture",
        "Slow descent\n≥0.10 m/s",
        "Stored rapid\ndescent",
    ]
    values = [table[column].dropna().to_numpy() for column in columns]
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 82 / 25.4), sharex=True)
    colors = ("#9eb7d5", "#bccde0", "#e7c582", "#cf8179")
    for ax, zoomed in zip(axes, (False, True)):
        box = ax.boxplot(
            values,
            tick_labels=labels,
            patch_artist=True,
            showfliers=not zoomed,
            flierprops={"markersize": 2.5, "markerfacecolor": "#555555", "markeredgewidth": 0},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#555555")
        ax.axhline(0.0, color="#222222", lw=0.8, ls="--")
        ax.grid(axis="y", color="#dddddd", lw=0.4)
        ax.set_title("Central range" if zoomed else "All events, including early outliers")
    axes[0].set_ylabel("Candidate anchor minus current onset (s)")
    axes[1].set_ylim(-0.8, 0.7)
    fig.suptitle("DGUHA onset is threshold-dependent", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, destination / "figure3_dguha_anchor_sensitivity")
    plt.close(fig)


def _plot_mmfall(table: pd.DataFrame, destination: Path) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(90 / 25.4, 68 / 25.4))
    for metric, color, label in (
        ("target_projection", "#76528b", "Frozen embedding projection"),
        ("head_score", "#d08a3d", "Existing DGUHA head"),
    ):
        selected = table[table["metric"] == metric]
        ax.plot(selected["relative_seconds"], selected["auroc"], marker="o", ms=2.5, color=color, label=label)
    ax.axhline(0.5, color="#777777", lw=0.7, ls=":")
    ax.set_xlabel("Seconds to mmFall fall-motion anchor")
    ax.set_ylabel("Validation AUROC vs mmFall normal recordings")
    ax.set_ylim(0.35, 1.0)
    ax.grid(axis="y", color="#dddddd", lw=0.4)
    ax.legend(fontsize=6)
    ax.set_title("External mmFall check (anchor semantics remain weak)")
    fig.tight_layout()
    _save_figure(fig, destination / "figure4_mmfall_external_tcn")
    plt.close(fig)


def _median_iqr(ax, table: pd.DataFrame, value: str, color: str, label: str) -> None:
    summary = table.groupby("relative_seconds")[value].agg(
        median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    )
    ax.plot(summary.index, summary["median"], color=color, lw=1.3, label=label)
    ax.fill_between(summary.index, summary["q25"], summary["q75"], color=color, alpha=0.16, lw=0)


def _save_embedding_npz(destination: Path, raw_sets: dict[str, EncodedRows]) -> None:
    payload: dict[str, np.ndarray] = {}
    for name, rows in raw_sets.items():
        prefix = name.replace("-", "_")
        payload[f"{prefix}_embedding"] = rows.embeddings.astype(np.float32)
        payload[f"{prefix}_head_score"] = rows.head_score.astype(np.float32)
    first = next(iter(raw_sets.values())).metadata
    for column in ("recording_id", "split", "subject_id", "group", "relative_seconds"):
        payload[column] = first[column].to_numpy()
    np.savez_compressed(destination / "embedding_trajectories.npz", **payload)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _action_from_source(value: str) -> str:
    parts = Path(str(value)).parts
    return parts[1] if len(parts) > 1 else "unknown"


def _describe(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def _verdict(
    trajectory: pd.DataFrame,
    separability: pd.DataFrame,
    anchor_summary: dict[str, Any],
    mmfall_summary: dict[str, Any],
) -> dict[str, Any]:
    pretrain = trajectory[trajectory["encoder"] == "pointnet_gru_activity_pretrained"].iloc[0]
    supervised = trajectory[trajectory["fall_label_supervised_encoder"]]
    return {
        "independent_pretrained_embedding_has_static_early_separation": (
            pretrain["earliest_stable_validation_separation_seconds"] is not None
            and not pd.isna(pretrain["earliest_stable_validation_separation_seconds"])
        ),
        "independent_pretrained_embedding_has_stable_forward_evolution": bool(
            pretrain["stable_forward_evolution"]
        ),
        "dguha_any_encoder_has_stable_forward_evolution": bool(
            trajectory["stable_forward_evolution"].any()
        ),
        "supervised_encoder_earliest_static_separation": {
            str(row["encoder"]): (
                None
                if pd.isna(row["earliest_stable_validation_separation_seconds"])
                else float(row["earliest_stable_validation_separation_seconds"])
            )
            for _, row in supervised.iterrows()
        },
        "representation_conclusion": (
            "DGUHA fall recordings are statically separable from normal recordings before the labelled "
            "horizon, including in a fall-untrained PointNet embedding, but no tested encoder shows a "
            "consistent forward trajectory toward the labelled fall target. Static separation at t=-3 s "
            "is more consistent with recording/action context or selection bias than advance prediction."
        ),
        "label_conclusion": (
            "descent_onset is a threshold-dependent whole-body displacement pseudo-anchor, not a "
            "verified loss-of-balance target; its timing cannot be judged clinically from DGUHA alone"
        ),
        "other_dataset_conclusion": (
            "mmFall supplies a second weak fall-motion anchor; IWR6843 has recording labels only; "
            "mmRadPose, RadHAR and OCPID supply normal actions but no prediction-positive timing"
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen embedding and DGUHA label feasibility analysis")
    parser.add_argument("--dguha-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--point-dataset", type=Path, required=True)
    parser.add_argument("--tcn-dataset", type=Path, required=True)
    parser.add_argument("--point-pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--point-finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--tcn-checkpoint", type=Path, required=True)
    parser.add_argument("--mmfall-dataset", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_analysis(
        dguha_root=args.dguha_root,
        events_file=args.events,
        point_dataset=args.point_dataset,
        tcn_dataset=args.tcn_dataset,
        point_pretrain_checkpoint=args.point_pretrain_checkpoint,
        point_finetuned_checkpoint=args.point_finetuned_checkpoint,
        tcn_checkpoint=args.tcn_checkpoint,
        mmfall_dataset=args.mmfall_dataset,
        output_directory=args.output_directory,
    )
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
