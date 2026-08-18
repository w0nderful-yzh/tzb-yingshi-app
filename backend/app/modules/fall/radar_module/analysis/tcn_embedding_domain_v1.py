from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from radar_module.contracts import RadarFrame, Room
from radar_module.dataset.v2_export import _load_replay_frames
from radar_module.model.temporal_models_v3 import (
    EXPERIMENT_MODEL_VERSION,
    TemporalBinaryModel,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


ANALYSIS_VERSION = "tcn_embedding_domain_v1"
FROZEN_CHECKPOINT_SHA256 = {
    "B0": "0792a712b57ae89875b2d57e6ba7a20763618a2718e961cf8c48acebe34970ef",
    "M1": "ee9a9abc2a6ce384f4aa0abb7f4dfcd8aafb70faba82c031c003757cdcfbaa9c",
    "M2": "c2d11a97696c9e34917b8b0c3aaebaf4d0e3fec17a355e938e071daae73725e2",
}
TARGET_HORIZON_SECONDS = (0.5, 1.0)


def analyze_tcn_embedding_domains(
    *,
    dataset_path: str | Path,
    checkpoints: Mapping[str, str | Path],
    normal_replay_path: str | Path,
    fall_replay_path: str | Path,
    phase_events_path: str | Path,
    phase_compliance_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dataset_file = Path(dataset_path).resolve()
    normal_replay = Path(normal_replay_path).resolve()
    fall_replay = Path(fall_replay_path).resolve()
    checkpoint_files = {name: Path(path).resolve() for name, path in checkpoints.items()}
    if set(checkpoint_files) != set(FROZEN_CHECKPOINT_SHA256):
        raise ValueError("checkpoints must contain exactly B0, M1 and M2")

    frozen_before = _verify_frozen_checkpoints(checkpoint_files)
    dataset = _load_dataset(dataset_file)
    dguha_train = _select_dguha_positive(dataset, split="train")
    dguha_validation = _select_dguha_positive(dataset, split="validation")

    normal_windows = _extract_replay_windows(normal_replay)
    fall_windows = _extract_replay_windows(fall_replay)
    onset, impact, fall_annotation = _load_fall_annotation(
        Path(phase_events_path).resolve(), Path(phase_compliance_path).resolve()
    )
    fall_masks = select_fall_window_masks(
        fall_windows["timestamps"], onset=onset, impact=impact
    )
    target_features = fall_windows["features"][fall_masks["target_prefall"]]
    context_features = fall_windows["features"][fall_masks["prefall_context"]]
    descent_features = fall_windows["features"][fall_masks["descent"]]
    if min(len(target_features), len(descent_features)) == 0:
        raise ValueError("video-aligned fall interval produced an empty analysis group")

    raw_domains = {
        "dguha_prefall_train": dguha_train,
        "dguha_prefall_validation": dguha_validation,
        "iwr_normal": normal_windows["features"],
        "iwr_fall_target_prefall": target_features,
        "iwr_fall_prefall_context": context_features,
        "iwr_fall_descent": descent_features,
    }
    model_outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    model_reports: dict[str, Any] = {}
    distance_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []

    for model_name, checkpoint_file in checkpoint_files.items():
        checkpoint = _load_checkpoint(checkpoint_file)
        model = _model_from_checkpoint(checkpoint)
        encoded = {
            domain: _encode(model, checkpoint, values)
            for domain, values in raw_domains.items()
        }
        model_outputs[model_name] = encoded
        geometry = _embedding_geometry(encoded)
        score_diagnostic = _score_diagnostic(encoded)
        model_reports[model_name] = {
            "checkpoint_file": str(checkpoint_file),
            "checkpoint_sha256": frozen_before[model_name],
            "decision_threshold": float(checkpoint["decision_threshold"]),
            "embedding_dimension": int(
                encoded["dguha_prefall_validation"]["embedding"].shape[1]
            ),
            "embedding_geometry": geometry,
            "score_diagnostic": score_diagnostic,
        }
        distance_rows.extend(_distance_table_rows(model_name, geometry))
        for domain, values in encoded.items():
            score_rows.append(
                {
                    "model": model_name,
                    "domain": domain,
                    **_prefixed_description("deployed_score", values["score"]),
                    **_prefixed_description("logit", values["logit"]),
                    **_prefixed_description(
                        "stable_log10_score", _log10_sigmoid(values["logit"])
                    ),
                }
            )
            absolute = np.abs(values["normalized_features"].astype(np.float64))
            ood_rows.append(
                {
                    "model": model_name,
                    "domain": domain,
                    "value_count": int(absolute.size),
                    "absolute_normalized_median": float(np.median(absolute)),
                    "absolute_normalized_p95": float(np.quantile(absolute, 0.95)),
                    "absolute_normalized_p99": float(np.quantile(absolute, 0.99)),
                    "fraction_above_5sigma": float(np.mean(absolute > 5.0)),
                    "fraction_above_10sigma": float(np.mean(absolute > 10.0)),
                }
            )

    feature_rows, feature_summary = _feature_statistics(raw_domains)
    conclusion = _conclude(model_reports, feature_summary)
    sample_counts = {name: int(len(values)) for name, values in raw_domains.items()}
    frozen_after = _verify_frozen_checkpoints(checkpoint_files)
    if frozen_after != frozen_before:
        raise RuntimeError("a frozen checkpoint changed during analysis")

    report: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "training_performed": False,
        "model_or_threshold_modified": False,
        "sealed_test_evaluated": False,
        "dataset_file": str(dataset_file),
        "dataset_sha256": _sha256(dataset_file),
        "feature_version": FEATURE_VERSION_V2,
        "feature_names": list(FEATURE_NAMES_V2),
        "dguha_reference_policy": (
            "headline geometry uses validation positives; training positives are reported "
            "separately; sealed test remains untouched"
        ),
        "iwr_fall_annotation": fall_annotation,
        "iwr_replays": {
            "normal": {
                "path": str(normal_replay),
                "sha256": _sha256(normal_replay),
                "quality_counts": normal_windows["quality_counts"],
            },
            "controlled_fall": {
                "path": str(fall_replay),
                "sha256": _sha256(fall_replay),
                "quality_counts": fall_windows["quality_counts"],
            },
        },
        "sample_counts": sample_counts,
        "frozen_checkpoint_sha256_before": frozen_before,
        "frozen_checkpoint_sha256_after": frozen_after,
        "models": model_reports,
        "feature_shift_summary": feature_summary,
        "decision": conclusion,
        "limitations": [
            "There is one video-aligned IWR6843 controlled-fall event, so windows are not independent events.",
            "DGUHA positive windows end 0.5-1.0 s before descent onset; the target-matched IWR group is therefore the primary comparison.",
            "Embedding distances are valid only within the same frozen checkpoint, not across B0/M1/M2 coordinate systems.",
            "A monotonic score calibration cannot improve score ranking AUROC.",
        ],
    }
    _write_csv(destination / "embedding_distances.csv", distance_rows)
    _write_csv(destination / "score_distributions.csv", score_rows)
    _write_csv(destination / "normalized_feature_ood.csv", ood_rows)
    _write_csv(destination / "feature_statistics.csv", feature_rows)
    _save_embeddings(destination / "embedding_samples.npz", model_outputs)
    _plot_embedding_pca(model_outputs, destination / "embedding_pca.png")
    _plot_score_distributions(model_outputs, destination / "score_distributions.png")
    _plot_feature_shift(feature_rows, destination / "feature_shift_heatmap.png")
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "ANALYSIS_REPORT.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return report


def select_fall_window_masks(
    timestamps: Sequence[datetime], *, onset: datetime, impact: datetime
) -> dict[str, np.ndarray]:
    if not onset < impact:
        raise ValueError("fall onset must precede impact")
    seconds = np.asarray([(value - onset).total_seconds() for value in timestamps])
    return {
        "target_prefall": (seconds >= -TARGET_HORIZON_SECONDS[1] - 1e-9)
        & (seconds <= -TARGET_HORIZON_SECONDS[0] + 1e-9),
        "prefall_context": (seconds >= -2.0 - 1e-9) & (seconds < -0.5 + 1e-9),
        "descent": (seconds >= -1e-9)
        & (seconds <= (impact - onset).total_seconds() + 1e-9),
    }


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as handle:
        required = {
            "features",
            "labels",
            "split",
            "dataset_origin",
            "feature_version",
            "feature_names",
            "seconds_to_onset",
        }
        missing = required.difference(handle.files)
        if missing:
            raise ValueError(f"dataset missing {sorted(missing)}")
        result = {key: np.asarray(handle[key]) for key in handle.files}
    if str(result["feature_version"].item()) != FEATURE_VERSION_V2:
        raise ValueError("dataset feature version is incompatible")
    if tuple(map(str, result["feature_names"])) != FEATURE_NAMES_V2:
        raise ValueError("dataset feature order is incompatible")
    return result


def _select_dguha_positive(dataset: Mapping[str, np.ndarray], *, split: str) -> np.ndarray:
    selected = (
        (dataset["dataset_origin"] == "dguha")
        & (dataset["labels"] == 1)
        & (dataset["split"] == split)
    )
    values = np.asarray(dataset["features"][selected], dtype=np.float32)
    if not len(values):
        raise ValueError(f"no DGUHA positive windows for {split}")
    horizons = np.asarray(dataset["seconds_to_onset"][selected], dtype=np.float64)
    if np.any(horizons < 0.5 - 1e-6) or np.any(horizons > 1.0 + 1e-6):
        raise ValueError("DGUHA positive horizon is not 0.5-1.0 seconds")
    return values


def _extract_replay_windows(path: Path) -> dict[str, Any]:
    frames = _load_replay_frames(path, default_room=Room.BATHROOM)
    extractor = RadarTemporalFeatureExtractorV2()
    history: deque[RadarFrame] = deque()
    features: list[np.ndarray] = []
    timestamps: list[datetime] = []
    qualities: Counter[str] = Counter()
    last_timestamp: datetime | None = None
    for frame in frames:
        history.append(frame)
        while history and (frame.timestamp - history[0].timestamp).total_seconds() > 2.2:
            history.popleft()
        if last_timestamp is not None and (
            frame.timestamp - last_timestamp
        ).total_seconds() < 0.095:
            continue
        last_timestamp = frame.timestamp
        if (frame.timestamp - history[0].timestamp).total_seconds() < 1.9:
            qualities["WARMUP"] += 1
            continue
        window = extractor.transform(tuple(history), end_timestamp=frame.timestamp)
        qualities[window.data_quality.value] += 1
        if window.data_quality is TemporalDataQuality.INSUFFICIENT_DATA:
            continue
        features.append(np.asarray(window.values, dtype=np.float32))
        timestamps.append(frame.timestamp)
    if not features:
        raise ValueError(f"replay produced no valid feature windows: {path}")
    return {
        "features": np.stack(features).astype(np.float32, copy=False),
        "timestamps": timestamps,
        "quality_counts": dict(qualities),
    }


def _load_fall_annotation(
    phase_events_path: Path, phase_compliance_path: Path
) -> tuple[datetime, datetime, dict[str, Any]]:
    compliance = json.loads(phase_compliance_path.read_text(encoding="utf-8"))
    phases = compliance["phases"]
    candidates = [
        (phase_id, value)
        for phase_id, value in phases.items()
        if value.get("analysis_label") == "controlled_forward_fall"
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one video-aligned controlled fall phase")
    phase_id, annotation = candidates[0]
    starts: dict[str, datetime] = {}
    for line in phase_events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "PHASE_START":
            starts[str(event["phase_id"])] = datetime.fromisoformat(
                str(event["frame_timestamp"])
            )
    phase_start = starts.get(phase_id)
    if phase_start is None:
        raise ValueError("annotated fall phase has no phase-start timestamp")
    onset_seconds = float(annotation["action_onset_seconds"])
    impact_seconds = float(annotation["impact_seconds"])
    onset = phase_start + timedelta(seconds=onset_seconds)
    impact = phase_start + timedelta(seconds=impact_seconds)
    return onset, impact, {
        "source": str(compliance.get("source")),
        "phase_id": phase_id,
        "phase_start": phase_start.isoformat(),
        "action_onset": onset.isoformat(),
        "impact": impact.isoformat(),
        "descent_duration_seconds": (impact - onset).total_seconds(),
        "target_horizon_seconds_before_onset": list(TARGET_HORIZON_SECONDS),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != EXPERIMENT_MODEL_VERSION:
        raise ValueError(f"checkpoint model version is incompatible: {path}")
    if checkpoint.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError(f"checkpoint feature version is incompatible: {path}")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES_V2:
        raise ValueError(f"checkpoint feature order is incompatible: {path}")
    return checkpoint


def _model_from_checkpoint(checkpoint: Mapping[str, Any]) -> TemporalBinaryModel:
    model = TemporalBinaryModel(
        architecture=str(checkpoint["model_architecture"]),
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _encode(
    model: TemporalBinaryModel,
    checkpoint: Mapping[str, Any],
    raw_features: np.ndarray,
) -> dict[str, np.ndarray]:
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = ((raw_features - mean[None, None]) / std[None, None]).astype(
        np.float32
    )
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 512):
            batch = torch.from_numpy(normalized[start : start + 512])
            embedding = model.encoder(batch)
            logit = model.output(embedding).squeeze(-1)
            score = torch.sigmoid(logit)
            embeddings.append(embedding.numpy().astype(np.float32))
            logits.append(logit.numpy().astype(np.float64))
            scores.append(score.numpy().astype(np.float64))
    return {
        "embedding": np.concatenate(embeddings),
        "logit": np.concatenate(logits),
        "score": np.concatenate(scores),
        "normalized_features": normalized,
    }


def _embedding_geometry(
    encoded: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    reference = encoded["dguha_prefall_validation"]["embedding"].astype(np.float64)
    reference_centroid = reference.mean(axis=0)
    reference_radius = np.linalg.norm(reference - reference_centroid, axis=1)
    reference_loo_nn = _leave_one_out_nearest_distance(reference)
    radius_scale = max(float(np.median(reference_radius)), 1e-12)
    nn_scale = max(float(np.median(reference_loo_nn)), 1e-12)
    domains: dict[str, Any] = {}
    for name, values in encoded.items():
        embedding = values["embedding"].astype(np.float64)
        centroid = embedding.mean(axis=0)
        distances = _pairwise_distances(embedding, reference)
        nearest = distances.min(axis=1)
        domains[name] = {
            "sample_count": int(len(embedding)),
            "centroid_euclidean_to_dguha_validation": float(
                np.linalg.norm(centroid - reference_centroid)
            ),
            "centroid_cosine_distance_to_dguha_validation": _cosine_distance(
                centroid, reference_centroid
            ),
            "centroid_distance_over_dguha_radius": float(
                np.linalg.norm(centroid - reference_centroid) / radius_scale
            ),
            "nearest_dguha_validation_distance_median": float(np.median(nearest)),
            "nearest_distance_over_dguha_loo_median": float(
                np.median(nearest) / nn_scale
            ),
            "within_domain_radius_median": float(
                np.median(np.linalg.norm(embedding - centroid, axis=1))
            ),
        }
    normal_centroid = encoded["iwr_normal"]["embedding"].mean(axis=0).astype(np.float64)
    for fall_domain in (
        "iwr_fall_target_prefall",
        "iwr_fall_prefall_context",
        "iwr_fall_descent",
    ):
        values = encoded[fall_domain]["embedding"].astype(np.float64)
        to_dguha = np.linalg.norm(values - reference_centroid, axis=1)
        to_normal = np.linalg.norm(values - normal_centroid, axis=1)
        domains[fall_domain]["fraction_closer_to_dguha_than_iwr_normal"] = float(
            np.mean(to_dguha < to_normal)
        )
        domains[fall_domain]["median_distance_to_iwr_normal_centroid"] = float(
            np.median(to_normal)
        )
    return {
        "dguha_validation_radius_median": radius_scale,
        "dguha_validation_loo_nearest_distance_median": nn_scale,
        "domains": domains,
    }


def _score_diagnostic(encoded: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    normal = encoded["iwr_normal"]["score"]
    normal_logit = encoded["iwr_normal"]["logit"]
    result: dict[str, Any] = {}
    for domain in (
        "iwr_fall_target_prefall",
        "iwr_fall_prefall_context",
        "iwr_fall_descent",
    ):
        fall = encoded[domain]["score"]
        fall_logit = encoded[domain]["logit"]
        result[domain] = {
            "deployed_float32_score_distribution": _describe(fall),
            "logit_distribution": _describe(fall_logit),
            "stable_log10_score_distribution": _describe(
                _log10_sigmoid(fall_logit)
            ),
            "logit_ranking_auroc_vs_iwr_normal": _binary_auroc(
                fall_logit, normal_logit
            ),
            "deployed_score_auroc_vs_iwr_normal": _binary_auroc(fall, normal),
            "median_logit_percentile_in_iwr_normal": float(
                np.mean(normal_logit <= np.median(fall_logit))
            ),
        }
    result["iwr_normal"] = {
        "deployed_float32_score_distribution": _describe(normal),
        "logit_distribution": _describe(normal_logit),
        "stable_log10_score_distribution": _describe(_log10_sigmoid(normal_logit)),
    }
    validation = encoded["dguha_prefall_validation"]
    result["dguha_validation_positive"] = {
        "deployed_float32_score_distribution": _describe(validation["score"]),
        "logit_distribution": _describe(validation["logit"]),
        "stable_log10_score_distribution": _describe(
            _log10_sigmoid(validation["logit"])
        ),
    }
    return result


def _feature_statistics(
    domains: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = {
        domain: _window_feature_summaries(values) for domain, values in domains.items()
    }
    reference = summaries["dguha_prefall_validation"]
    rows: list[dict[str, Any]] = []
    top_shifts: dict[str, list[dict[str, Any]]] = {}
    for domain, aggregate in summaries.items():
        domain_rows: list[dict[str, Any]] = []
        for statistic_name, values in aggregate.items():
            reference_values = reference[statistic_name]
            for feature_index, feature_name in enumerate(FEATURE_NAMES_V2):
                current = values[:, feature_index]
                baseline = reference_values[:, feature_index]
                effect = _standardized_mean_difference(current, baseline)
                row = {
                    "domain": domain,
                    "statistic": statistic_name,
                    "feature": feature_name,
                    "sample_count": int(len(current)),
                    "mean": float(np.mean(current)),
                    "median": float(np.median(current)),
                    "p10": float(np.quantile(current, 0.10)),
                    "p90": float(np.quantile(current, 0.90)),
                    "standardized_mean_difference_vs_dguha_validation": effect,
                }
                rows.append(row)
                domain_rows.append(row)
        ranked = sorted(
            domain_rows,
            key=lambda row: abs(
                row["standardized_mean_difference_vs_dguha_validation"]
            ),
            reverse=True,
        )
        top_shifts[domain] = [
            {
                "statistic": row["statistic"],
                "feature": row["feature"],
                "standardized_mean_difference": row[
                    "standardized_mean_difference_vs_dguha_validation"
                ],
            }
            for row in ranked[:10]
        ]
    primary = [
        abs(row["standardized_mean_difference_vs_dguha_validation"])
        for row in rows
        if row["domain"] == "iwr_fall_target_prefall"
        and row["statistic"] in {"window_mean", "temporal_slope"}
    ]
    primary_features = (
        "centroid_z",
        "height_range",
        "point_count",
        "mean_velocity",
        "max_abs_velocity",
        "velocity_std",
        "vertical_velocity",
    )
    primary_domains = (
        "dguha_prefall_validation",
        "iwr_normal",
        "iwr_fall_target_prefall",
        "iwr_fall_descent",
    )
    primary_feature_window_means = {
        feature: {
            domain: float(summaries[domain]["window_mean"][:, index].mean())
            for domain in primary_domains
        }
        for index, feature in enumerate(FEATURE_NAMES_V2)
        if feature in primary_features
    }
    return rows, {
        "top_absolute_shifts": top_shifts,
        "iwr_target_prefall_median_absolute_effect_size": float(np.median(primary)),
        "iwr_target_prefall_features_above_effect_size_1": int(
            np.sum(np.asarray(primary) >= 1.0)
        ),
        "iwr_target_prefall_effect_size_component_count": int(len(primary)),
        "primary_feature_window_means": primary_feature_window_means,
        "comparison_unit": "per-window mean, final value and temporal slope",
    }


def _window_feature_summaries(values: np.ndarray) -> dict[str, np.ndarray]:
    time = np.arange(values.shape[1], dtype=np.float64)
    time -= time.mean()
    denominator = float(np.sum(time * time))
    centered = values.astype(np.float64) - values.mean(axis=1, keepdims=True)
    slope = np.sum(centered * time[None, :, None], axis=1) / denominator
    return {
        "window_mean": values.mean(axis=1, dtype=np.float64),
        "final_value": values[:, -1].astype(np.float64),
        "temporal_slope": slope,
    }


def _conclude(
    model_reports: Mapping[str, Any], feature_summary: Mapping[str, Any]
) -> dict[str, Any]:
    b0 = model_reports["B0"]
    target_geometry = b0["embedding_geometry"]["domains"][
        "iwr_fall_target_prefall"
    ]
    target_score = b0["score_diagnostic"]["iwr_fall_target_prefall"]
    descent_score = b0["score_diagnostic"]["iwr_fall_descent"]
    embedding_outside = (
        target_geometry["nearest_distance_over_dguha_loo_median"] > 2.0
        or target_geometry["centroid_distance_over_dguha_radius"] > 2.0
    )
    ranking_weak = max(
        target_score["logit_ranking_auroc_vs_iwr_normal"],
        descent_score["logit_ranking_auroc_vs_iwr_normal"],
    ) < 0.70
    feature_shift = (
        feature_summary["iwr_target_prefall_median_absolute_effect_size"] >= 1.0
        or feature_summary["iwr_target_prefall_features_above_effect_size_1"] >= 10
    )
    calibration_only_supported = not embedding_outside and not ranking_weak
    decision = "B_score_calibration_only" if calibration_only_supported else "A_positive_domain_adaptation"
    return {
        "primary_decision": decision,
        "score_calibration_only_supported": calibration_only_supported,
        "embedding_outside_dguha_positive_support": bool(embedding_outside),
        "score_ranking_is_too_weak_for_monotonic_calibration": bool(ranking_weak),
        "large_input_feature_shift": bool(feature_shift),
        "decision_rule": (
            "Calibration-only requires B0 target embeddings within 2x DGUHA internal scales "
            "and target/descent logit-ranking AUROC at least 0.70 versus IWR normal."
        ),
        "interpretation": (
            "单调校准只能改变分数尺度，不能改变 embedding 支持域或样本排序；"
            "任一条件失败都更支持使用少量、严格视频对齐的 IWR 正样本做适配。"
        ),
    }


def _distance_table_rows(model_name: str, geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"model": model_name, "domain": domain, **metrics}
        for domain, metrics in geometry["domains"].items()
    ]


def _describe(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _prefixed_description(prefix: str, values: np.ndarray) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in _describe(values).items()}


def _log10_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return -np.logaddexp(0.0, -values) / np.log(10.0)


def _binary_auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _standardized_mean_difference(values: np.ndarray, reference: np.ndarray) -> float:
    pooled = np.sqrt((np.var(values) + np.var(reference)) / 2.0)
    if pooled < 1e-12:
        return 0.0 if abs(float(np.mean(values) - np.mean(reference))) < 1e-12 else float(
            np.sign(np.mean(values) - np.mean(reference)) * 1e6
        )
    return float((np.mean(values) - np.mean(reference)) / pooled)


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _leave_one_out_nearest_distance(values: np.ndarray) -> np.ndarray:
    distances = _pairwise_distances(values, values)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return 0.0 if np.allclose(left, right) else 1.0
    return float(1.0 - np.dot(left, right) / denominator)


def _verify_frozen_checkpoints(paths: Mapping[str, Path]) -> dict[str, str]:
    hashes = {name: _sha256(path) for name, path in paths.items()}
    for name, expected in FROZEN_CHECKPOINT_SHA256.items():
        if hashes[name] != expected:
            raise ValueError(f"frozen {name} checkpoint SHA256 mismatch")
    return hashes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_embeddings(
    path: Path, outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]
) -> None:
    payload: dict[str, np.ndarray] = {}
    for model, domains in outputs.items():
        for domain, values in domains.items():
            payload[f"{model}_{domain}_embedding"] = values["embedding"]
            payload[f"{model}_{domain}_score"] = values["score"].astype(np.float32)
            payload[f"{model}_{domain}_logit"] = values["logit"].astype(np.float32)
    np.savez_compressed(path, **payload)


def _plot_embedding_pca(
    outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], path: Path
) -> None:
    domains = (
        "dguha_prefall_validation",
        "iwr_normal",
        "iwr_fall_target_prefall",
        "iwr_fall_descent",
    )
    colors = {
        "dguha_prefall_validation": "#2166ac",
        "iwr_normal": "#999999",
        "iwr_fall_target_prefall": "#d73027",
        "iwr_fall_descent": "#fdae61",
    }
    labels = {
        "dguha_prefall_validation": "DGUHA pre-fall val",
        "iwr_normal": "IWR normal",
        "iwr_fall_target_prefall": "IWR target pre-fall",
        "iwr_fall_descent": "IWR descent",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    rng = np.random.default_rng(20260809)
    for axis, (model_name, model_values) in zip(axes, outputs.items()):
        sampled: dict[str, np.ndarray] = {}
        for domain in domains:
            values = model_values[domain]["embedding"].astype(np.float64)
            if domain == "iwr_normal" and len(values) > 500:
                values = values[rng.choice(len(values), 500, replace=False)]
            sampled[domain] = values
        combined = np.concatenate(list(sampled.values()))
        centered = combined - combined.mean(axis=0)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        projection = right[:2].T
        for domain, values in sampled.items():
            points = (values - combined.mean(axis=0)) @ projection
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=10 if "fall" in domain else 5,
                alpha=0.9 if "fall" in domain else 0.3,
                color=colors[domain],
                label=labels[domain],
            )
        axis.set_title(model_name)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Frozen TCN encoder domains (PCA is descriptive only)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_score_distributions(
    outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], path: Path
) -> None:
    domains = (
        "dguha_prefall_validation",
        "iwr_normal",
        "iwr_fall_target_prefall",
        "iwr_fall_descent",
    )
    labels = ("DGUHA val +", "IWR normal", "IWR pre-fall", "IWR descent")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    rng = np.random.default_rng(20260809)
    for axis, (model_name, model_values) in zip(axes, outputs.items()):
        plotted: list[np.ndarray] = []
        for domain in domains:
            logits = model_values[domain]["logit"]
            if domain == "iwr_normal" and len(logits) > 700:
                logits = logits[rng.choice(len(logits), 700, replace=False)]
            plotted.append(_log10_sigmoid(logits))
        axis.boxplot(plotted, tick_labels=labels, showfliers=False)
        for index, values in enumerate(plotted, start=1):
            jitter = rng.normal(index, 0.035, len(values))
            axis.scatter(jitter, values, s=5, alpha=0.18)
        axis.set_title(model_name)
        axis.tick_params(axis="x", rotation=28, labelsize=7)
        axis.set_ylabel("stable log10(score)")
    fig.suptitle("Stable score distributions from logits; calibration cannot change ordering")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_feature_shift(rows: list[dict[str, Any]], path: Path) -> None:
    domains = (
        "iwr_normal",
        "iwr_fall_target_prefall",
        "iwr_fall_descent",
    )
    statistics = ("window_mean", "temporal_slope")
    columns = [(domain, statistic) for domain in domains for statistic in statistics]
    values = np.zeros((len(FEATURE_NAMES_V2), len(columns)), dtype=np.float64)
    lookup = {
        (row["feature"], row["domain"], row["statistic"]): row[
            "standardized_mean_difference_vs_dguha_validation"
        ]
        for row in rows
    }
    for feature_index, feature in enumerate(FEATURE_NAMES_V2):
        for column_index, (domain, statistic) in enumerate(columns):
            values[feature_index, column_index] = lookup[(feature, domain, statistic)]
    clipped = np.clip(values, -5.0, 5.0)
    fig, axis = plt.subplots(figsize=(9.5, 7.5))
    image = axis.imshow(clipped, aspect="auto", cmap="coolwarm", vmin=-5, vmax=5)
    axis.set_yticks(np.arange(len(FEATURE_NAMES_V2)), FEATURE_NAMES_V2, fontsize=7)
    axis.set_xticks(
        np.arange(len(columns)),
        [f"{domain.replace('iwr_', '')}\n{statistic}" for domain, statistic in columns],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    axis.set_title("Raw feature standardized shift vs DGUHA validation positives")
    fig.colorbar(image, ax=axis, label="standardized mean difference (clipped ±5)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# 冻结 TCN embedding 域分析",
        "",
        f"分析版本：`{report['analysis_version']}`。未训练模型，未修改 checkpoint 或阈值，sealed test 未打开。",
        "",
        "## 样本",
        "",
    ]
    for domain, count in report["sample_counts"].items():
        lines.append(f"- `{domain}`：{count} 个窗口")
    lines.extend(
        [
            "",
            "IWR 跌倒主比较组严格匹配 DGUHA 标签：窗口结束于视频标注下降起点前 0.5–1.0 秒。下降段另行报告。",
            "",
            "## Embedding 与排序",
            "",
            "| 模型 | 目标窗最近邻/类内尺度 | 目标窗质心/类内半径 | 更靠近DGUHA占比 | 目标窗logit AUROC | 下降段logit AUROC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in ("B0", "M1", "M2"):
        model = report["models"][model_name]
        target_geometry = model["embedding_geometry"]["domains"][
            "iwr_fall_target_prefall"
        ]
        target_score = model["score_diagnostic"]["iwr_fall_target_prefall"]
        descent_score = model["score_diagnostic"]["iwr_fall_descent"]
        lines.append(
            f"| {model_name} | "
            f"{target_geometry['nearest_distance_over_dguha_loo_median']:.2f}x | "
            f"{target_geometry['centroid_distance_over_dguha_radius']:.2f}x | "
            f"{target_geometry['fraction_closer_to_dguha_than_iwr_normal']:.1%} | "
            f"{target_score['logit_ranking_auroc_vs_iwr_normal']:.3f} | "
            f"{descent_score['logit_ranking_auroc_vs_iwr_normal']:.3f} |"
        )
    lines.extend(
        [
            "",
            "M1/M2 的部署 float32 sigmoid 在 IWR 跌倒窗口已下溢为精确 0；表中 AUROC 使用下溢前的 logit，避免并列值掩盖真实排序。",
            "",
            "## 原始特征窗口均值",
            "",
            "| 特征 | DGUHA pre-fall val | IWR normal | IWR target pre-fall | IWR descent |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    feature_means = report["feature_shift_summary"]["primary_feature_window_means"]
    for feature, values in feature_means.items():
        lines.append(
            f"| {feature} | {values['dguha_prefall_validation']:.4g} | "
            f"{values['iwr_normal']:.4g} | {values['iwr_fall_target_prefall']:.4g} | "
            f"{values['iwr_fall_descent']:.4g} |"
        )
    feature_summary = report["feature_shift_summary"]
    lines.extend(
        [
            "",
            f"目标匹配 IWR pre-fall 在 mean/slope 的 "
            f"{feature_summary['iwr_target_prefall_effect_size_component_count']} 个比较项中，"
            f"有 {feature_summary['iwr_target_prefall_features_above_effect_size_1']} 个绝对效应量不小于 1。",
        ]
    )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## 判断",
            "",
            f"结论：`{decision['primary_decision']}`。",
            "",
            decision["interpretation"],
            "",
            "当前证据排除仅做单调 score 校准：校准不能把 IWR embedding 移入 DGUHA 正类支持域，也不能修复低于 normal 的 logit 排序。",
            "",
            "这是一例真实受控跌倒的描述性域诊断，只能决定下一步实验方向，不能据此估计总体召回率或适配后的收益。",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen B0/M1/M2 TCN embedding-domain analysis"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint-b0", required=True, type=Path)
    parser.add_argument("--checkpoint-m1", required=True, type=Path)
    parser.add_argument("--checkpoint-m2", required=True, type=Path)
    parser.add_argument("--normal-replay", required=True, type=Path)
    parser.add_argument("--fall-replay", required=True, type=Path)
    parser.add_argument("--phase-events", required=True, type=Path)
    parser.add_argument("--phase-compliance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_tcn_embedding_domains(
        dataset_path=args.dataset,
        checkpoints={
            "B0": args.checkpoint_b0,
            "M1": args.checkpoint_m1,
            "M2": args.checkpoint_m2,
        },
        normal_replay_path=args.normal_replay,
        fall_replay_path=args.fall_replay,
        phase_events_path=args.phase_events,
        phase_compliance_path=args.phase_compliance,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "analysis_version": report["analysis_version"],
                "decision": report["decision"],
                "sample_counts": report["sample_counts"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
