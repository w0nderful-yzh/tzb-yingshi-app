from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


DATASET_MODE = "DGUHA_TCN_HARD_NEGATIVE_V1"
EARLY_NEGATIVE_MINIMUM_LEAD_SECONDS = 1.2
POSITIVE_CONFIDENCE_WEIGHTS = {
    "stable": 1.25,
    "moderate": 0.85,
    "unstable": 0.50,
}
HARD_NEGATIVE_WEIGHTS = {
    "same_recording_early": 4.0,
    "fast_sit_proxy": 3.0,
    "sit_down_proxy": 2.0,
    "stand_up_proxy": 2.0,
    "sit_stand_other": 1.5,
    "squat_or_crouch": 2.0,
    "bend": 2.0,
    "floor_sit": 2.5,
    "lunge_lowering": 1.5,
}


def build_tcn_hard_negative_dataset(
    *,
    base_dguha_path: str | Path,
    early_dguha_path: str | Path,
    events_path: str | Path,
    output_path: str | Path,
    mmfall_path: str | Path | None = None,
    mmradpose_path: str | Path | None = None,
    radhar_path: str | Path | None = None,
    iwr6843_path: str | Path | None = None,
) -> dict[str, Any]:
    base_path = Path(base_dguha_path).resolve()
    early_path = Path(early_dguha_path).resolve()
    event_path = Path(events_path).resolve()
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    for source in (base_path, early_path, event_path):
        if not source.is_file():
            raise FileNotFoundError(f"required source does not exist: {source}")

    base = _load_feature_dataset(base_path)
    early_source = _load_feature_dataset(early_path)
    _validate_base_contract(base, early_source)
    events = json.loads(event_path.read_text(encoding="utf-8"))
    event_by_source = {str(event["source_file"]): event for event in events}

    sit_velocity_index = FEATURE_NAMES_V2.index("vertical_velocity")
    train_sit = (
        (base["split"] == "train")
        & (base["action"] == "SIT_STAND")
        & (base["labels"] == 0)
    )
    sit_velocities = base["features"][train_sit, -1, sit_velocity_index]
    if not len(sit_velocities):
        raise ValueError("base dataset has no training SIT_STAND negatives")
    fast_sit_cutoff = float(np.quantile(sit_velocities, 0.10))

    blocks: list[dict[str, np.ndarray]] = []
    base_categories = np.full(len(base["labels"]), "ordinary_negative", dtype="U32")
    base_weights = np.ones(len(base["labels"]), dtype=np.float32)
    base_confidence = np.full(len(base["labels"]), "not_applicable", dtype="U32")
    positive = base["labels"] == 1
    for index in np.flatnonzero(positive):
        source_file = str(base["source_files"][index])
        event = event_by_source.get(source_file)
        if event is None:
            raise ValueError(f"positive sample has no event metadata: {source_file}")
        confidence = _event_confidence(event)
        base_confidence[index] = confidence
        base_weights[index] = POSITIVE_CONFIDENCE_WEIGHTS[confidence]
        base_categories[index] = "positive"

    sit = (base["action"] == "SIT_STAND") & (base["labels"] == 0)
    sit_indices = np.flatnonzero(sit)
    sit_velocity = base["features"][sit_indices, -1, sit_velocity_index]
    fast = sit_velocity <= fast_sit_cutoff
    descending = (sit_velocity < 0.0) & ~fast
    rising = sit_velocity > 0.0
    for local_mask, category in (
        (fast, "fast_sit_proxy"),
        (descending, "sit_down_proxy"),
        (rising, "stand_up_proxy"),
        (~(fast | descending | rising), "sit_stand_other"),
    ):
        selected = sit_indices[local_mask]
        base_categories[selected] = category
        base_weights[selected] = HARD_NEGATIVE_WEIGHTS[category]
    blocks.append(
        _standardize_block(
            base,
            indices=np.arange(len(base["labels"])),
            origin="dguha",
            categories=base_categories,
            sample_weights=base_weights,
            label_confidence=base_confidence,
            normalization_reference=True,
        )
    )

    early_mask = (
        (early_source["labels"] == 1)
        & (early_source["seconds_to_onset"] >= EARLY_NEGATIVE_MINIMUM_LEAD_SECONDS - 1e-6)
    )
    early_indices = np.flatnonzero(early_mask)
    if not len(early_indices):
        raise ValueError("early source has no same-recording early windows")
    early_block = _standardize_block(
        early_source,
        indices=early_indices,
        origin="dguha",
        categories=np.full(len(early_indices), "same_recording_early"),
        sample_weights=np.full(
            len(early_indices), HARD_NEGATIVE_WEIGHTS["same_recording_early"], np.float32
        ),
        label_confidence=np.full(len(early_indices), "task_horizon_negative"),
        normalization_reference=True,
    )
    early_block["labels"][:] = 0
    early_block["label_source"][:] = "dguha_same_recording_early_negative"
    blocks.append(early_block)

    external_specs = (
        (mmfall_path, "mmfall_ds2", "train", {
            "bending": "bend",
            "crouching": "squat_or_crouch",
            "sitting_on_floor": "floor_sit",
        }),
        (mmradpose_path, "mmradpose", "train", {
            "squats": "squat_or_crouch",
            "torso_forward_bending": "bend",
            "left_front_lunge": "lunge_lowering",
            "right_front_lunge": "lunge_lowering",
        }),
        (radhar_path, "radhar", "external_train_pool", {
            "squats": "squat_or_crouch",
        }),
        (iwr6843_path, "iwr6843_fall_102", "train", {
            "bow": "bend",
            "squat": "squat_or_crouch",
        }),
    )
    input_paths = [base_path, early_path, event_path]
    for source_value, origin, allowed_split, mapping in external_specs:
        if source_value is None:
            continue
        source_path = Path(source_value).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"external source does not exist: {source_path}")
        external = _load_feature_dataset(source_path)
        action_key = "source_categories" if "source_categories" in external else "action"
        actions = external[action_key]
        mask = (external["split"] == allowed_split) & (external["labels"] == 0)
        mask &= np.isin(actions, tuple(mapping))
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"no selected hard negatives in {source_path}")
        categories = np.asarray([mapping[str(actions[index])] for index in indices])
        weights = np.asarray(
            [HARD_NEGATIVE_WEIGHTS[str(category)] for category in categories],
            dtype=np.float32,
        )
        block = _standardize_block(
            external,
            indices=indices,
            origin=origin,
            categories=categories,
            sample_weights=weights,
            label_confidence=np.full(len(indices), "recording_level_negative"),
            normalization_reference=False,
            force_split="train",
        )
        blocks.append(block)
        input_paths.append(source_path)

    arrays = {
        key: np.concatenate([block[key] for block in blocks])
        for key in blocks[0]
    }
    _assert_strict_group_split(arrays["event_id"], arrays["split"])
    _assert_dguha_subject_split(
        arrays["subject_id"][arrays["dataset_origin"] == "dguha"],
        arrays["split"][arrays["dataset_origin"] == "dguha"],
    )
    if np.any(arrays["labels"][arrays["dataset_origin"] != "dguha"] != 0):
        raise ValueError("external sources must contribute negatives only")

    payload = dict(arrays)
    payload.update(
        {
            "feature_version": np.asarray(FEATURE_VERSION_V2),
            "feature_names": np.asarray(FEATURE_NAMES_V2),
            "dataset_mode": np.asarray(DATASET_MODE),
            "positive_anchor": np.asarray("descent_onset"),
            "prediction_horizon_seconds": np.asarray((0.5, 1.0), np.float32),
            "deployment_eligible": np.asarray(False),
            "fast_sit_proxy_definition": np.asarray(
                "DGUHA SIT_STAND windows at or below the training-only 10th percentile "
                "of final-step vertical_velocity; proxy label, not an authored action label"
            ),
            "fast_sit_proxy_cutoff": np.asarray(fast_sit_cutoff, np.float32),
            "missing_requested_categories": np.asarray(
                ("authored_fast_sit", "kneel", "authored_sit_vs_rise_phase")
            ),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    labels = arrays["labels"]
    splits = arrays["split"]
    categories, category_counts = np.unique(
        arrays["hard_negative_category"], return_counts=True
    )
    confidence, confidence_counts = np.unique(
        arrays["label_confidence"], return_counts=True
    )
    report = {
        "dataset_mode": DATASET_MODE,
        "output_file": str(destination),
        "output_sha256": _sha256(destination),
        "input_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in input_paths
        ],
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "split_counts": {
            split: int(np.sum(splits == split))
            for split in ("train", "validation", "test")
        },
        "hard_negative_category_counts": dict(
            zip(categories.tolist(), category_counts.astype(int).tolist())
        ),
        "label_confidence_counts": dict(
            zip(confidence.tolist(), confidence_counts.astype(int).tolist())
        ),
        "sample_weight_by_category": HARD_NEGATIVE_WEIGHTS,
        "positive_confidence_weights": POSITIVE_CONFIDENCE_WEIGHTS,
        "event_confidence_definition": {
            "stable": "rapid_descent_onset - descent_onset <= 0.30 s",
            "moderate": "difference > 0.30 s and <= 0.75 s",
            "unstable": "difference > 0.75 s or timing unavailable",
        },
        "fast_sit_proxy_cutoff": fast_sit_cutoff,
        "strict_group_split_check": "passed: every dataset_origin::source_file belongs to one split",
        "strict_dguha_subject_split_check": "passed",
        "external_sources_used_for_training_only": True,
        "requested_category_limitations": [
            "No authored fast-sit label; a training-only velocity-tail proxy is identified inside DGUHA SIT_STAND.",
            "No kneeling label is available and sitting_on_floor is not relabelled as kneeling.",
            "DGUHA combines sitting down and standing up in one recording; phase names are radar-derived proxies only.",
        ],
        "test_split_inspected_for_metrics": False,
        "deployment_eligible": False,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _load_feature_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        required = {"features", "labels", "split", "source_files", "feature_version", "feature_names"}
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(f"dataset is missing fields {sorted(missing)}: {path}")
        if str(dataset["feature_version"].item()) != FEATURE_VERSION_V2:
            raise ValueError(f"incompatible feature version: {path}")
        if tuple(str(value) for value in dataset["feature_names"]) != FEATURE_NAMES_V2:
            raise ValueError(f"incompatible feature order: {path}")
        result = {name: np.asarray(dataset[name]) for name in dataset.files}
    features = np.asarray(result["features"], dtype=np.float32)
    if features.shape[1:] != (WINDOW_SIZE_V2, len(FEATURE_NAMES_V2)):
        raise ValueError(f"incompatible feature tensor: {path}")
    result["features"] = features
    result["labels"] = np.asarray(result["labels"], dtype=np.int8)
    return result


def _validate_base_contract(base: dict[str, np.ndarray], early: dict[str, np.ndarray]) -> None:
    for dataset, name in ((base, "base"), (early, "early")):
        for key in ("action", "subject_id", "seconds_to_onset", "label_source", "data_quality"):
            if key not in dataset:
                raise ValueError(f"{name} DGUHA dataset is missing {key}")
    horizon = tuple(float(value) for value in np.asarray(base["prediction_horizon_seconds"]))
    if not np.allclose(horizon, (0.5, 1.0), atol=1e-6):
        raise ValueError("base DGUHA horizon must be 0.5-1.0 s")
    early_horizon = tuple(
        float(value) for value in np.asarray(early["prediction_horizon_seconds"])
    )
    if early_horizon[1] < EARLY_NEGATIVE_MINIMUM_LEAD_SECONDS:
        raise ValueError("early DGUHA source does not cover early negatives")


def _event_confidence(event: dict[str, Any]) -> str:
    try:
        onset = datetime.fromisoformat(str(event["descent_onset"]))
        rapid = datetime.fromisoformat(str(event["rapid_descent_onset"]))
    except (KeyError, TypeError, ValueError):
        return "unstable"
    difference = abs((rapid - onset).total_seconds())
    if difference <= 0.30 + 1e-9:
        return "stable"
    if difference <= 0.75 + 1e-9:
        return "moderate"
    return "unstable"


def _standardize_block(
    source: dict[str, np.ndarray],
    *,
    indices: np.ndarray,
    origin: str,
    categories: np.ndarray,
    sample_weights: np.ndarray,
    label_confidence: np.ndarray,
    normalization_reference: bool,
    force_split: str | None = None,
) -> dict[str, np.ndarray]:
    count = len(indices)
    source_files = np.asarray(source["source_files"])[indices].astype("U256")
    split = (
        np.full(count, force_split, dtype="U32")
        if force_split is not None
        else np.asarray(source["split"])[indices].astype("U32")
    )
    action_key = "action" if "action" in source else "source_categories"
    action = np.asarray(source[action_key])[indices].astype("U64")
    subject = (
        np.asarray(source["subject_id"])[indices].astype("U64")
        if "subject_id" in source
        else np.full(count, "unknown", dtype="U64")
    )
    return {
        "features": np.asarray(source["features"])[indices].astype(np.float32, copy=False),
        "labels": np.asarray(source["labels"])[indices].astype(np.int8, copy=True),
        "split": split,
        "action": action,
        "subject_id": subject,
        "source_files": source_files,
        "event_id": np.asarray([f"{origin}::{value}" for value in source_files]),
        "dataset_origin": np.full(count, origin, dtype="U32"),
        "hard_negative_category": np.asarray(categories).astype("U32"),
        "sample_weight": np.asarray(sample_weights, dtype=np.float32),
        "label_confidence": np.asarray(label_confidence).astype("U32"),
        "normalization_reference": np.full(count, normalization_reference, dtype=bool),
        "window_end_seconds": _optional_numeric(source, "window_end_seconds", indices),
        "seconds_to_onset": _optional_numeric(source, "seconds_to_onset", indices),
        "seconds_to_anchor": _optional_numeric(source, "seconds_to_anchor", indices),
        "label_source": _optional_text(source, "label_source", indices, "recording_level_negative"),
        "data_quality": _optional_text(source, "data_quality", indices, "unknown"),
    }


def _optional_numeric(
    source: dict[str, np.ndarray], key: str, indices: np.ndarray
) -> np.ndarray:
    if key not in source:
        return np.full(len(indices), np.nan, dtype=np.float32)
    return np.asarray(source[key])[indices].astype(np.float32)


def _optional_text(
    source: dict[str, np.ndarray], key: str, indices: np.ndarray, default: str
) -> np.ndarray:
    if key not in source:
        return np.full(len(indices), default, dtype="U64")
    return np.asarray(source[key])[indices].astype("U64")


def _assert_strict_group_split(event_ids: np.ndarray, splits: np.ndarray) -> None:
    assignments: dict[str, str] = {}
    for event_id, split in zip(event_ids, splits, strict=True):
        previous = assignments.setdefault(str(event_id), str(split))
        if previous != str(split):
            raise ValueError(f"event/source group crosses splits: {event_id}")


def _assert_dguha_subject_split(subjects: np.ndarray, splits: np.ndarray) -> None:
    assignments: dict[str, str] = {}
    for subject, split in zip(subjects, splits, strict=True):
        previous = assignments.setdefault(str(subject), str(split))
        if previous != str(split):
            raise ValueError(f"DGUHA subject crosses splits: {subject}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TCN hard-negative dataset.")
    parser.add_argument("--base-dguha", required=True, type=Path)
    parser.add_argument("--early-dguha", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mmfall", type=Path)
    parser.add_argument("--mmradpose", type=Path)
    parser.add_argument("--radhar", type=Path)
    parser.add_argument("--iwr6843", type=Path)
    args = parser.parse_args()
    report = build_tcn_hard_negative_dataset(
        base_dguha_path=args.base_dguha,
        early_dguha_path=args.early_dguha,
        events_path=args.events,
        output_path=args.output,
        mmfall_path=args.mmfall,
        mmradpose_path=args.mmradpose,
        radhar_path=args.radhar,
        iwr6843_path=args.iwr6843,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
