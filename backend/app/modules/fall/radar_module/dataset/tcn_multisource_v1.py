from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radar_module.dataset.dguha_research_v2 import DGUHA_SPLIT_BY_SUBJECT
from radar_module.dataset.tcn_hard_negative_v1 import (
    EARLY_NEGATIVE_MINIMUM_LEAD_SECONDS,
    HARD_NEGATIVE_WEIGHTS,
    POSITIVE_CONFIDENCE_WEIGHTS,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


DATASET_MODE = "TCN_MULTISOURCE_NORMAL_DOMAIN_V1"
EXTERNAL_ORIGINS = (
    "mmradpose",
    "radhar",
    "mmwave_ocpid",
    "iwr6843_fall_102",
)


def build_tcn_multisource_dataset(
    *,
    base_dguha_path: str | Path,
    early_dguha_path: str | Path,
    events_path: str | Path,
    mmradpose_path: str | Path,
    radhar_path: str | Path,
    ocpid_path: str | Path,
    iwr6843_path: str | Path,
    output_path: str | Path,
    augmented_paths: Mapping[str, str | Path] | None = None,
    augmentation_spec: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build B0-compatible M1/M2 data without any external positive label."""

    paths = {
        "dguha_base": Path(base_dguha_path).resolve(),
        "dguha_early": Path(early_dguha_path).resolve(),
        "mmradpose": Path(mmradpose_path).resolve(),
        "radhar": Path(radhar_path).resolve(),
        "mmwave_ocpid": Path(ocpid_path).resolve(),
        "iwr6843_fall_102": Path(iwr6843_path).resolve(),
    }
    event_file = Path(events_path).resolve()
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")
    for path in (*paths.values(), event_file):
        if not path.is_file():
            raise FileNotFoundError(f"required source does not exist: {path}")

    clean = {name: _load_feature_dataset(path) for name, path in paths.items()}
    augmented: dict[str, dict[str, np.ndarray]] = {}
    for name, source_value in (augmented_paths or {}).items():
        if name not in paths:
            raise ValueError(f"unknown augmented source: {name}")
        augmented_path = Path(source_value).resolve()
        if not augmented_path.is_file():
            raise FileNotFoundError(f"augmented source does not exist: {augmented_path}")
        candidate = _load_feature_dataset(augmented_path)
        _assert_parallel_export(clean[name], candidate, name)
        augmented[name] = candidate

    events = json.loads(event_file.read_text(encoding="utf-8"))
    event_by_source = {str(event["source_file"]): event for event in events}
    base = clean["dguha_base"]
    early = clean["dguha_early"]
    _validate_dguha_contract(base, early)

    sit_velocity_index = FEATURE_NAMES_V2.index("vertical_velocity")
    train_sit = (
        (base["split"] == "train")
        & (base["action"] == "SIT_STAND")
        & (base["labels"] == 0)
    )
    sit_velocities = base["features"][train_sit, -1, sit_velocity_index]
    if not len(sit_velocities):
        raise ValueError("DGUHA training split has no SIT_STAND negatives")
    fast_sit_cutoff = float(np.quantile(sit_velocities, 0.10))

    blocks: list[dict[str, np.ndarray]] = []
    base_categories = np.full(len(base["labels"]), "ordinary_negative", dtype="U48")
    base_weights = np.ones(len(base["labels"]), dtype=np.float32)
    base_confidence = np.full(len(base["labels"]), "not_applicable", dtype="U32")
    for index in np.flatnonzero(base["labels"] == 1):
        source_file = str(base["source_files"][index])
        event = event_by_source.get(source_file)
        if event is None:
            raise ValueError(f"positive DGUHA sample has no event: {source_file}")
        confidence = _event_confidence(event)
        base_categories[index] = "positive"
        base_confidence[index] = confidence
        base_weights[index] = POSITIVE_CONFIDENCE_WEIGHTS[confidence]

    sit_indices = np.flatnonzero(
        (base["action"] == "SIT_STAND") & (base["labels"] == 0)
    )
    final_velocity = base["features"][sit_indices, -1, sit_velocity_index]
    fast = final_velocity <= fast_sit_cutoff
    descending = (final_velocity < 0.0) & ~fast
    rising = final_velocity > 0.0
    for mask, category in (
        (fast, "fast_sit_proxy"),
        (descending, "sit_down_proxy"),
        (rising, "stand_up_proxy"),
        (~(fast | descending | rising), "sit_stand_other"),
    ):
        selected = sit_indices[mask]
        base_categories[selected] = category
        base_weights[selected] = HARD_NEGATIVE_WEIGHTS[category]

    base_indices = np.arange(len(base["labels"]))
    blocks.append(
        _standardize_block(
            base,
            indices=base_indices,
            origin="dguha",
            split_values=np.asarray(base["split"]).astype("U32"),
            categories=base_categories,
            sample_weights=base_weights,
            label_confidence=base_confidence,
            feature_values=_training_feature_choice(
                base,
                augmented.get("dguha_base"),
                base_indices,
                np.asarray(base["split"]),
            ),
        )
    )

    early_mask = (
        (early["labels"] == 1)
        & (
            early["seconds_to_onset"]
            >= EARLY_NEGATIVE_MINIMUM_LEAD_SECONDS - 1e-6
        )
    )
    early_indices = np.flatnonzero(early_mask)
    if not len(early_indices):
        raise ValueError("DGUHA early artifact has no same-recording negatives")
    early_splits = np.asarray(early["split"])[early_indices].astype("U32")
    early_block = _standardize_block(
        early,
        indices=early_indices,
        origin="dguha",
        split_values=early_splits,
        categories=np.full(len(early_indices), "same_recording_early"),
        sample_weights=np.full(
            len(early_indices), HARD_NEGATIVE_WEIGHTS["same_recording_early"], np.float32
        ),
        label_confidence=np.full(len(early_indices), "task_horizon_negative"),
        feature_values=_training_feature_choice(
            early,
            augmented.get("dguha_early"),
            early_indices,
            early_splits,
        ),
    )
    early_block["labels"][:] = 0
    early_block["label_source"][:] = "dguha_same_recording_early_negative"
    blocks.append(early_block)

    radhar_validation_files = _radhar_validation_files(clean["radhar"])
    external_counts: dict[str, dict[str, int]] = {}
    for origin in EXTERNAL_ORIGINS:
        source = clean[origin]
        labels = np.asarray(source["labels"], dtype=np.int8)
        indices = np.flatnonzero(labels == 0)
        if not len(indices):
            raise ValueError(f"{origin} has no normal samples")
        mapped_splits = _map_external_splits(
            origin,
            np.asarray(source["split"])[indices],
            np.asarray(source["source_files"])[indices],
            radhar_validation_files=radhar_validation_files,
        )
        actions = _actions(source)[indices]
        categories = np.asarray(
            [_external_category(origin, str(action)) for action in actions]
        )
        weights = np.asarray(
            [_external_weight(origin, str(action)) for action in actions],
            dtype=np.float32,
        )
        block = _standardize_block(
            source,
            indices=indices,
            origin=origin,
            split_values=mapped_splits,
            categories=categories,
            sample_weights=weights,
            label_confidence=np.full(len(indices), "recording_level_negative"),
            feature_values=_training_feature_choice(
                source,
                augmented.get(origin),
                indices,
                mapped_splits,
            ),
        )
        blocks.append(block)
        external_counts[origin] = {
            split: int(np.sum(mapped_splits == split))
            for split in ("train", "external_validation", "external_test")
        }

    arrays = {
        key: np.concatenate([block[key] for block in blocks])
        for key in blocks[0]
    }
    _assert_event_split(arrays["event_id"], arrays["split"])
    _assert_dguha_subject_split(
        arrays["subject_id"][arrays["dataset_origin"] == "dguha"],
        arrays["split"][arrays["dataset_origin"] == "dguha"],
    )
    external_mask = arrays["dataset_origin"] != "dguha"
    if np.any(arrays["labels"][external_mask] != 0):
        raise ValueError("external sources must be negative-only")
    if set(np.unique(arrays["dataset_origin"][arrays["labels"] == 1])) != {"dguha"}:
        raise ValueError("DGUHA must be the only positive source")

    clean_normalization_features = np.concatenate(
        (
            base["features"][base["split"] == "train"],
            early["features"][early_mask & (early["split"] == "train")],
        )
    )
    normalization_mean = clean_normalization_features.mean(
        axis=(0, 1), dtype=np.float64
    ).astype(np.float32)
    normalization_std = clean_normalization_features.std(
        axis=(0, 1), dtype=np.float64
    ).astype(np.float32)
    normalization_std = np.where(normalization_std < 1e-6, 1.0, normalization_std)

    payload = dict(arrays)
    payload.update(
        {
            "feature_version": np.asarray(FEATURE_VERSION_V2),
            "feature_names": np.asarray(FEATURE_NAMES_V2),
            "dataset_mode": np.asarray(DATASET_MODE),
            "positive_anchor": np.asarray("descent_onset"),
            "prediction_horizon_seconds": np.asarray((0.5, 1.0), np.float32),
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
            "normalization_reference_count": np.asarray(
                len(clean_normalization_features), dtype=np.int64
            ),
            "augmentation_enabled": np.asarray(bool(augmented)),
            "deployment_eligible": np.asarray(False),
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

    split_counts = {
        split: int(np.sum(arrays["split"] == split))
        for split in (
            "train",
            "validation",
            "test",
            "external_validation",
            "external_test",
        )
    }
    origin_values, origin_sizes = np.unique(arrays["dataset_origin"], return_counts=True)
    report = {
        "dataset_mode": DATASET_MODE,
        "output_file": str(destination),
        "output_sha256": _sha256(destination),
        "input_files": [
            {"name": name, "path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        ]
        + [{"name": "events", "path": str(event_file), "sha256": _sha256(event_file)}],
        "augmented_input_files": [
            {
                "name": name,
                "path": str(Path(value).resolve()),
                "sha256": _sha256(Path(value).resolve()),
            }
            for name, value in (augmented_paths or {}).items()
        ],
        "augmentation_enabled": bool(augmented),
        "augmentation_spec": dict(augmentation_spec or {}),
        "sample_count": int(len(arrays["labels"])),
        "positive_count": int(arrays["labels"].sum()),
        "negative_count": int(len(arrays["labels"]) - arrays["labels"].sum()),
        "positive_sources": ["dguha"],
        "origin_counts": dict(zip(origin_values.tolist(), origin_sizes.astype(int).tolist())),
        "split_counts": split_counts,
        "external_split_counts": external_counts,
        "normalization_policy": "clean DGUHA training windows only",
        "normalization_reference_count": int(len(clean_normalization_features)),
        "fast_sit_proxy_cutoff": fast_sit_cutoff,
        "strict_event_split_check": "passed",
        "strict_dguha_subject_split_check": "passed",
        "mmfall_used": False,
        "test_split_evaluated": False,
        "deployment_eligible": False,
        "limitations": [
            "RadHAR has no reliable subject identifiers; its split is recording-level only.",
            "IWR6843 fall recordings are excluded because they have no pre-fall time anchor.",
            "All external sources are normal-action negatives and cannot add IWR6843-domain positives.",
        ],
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _load_feature_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "features",
            "labels",
            "split",
            "source_files",
            "feature_version",
            "feature_names",
        }
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(f"dataset missing {sorted(missing)}: {path}")
        result = {name: np.asarray(dataset[name]) for name in dataset.files}
    if str(result["feature_version"].item()) != FEATURE_VERSION_V2:
        raise ValueError(f"incompatible feature version: {path}")
    if tuple(map(str, result["feature_names"])) != FEATURE_NAMES_V2:
        raise ValueError(f"incompatible feature order: {path}")
    features = np.asarray(result["features"], dtype=np.float32)
    if features.shape[1:] != (WINDOW_SIZE_V2, len(FEATURE_NAMES_V2)):
        raise ValueError(f"incompatible feature shape: {path}")
    result["features"] = features
    result["labels"] = np.asarray(result["labels"], dtype=np.int8)
    return result


def _assert_parallel_export(
    clean: dict[str, np.ndarray],
    augmented: dict[str, np.ndarray],
    name: str,
) -> None:
    if clean["features"].shape != augmented["features"].shape:
        raise ValueError(f"parallel export shape mismatch: {name}")
    for key in ("labels", "split", "source_files"):
        if not np.array_equal(clean[key], augmented[key]):
            raise ValueError(f"parallel export {key} mismatch: {name}")
    if "window_end_seconds" in clean and not np.allclose(
        clean["window_end_seconds"],
        augmented["window_end_seconds"],
        equal_nan=True,
    ):
        raise ValueError(f"parallel export window endpoints mismatch: {name}")


def _training_feature_choice(
    clean: dict[str, np.ndarray],
    augmented: dict[str, np.ndarray] | None,
    indices: np.ndarray,
    mapped_splits: np.ndarray,
) -> np.ndarray:
    values = np.asarray(clean["features"])[indices].astype(np.float32, copy=True)
    if augmented is None:
        return values
    training = np.asarray(mapped_splits) == "train"
    values[training] = np.asarray(augmented["features"])[indices[training]]
    return values


def _validate_dguha_contract(
    base: dict[str, np.ndarray], early: dict[str, np.ndarray]
) -> None:
    for dataset, label in ((base, "base"), (early, "early")):
        for key in (
            "action",
            "subject_id",
            "seconds_to_onset",
            "label_source",
            "data_quality",
        ):
            if key not in dataset:
                raise ValueError(f"{label} DGUHA artifact missing {key}")
    if not np.allclose(base["prediction_horizon_seconds"], (0.5, 1.0)):
        raise ValueError("base DGUHA horizon must be 0.5-1.0 seconds")


def _standardize_block(
    source: dict[str, np.ndarray],
    *,
    indices: np.ndarray,
    origin: str,
    split_values: np.ndarray,
    categories: np.ndarray,
    sample_weights: np.ndarray,
    label_confidence: np.ndarray,
    feature_values: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(indices)
    source_files = np.asarray(source["source_files"])[indices].astype("U256")
    subjects = (
        np.asarray(source["subject_id"])[indices].astype("U64")
        if "subject_id" in source
        else np.full(count, "unknown", dtype="U64")
    )
    actions = _actions(source)[indices].astype("U64")
    return {
        "features": np.asarray(feature_values, dtype=np.float32),
        "labels": np.asarray(source["labels"])[indices].astype(np.int8, copy=True),
        "split": np.asarray(split_values).astype("U32"),
        "action": actions,
        "subject_id": subjects,
        "source_files": source_files,
        "event_id": np.asarray([f"{origin}::{value}" for value in source_files]),
        "dataset_origin": np.full(count, origin, dtype="U32"),
        "hard_negative_category": np.asarray(categories).astype("U48"),
        "sample_weight": np.asarray(sample_weights, dtype=np.float32),
        "label_confidence": np.asarray(label_confidence).astype("U32"),
        "window_end_seconds": _optional_numeric(source, "window_end_seconds", indices),
        "seconds_to_onset": _optional_numeric(source, "seconds_to_onset", indices),
        "seconds_to_anchor": _optional_numeric(source, "seconds_to_anchor", indices),
        "label_source": _optional_text(
            source, "label_source", indices, "recording_level_negative"
        ),
        "data_quality": _optional_text(source, "data_quality", indices, "unknown"),
    }


def _actions(source: dict[str, np.ndarray]) -> np.ndarray:
    key = "source_categories" if "source_categories" in source else "action"
    return np.asarray(source[key])


def _optional_numeric(
    source: dict[str, np.ndarray], key: str, indices: np.ndarray
) -> np.ndarray:
    if key in source:
        return np.asarray(source[key])[indices].astype(np.float32)
    return np.full(len(indices), np.nan, dtype=np.float32)


def _optional_text(
    source: dict[str, np.ndarray], key: str, indices: np.ndarray, default: str
) -> np.ndarray:
    if key in source:
        return np.asarray(source[key])[indices].astype("U96")
    return np.full(len(indices), default, dtype="U96")


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


def _radhar_validation_files(source: dict[str, np.ndarray]) -> set[str]:
    train_files = sorted(
        set(
            str(value)
            for value in source["source_files"][source["split"] == "external_train_pool"]
        ),
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(train_files) < 2:
        raise ValueError("RadHAR training pool is too small for recording validation")
    validation_count = max(1, int(round(len(train_files) * 0.20)))
    return set(train_files[:validation_count])


def _map_external_splits(
    origin: str,
    source_splits: np.ndarray,
    source_files: np.ndarray,
    *,
    radhar_validation_files: set[str],
) -> np.ndarray:
    result: list[str] = []
    for split, source_file in zip(source_splits, source_files):
        split_value = str(split)
        if origin == "radhar":
            if split_value == "external_test":
                result.append("external_test")
            elif str(source_file) in radhar_validation_files:
                result.append("external_validation")
            else:
                result.append("train")
        elif split_value in {"train", "external_train_pool"}:
            result.append("train")
        elif split_value in {"validation", "external_validation"}:
            result.append("external_validation")
        elif split_value in {"test", "external_test"}:
            result.append("external_test")
        else:
            raise ValueError(f"unsupported {origin} split: {split_value}")
    return np.asarray(result, dtype="U32")


def _external_category(origin: str, action: str) -> str:
    if origin == "mmwave_ocpid":
        return "occluded_walk"
    return f"{origin}:{action}"


def _external_weight(origin: str, action: str) -> float:
    if origin == "iwr6843_fall_102":
        return 2.5
    if action in {
        "squats",
        "squat",
        "torso_forward_bending",
        "bow",
        "left_front_lunge",
        "right_front_lunge",
        "jump",
    }:
        return 2.0
    if origin == "mmwave_ocpid":
        return 1.5
    return 1.5


def _assert_event_split(event_ids: np.ndarray, splits: np.ndarray) -> None:
    seen: dict[str, str] = {}
    for event, split in zip(event_ids, splits):
        previous = seen.setdefault(str(event), str(split))
        if previous != str(split):
            raise ValueError(f"event crosses splits: {event}")


def _assert_dguha_subject_split(subjects: np.ndarray, splits: np.ndarray) -> None:
    for subject, split in zip(subjects, splits):
        expected = DGUHA_SPLIT_BY_SUBJECT.get(str(subject))
        if expected is None or expected != str(split):
            raise ValueError(f"DGUHA subject split mismatch: {subject} -> {split}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build negative-only multisource TCN data")
    parser.add_argument("--dguha-base", required=True, type=Path)
    parser.add_argument("--dguha-early", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--mmradpose", required=True, type=Path)
    parser.add_argument("--radhar", required=True, type=Path)
    parser.add_argument("--ocpid", required=True, type=Path)
    parser.add_argument("--iwr6843", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--augmented-dguha-base", type=Path)
    parser.add_argument("--augmented-dguha-early", type=Path)
    parser.add_argument("--augmented-mmradpose", type=Path)
    parser.add_argument("--augmented-radhar", type=Path)
    parser.add_argument("--augmented-ocpid", type=Path)
    parser.add_argument("--augmented-iwr6843", type=Path)
    parser.add_argument("--augmentation-report", type=Path)
    args = parser.parse_args()
    augmented_values = {
        "dguha_base": args.augmented_dguha_base,
        "dguha_early": args.augmented_dguha_early,
        "mmradpose": args.augmented_mmradpose,
        "radhar": args.augmented_radhar,
        "mmwave_ocpid": args.augmented_ocpid,
        "iwr6843_fall_102": args.augmented_iwr6843,
    }
    supplied = {name: value for name, value in augmented_values.items() if value is not None}
    if supplied and len(supplied) != len(augmented_values):
        raise ValueError("all six augmented source paths must be supplied together")
    augmentation_spec: dict[str, object] | None = None
    if args.augmentation_report is not None:
        augmentation_document = json.loads(
            args.augmentation_report.read_text(encoding="utf-8")
        )
        augmentation_spec = dict(augmentation_document["augmentation_spec"])
    report = build_tcn_multisource_dataset(
        base_dguha_path=args.dguha_base,
        early_dguha_path=args.dguha_early,
        events_path=args.events,
        mmradpose_path=args.mmradpose,
        radhar_path=args.radhar,
        ocpid_path=args.ocpid,
        iwr6843_path=args.iwr6843,
        output_path=args.output,
        augmented_paths=supplied or None,
        augmentation_spec=augmentation_spec,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
