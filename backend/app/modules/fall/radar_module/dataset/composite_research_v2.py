from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from radar_module.dataset.dguha_research_v2 import DGUHA_DATASET_MODE
from radar_module.dataset.iwr6843_fall_v1 import DATASET_MODE as IWR6843_DATASET_MODE
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    WINDOW_SIZE_V2,
)


RESEARCH_MODEL_MODE = "RESEARCH_WEAK_SUPERVISION"


@dataclass(frozen=True, slots=True)
class CompositeResearchSummary:
    output_file: str
    output_sha256: str
    mmfall_file: str
    dguha_file: str
    mmradpose_file: str
    radhar_file: str | None
    iwr6843_file: str | None
    sample_count: int
    positive_count: int
    negative_count: int
    train_count: int
    validation_count: int
    test_count: int
    mmfall_count: int
    dguha_count: int
    mmradpose_train_negative_count: int
    radhar_train_negative_count: int
    iwr6843_train_negative_count: int
    feature_version: str
    window_size: int
    input_size: int
    deployment_eligible: bool


def build_composite_research_npz(
    mmfall_path: str | Path,
    dguha_path: str | Path,
    mmradpose_path: str | Path,
    output_path: str | Path,
    *,
    radhar_path: str | Path | None = None,
    iwr6843_path: str | Path | None = None,
    max_mmradpose_train_negatives: int = 4000,
) -> CompositeResearchSummary:
    if max_mmradpose_train_negatives <= 0:
        raise ValueError("max_mmradpose_train_negatives must be positive")
    mmfall_file = Path(mmfall_path).resolve()
    dguha_file = Path(dguha_path).resolve()
    mmradpose_file = Path(mmradpose_path).resolve()
    radhar_file = Path(radhar_path).resolve() if radhar_path is not None else None
    iwr6843_file = (
        Path(iwr6843_path).resolve() if iwr6843_path is not None else None
    )
    destination = Path(output_path).resolve()
    inputs = [mmfall_file, dguha_file, mmradpose_file]
    if radhar_file is not None:
        inputs.append(radhar_file)
    if iwr6843_file is not None:
        inputs.append(iwr6843_file)
    for source in inputs:
        if not source.is_file():
            raise FileNotFoundError(f"component dataset does not exist: {source}")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end with .npz")

    mmfall = _load_component(
        mmfall_file,
        expected_mode=RESEARCH_MODEL_MODE,
        require_both_labels=True,
    )
    dguha = _load_component(
        dguha_file,
        expected_mode=DGUHA_DATASET_MODE,
        require_both_labels=True,
    )
    mmradpose = _load_component(
        mmradpose_file,
        expected_mode="EXTERNAL_HARD_NEGATIVE_ONLY",
        require_both_labels=False,
    )
    if np.any(mmradpose["labels"] != 0):
        raise ValueError("mmRadPose component must contain negatives only")
    if not bool(mmradpose["source_complete"]):
        raise ValueError("mmRadPose component must be the complete published release")
    train_indices = np.flatnonzero(mmradpose["split"] == "train")
    if not len(train_indices):
        raise ValueError("mmRadPose component has no training subjects")
    if len(train_indices) > max_mmradpose_train_negatives:
        positions = np.linspace(
            0,
            len(train_indices) - 1,
            max_mmradpose_train_negatives,
            dtype=np.int64,
        )
        train_indices = train_indices[positions]

    radhar: dict[str, np.ndarray | bool] | None = None
    radhar_train_indices = np.asarray([], dtype=np.int64)
    if radhar_file is not None:
        radhar = _load_component(
            radhar_file,
            expected_mode="EXTERNAL_HARD_NEGATIVE_ONLY",
            require_both_labels=False,
        )
        if np.any(radhar["labels"] != 0):
            raise ValueError("RadHAR component must contain negatives only")
        radhar_train_indices = np.flatnonzero(
            radhar["split"] == "external_train_pool"
        )
        if not len(radhar_train_indices):
            raise ValueError("RadHAR component has no external training pool")

    iwr6843: dict[str, np.ndarray | bool] | None = None
    iwr6843_train_indices = np.asarray([], dtype=np.int64)
    if iwr6843_file is not None:
        iwr6843 = _load_component(
            iwr6843_file,
            expected_mode=IWR6843_DATASET_MODE,
            require_both_labels=True,
        )
        iwr6843_train_indices = np.flatnonzero(
            (iwr6843["split"] == "train") & (iwr6843["labels"] == 0)
        )
        if not len(iwr6843_train_indices):
            raise ValueError("IWR6843 component has no training-subject negatives")

    feature_blocks = [
        mmfall["features"],
        dguha["features"],
        mmradpose["features"][train_indices],
    ]
    label_blocks = [
        mmfall["labels"],
        dguha["labels"],
        mmradpose["labels"][train_indices],
    ]
    split_blocks = [
        mmfall["split"],
        dguha["split"],
        np.full(len(train_indices), "train"),
    ]
    origin_blocks = [
        np.full(len(mmfall["labels"]), "mmfall_ds2"),
        np.full(len(dguha["labels"]), "dguha"),
        np.full(len(train_indices), "mmradpose"),
    ]
    source_blocks = [
        mmfall["source_files"],
        dguha["source_files"],
        mmradpose["source_files"][train_indices],
    ]
    activity_blocks = [
        mmfall["source_categories"],
        dguha["action"],
        mmradpose["action"][train_indices],
    ]
    label_source_blocks = [
        mmfall["label_source"],
        dguha["label_source"],
        np.full(len(train_indices), "mmradpose_recording_activity_label"),
    ]
    if radhar is not None:
        feature_blocks.append(radhar["features"][radhar_train_indices])
        label_blocks.append(radhar["labels"][radhar_train_indices])
        split_blocks.append(np.full(len(radhar_train_indices), "train"))
        origin_blocks.append(np.full(len(radhar_train_indices), "radhar"))
        source_blocks.append(radhar["source_files"][radhar_train_indices])
        activity_blocks.append(radhar["action"][radhar_train_indices])
        label_source_blocks.append(
            np.full(len(radhar_train_indices), "radhar_recording_activity_label")
        )
    if iwr6843 is not None:
        feature_blocks.append(iwr6843["features"][iwr6843_train_indices])
        label_blocks.append(iwr6843["labels"][iwr6843_train_indices])
        split_blocks.append(np.full(len(iwr6843_train_indices), "train"))
        origin_blocks.append(
            np.full(len(iwr6843_train_indices), "iwr6843_fall_102")
        )
        source_blocks.append(iwr6843["source_files"][iwr6843_train_indices])
        activity_blocks.append(iwr6843["action"][iwr6843_train_indices])
        label_source_blocks.append(
            np.full(
                len(iwr6843_train_indices),
                "iwr6843_recording_level_nonfall_only",
            )
        )

    features = np.concatenate(feature_blocks).astype(np.float32, copy=False)
    labels = np.concatenate(label_blocks).astype(np.int8, copy=False)
    splits = np.concatenate(split_blocks)
    origins = np.concatenate(origin_blocks)
    source_files = np.concatenate(source_blocks)
    activities = np.concatenate(activity_blocks)
    label_sources = np.concatenate(label_source_blocks)
    if not set(np.unique(splits)).issubset({"train", "validation", "test"}):
        raise ValueError("composite dataset contains an unsupported split")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    description = (
        "mmFall DS2 weak anchor labels + DGUHA skeleton-derived descent-onset "
        "pseudo-labels + mmRadPose training-subject hard negatives"
        + (
            " + RadHAR external-training-pool hard negatives"
            if radhar is not None
            else ""
        )
        + (
            " + IWR6843 training-subject bow/squat/walk hard negatives"
            if iwr6843 is not None
            else ""
        )
    )
    positive_definition = (
        "mmFall: 0.2-1.5 s before official fall-motion frame anchor; DGUHA: "
        "0.1-0.6 s before skeleton-derived whole-body descent onset. Both are "
        "weak research labels, not clinically verified loss-of-balance labels."
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=features,
                labels=labels,
                split=splits,
                dataset_origin=origins,
                source_files=source_files,
                activity=activities,
                label_source=label_sources,
                feature_version=np.asarray(FEATURE_VERSION_V2),
                feature_names=np.asarray(FEATURE_NAMES_V2),
                dataset_mode=np.asarray(RESEARCH_MODEL_MODE),
                dataset_description=np.asarray(description),
                positive_label_definition=np.asarray(positive_definition),
                mmradpose_training_subjects_only=np.asarray(True),
                deployment_eligible=np.asarray(False),
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = CompositeResearchSummary(
        output_file=str(destination),
        output_sha256=_sha256(destination),
        mmfall_file=str(mmfall_file),
        dguha_file=str(dguha_file),
        mmradpose_file=str(mmradpose_file),
        radhar_file=str(radhar_file) if radhar_file is not None else None,
        iwr6843_file=str(iwr6843_file) if iwr6843_file is not None else None,
        sample_count=len(labels),
        positive_count=int(labels.sum()),
        negative_count=int(len(labels) - labels.sum()),
        train_count=int(np.sum(splits == "train")),
        validation_count=int(np.sum(splits == "validation")),
        test_count=int(np.sum(splits == "test")),
        mmfall_count=len(mmfall["labels"]),
        dguha_count=len(dguha["labels"]),
        mmradpose_train_negative_count=len(train_indices),
        radhar_train_negative_count=len(radhar_train_indices),
        iwr6843_train_negative_count=len(iwr6843_train_indices),
        feature_version=FEATURE_VERSION_V2,
        window_size=WINDOW_SIZE_V2,
        input_size=len(FEATURE_NAMES_V2),
        deployment_eligible=False,
    )
    _write_manifest(
        destination,
        summary,
        inputs=tuple(inputs),
        origins=origins,
        splits=splits,
        labels=labels,
        description=description,
        positive_definition=positive_definition,
    )
    return summary


def _load_component(
    path: Path,
    *,
    expected_mode: str,
    require_both_labels: bool,
) -> dict[str, np.ndarray | bool]:
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "features",
            "labels",
            "split",
            "feature_version",
            "feature_names",
            "dataset_mode",
        }
        missing = sorted(required.difference(dataset.files))
        if missing:
            raise ValueError(f"component dataset is incomplete: {missing}")
        if str(dataset["dataset_mode"].item()) != expected_mode:
            raise ValueError(f"unexpected component dataset mode: {path}")
        if str(dataset["feature_version"].item()) != FEATURE_VERSION_V2:
            raise ValueError("component feature version is incompatible")
        if tuple(str(value) for value in dataset["feature_names"]) != FEATURE_NAMES_V2:
            raise ValueError("component feature names/order are incompatible")
        features = np.asarray(dataset["features"], dtype=np.float32)
        labels = np.asarray(dataset["labels"], dtype=np.int8)
        splits = np.asarray(dataset["split"])
        result: dict[str, np.ndarray | bool] = {
            name: np.asarray(dataset[name])
            for name in dataset.files
            if name
            in {
                "source_files",
                "source_categories",
                "action",
                "label_source",
                "source_complete",
                "subject_id",
            }
        }
    if features.shape[1:] != (WINDOW_SIZE_V2, len(FEATURE_NAMES_V2)):
        raise ValueError("component feature tensor has incompatible shape")
    if labels.shape != (len(features),) or splits.shape != (len(features),):
        raise ValueError("component labels/splits have incompatible shape")
    if not np.isfinite(features).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("component features or labels are invalid")
    if require_both_labels and len(np.unique(labels)) != 2:
        raise ValueError("component must contain both labels")
    result.update({"features": features, "labels": labels, "split": splits})
    result["source_complete"] = bool(
        np.asarray(result.get("source_complete", True)).item()
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    path: Path,
    summary: CompositeResearchSummary,
    *,
    inputs: tuple[Path, ...],
    origins: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    description: str,
    positive_definition: str,
) -> None:
    payload = asdict(summary)
    payload["inputs"] = [
        {"path": str(source), "sha256": _sha256(source)} for source in inputs
    ]
    payload["dataset_description"] = description
    payload["positive_label_definition"] = positive_definition
    payload["counts"] = {
        origin: {
            split: {
                "positive": int(np.sum((origins == origin) & (splits == split) & (labels == 1))),
                "negative": int(np.sum((origins == origin) & (splits == split) & (labels == 0))),
            }
            for split in ("train", "validation", "test")
        }
        for origin in sorted(set(origins.tolist()))
    }
    payload["known_limitations"] = [
        "all positive labels are weak or skeleton-derived pseudo-labels",
        "DGUHA contains forward falls by young healthy subjects only",
        "mmRadPose contributes training negatives only and has no falls",
        "RadHAR contributes training negatives only when supplied and has no falls",
        "IWR6843 contributes only training-subject non-fall recordings; its fall recordings are excluded because they have no pre-fall timing labels",
        "mmFall has no reliable subject identifiers",
        "the composite artifact is research-only and non-deployable",
    ]
    path.with_suffix(".manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a composite v2 research dataset.")
    parser.add_argument("--mmfall", required=True, type=Path)
    parser.add_argument("--dguha", required=True, type=Path)
    parser.add_argument("--mmradpose", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radhar", type=Path)
    parser.add_argument("--iwr6843", type=Path)
    parser.add_argument("--max-mmradpose-train-negatives", type=int, default=4000)
    args = parser.parse_args()
    summary = build_composite_research_npz(
        args.mmfall,
        args.dguha,
        args.mmradpose,
        args.output,
        radhar_path=args.radhar,
        iwr6843_path=args.iwr6843,
        max_mmradpose_train_negatives=args.max_mmradpose_train_negatives,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
