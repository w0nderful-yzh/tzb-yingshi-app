from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from radar_module.preprocess.relative_temporal_features_v3 import (
    BASELINE_FRAME_COUNT_V3,
    FEATURE_NAMES_V3,
    FEATURE_VERSION_V3,
    transform_v2_values_to_v3,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
)


def convert_dataset_to_relative_features_v3(
    source_path: str | Path,
    output_path: str | Path,
    *,
    baseline_frame_count: int = BASELINE_FRAME_COUNT_V3,
) -> dict[str, object]:
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    with np.load(source, allow_pickle=False) as dataset:
        if str(dataset["feature_version"].item()) != FEATURE_VERSION_V2:
            raise ValueError("source dataset must use radar_features_v2")
        if tuple(str(value) for value in dataset["feature_names"]) != FEATURE_NAMES_V2:
            raise ValueError("source feature order is incompatible")
        arrays = {name: np.asarray(dataset[name]) for name in dataset.files}

    arrays["features"] = transform_v2_values_to_v3(
        arrays["features"], baseline_frame_count=baseline_frame_count
    )
    arrays["feature_version"] = np.asarray(FEATURE_VERSION_V3)
    arrays["feature_names"] = np.asarray(FEATURE_NAMES_V3)
    arrays["deployment_eligible"] = np.asarray(False)
    arrays["relative_baseline_frame_count"] = np.asarray(
        baseline_frame_count, dtype=np.int16
    )
    arrays["relative_feature_definition"] = np.asarray(
        "absolute z/range/spread and log1p(point_count) minus the median of "
        "valid point frames in the first baseline segment"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    labels = np.asarray(arrays["labels"], dtype=np.int8)
    splits = np.asarray(arrays["split"])
    report = {
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "output_file": str(destination),
        "output_sha256": _sha256(destination),
        "feature_version": FEATURE_VERSION_V3,
        "feature_names": list(FEATURE_NAMES_V3),
        "baseline_frame_count": baseline_frame_count,
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "split_counts": {
            name: int(np.sum(splits == name))
            for name in ("train", "validation", "test")
        },
        "labels_and_splits_preserved": True,
        "deployment_eligible": False,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an existing v2 temporal dataset to relative geometry v3."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--baseline-frame-count", type=int, default=BASELINE_FRAME_COUNT_V3
    )
    args = parser.parse_args()
    report = convert_dataset_to_relative_features_v3(
        args.source,
        args.output,
        baseline_frame_count=args.baseline_frame_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
