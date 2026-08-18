from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from radar_module.preprocess.hybrid_temporal_features_v4 import (
    BASELINE_FRAME_COUNT_V4,
    FEATURE_NAMES_V4,
    FEATURE_VERSION_V4,
    transform_v2_values_to_v4,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
)


def convert_dataset_to_hybrid_features_v4(
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    with np.load(source, allow_pickle=False) as dataset:
        if str(dataset["feature_version"].item()) != FEATURE_VERSION_V2:
            raise ValueError("source dataset must use radar_features_v2")
        if tuple(str(value) for value in dataset["feature_names"]) != FEATURE_NAMES_V2:
            raise ValueError("source feature order is incompatible")
        arrays = {name: np.asarray(dataset[name]) for name in dataset.files}
    arrays["features"] = transform_v2_values_to_v4(arrays["features"])
    arrays["feature_version"] = np.asarray(FEATURE_VERSION_V4)
    arrays["feature_names"] = np.asarray(FEATURE_NAMES_V4)
    arrays["deployment_eligible"] = np.asarray(False)
    arrays["hybrid_baseline_frame_count"] = np.asarray(
        BASELINE_FRAME_COUNT_V4, dtype=np.int16
    )
    arrays["hybrid_feature_definition"] = np.asarray(
        "baseline-relative centroid/range, centroid-relative z quantiles, "
        "retained shape widths, and log1p absolute point count"
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
        "feature_version": FEATURE_VERSION_V4,
        "feature_names": list(FEATURE_NAMES_V4),
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
    parser = argparse.ArgumentParser(description="Build hybrid-geometry v4 features.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = convert_dataset_to_hybrid_features_v4(args.source, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
