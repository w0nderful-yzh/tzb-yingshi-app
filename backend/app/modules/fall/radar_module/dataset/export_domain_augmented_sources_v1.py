from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from radar_module.dataset.dguha_research_v2 import export_dguha_research_npz
from radar_module.dataset.iwr6843_fall_v1 import export_iwr6843_fall_sequence_npz
from radar_module.dataset.mmradpose_converter import export_mmradpose_hard_negative_npz
from radar_module.dataset.mmwave_ocpid_v1 import export_mmwave_ocpid_hard_negative_npz
from radar_module.dataset.radhar_converter import export_radhar_hard_negative_npz
from radar_module.preprocess.radar_domain_augmentation_v1 import (
    RadarDomainAugmentationConfigV1,
    RadarDomainAugmentedFeatureExtractorV1,
)


def export_domain_augmented_sources(
    *,
    dguha_root: str | Path,
    mmradpose_root: str | Path,
    radhar_root: str | Path,
    ocpid_root: str | Path,
    iwr6843_root: str | Path,
    clean_processed_root: str | Path,
    output_directory: str | Path,
    seed: int = 20260809,
) -> dict[str, object]:
    output_root = Path(output_directory).resolve()
    clean_root = Path(clean_processed_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extractor = RadarDomainAugmentedFeatureExtractorV1(
        RadarDomainAugmentationConfigV1(seed=seed)
    )
    outputs = {
        "dguha_base": output_root / "dguha_prefall_0p5_1p0_augmented_v1.npz",
        "dguha_early": output_root / "dguha_prefall_0p5_2p0_augmented_v1.npz",
        "mmradpose": output_root / "mmradpose_augmented_v1.npz",
        "radhar": output_root / "radhar_augmented_v1.npz",
        "mmwave_ocpid": output_root / "mmwave_ocpid_augmented_v1.npz",
        "iwr6843_fall_102": output_root / "iwr6843_normal_augmented_v1.npz",
    }
    summaries: dict[str, object] = {}

    print("[1/6] exporting augmented DGUHA 0.5-1.0 s", flush=True)
    if outputs["dguha_base"].is_file():
        summaries["dguha_base"] = _existing_manifest(outputs["dguha_base"])
    else:
        summaries["dguha_base"] = asdict(
            export_dguha_research_npz(
                dguha_root,
                outputs["dguha_base"],
                allow_skeleton_pseudolabels=True,
                minimum_lead_seconds=0.5,
                maximum_lead_seconds=1.0,
                positive_stride_seconds=0.1,
                negative_stride_seconds=0.2,
                max_negative_windows_per_recording=100,
                positive_anchor="descent_onset",
                minimum_pre_descent_margin_seconds=0.1,
                extractor=extractor,
            )
        )
    _assert_parallel(
        clean_root / "dguha_prefall_0p5_1p0_dense_v3.npz",
        outputs["dguha_base"],
    )

    print("[2/6] exporting augmented DGUHA 0.5-2.0 s", flush=True)
    if outputs["dguha_early"].is_file():
        summaries["dguha_early"] = _existing_manifest(outputs["dguha_early"])
    else:
        summaries["dguha_early"] = asdict(
            export_dguha_research_npz(
                dguha_root,
                outputs["dguha_early"],
                allow_skeleton_pseudolabels=True,
                minimum_lead_seconds=0.5,
                maximum_lead_seconds=2.0,
                positive_stride_seconds=0.1,
                negative_stride_seconds=0.2,
                max_negative_windows_per_recording=100,
                positive_anchor="descent_onset",
                minimum_pre_descent_margin_seconds=0.1,
                extractor=extractor,
            )
        )
    _assert_parallel(
        clean_root / "dguha_prefall_0p5_2p0_dense_v3.npz",
        outputs["dguha_early"],
    )

    print("[3/6] exporting augmented mmRadPose", flush=True)
    if outputs["mmradpose"].is_file():
        summaries["mmradpose"] = _existing_manifest(outputs["mmradpose"])
    else:
        summaries["mmradpose"] = asdict(
            export_mmradpose_hard_negative_npz(
                mmradpose_root,
                outputs["mmradpose"],
                extractor=extractor,
            )
        )
    _assert_parallel(clean_root / "mmradpose_hard_negatives_v2.npz", outputs["mmradpose"])

    print("[4/6] exporting augmented RadHAR", flush=True)
    if outputs["radhar"].is_file():
        summaries["radhar"] = _existing_manifest(outputs["radhar"])
    else:
        summaries["radhar"] = asdict(
            export_radhar_hard_negative_npz(
                radhar_root,
                outputs["radhar"],
                extractor=extractor,
            )
        )
    _assert_parallel(
        clean_root / "radhar_squat_jump_hard_negatives_v2.npz",
        outputs["radhar"],
    )

    print("[5/6] exporting augmented OCPID", flush=True)
    if outputs["mmwave_ocpid"].is_file():
        summaries["mmwave_ocpid"] = _existing_manifest(outputs["mmwave_ocpid"])
    else:
        summaries["mmwave_ocpid"] = asdict(
            export_mmwave_ocpid_hard_negative_npz(
                ocpid_root,
                outputs["mmwave_ocpid"],
                extractor=extractor,
            )
        )
    _assert_parallel(
        clean_root / "mmwave_ocpid_cfar_walking_hard_negative_v1.npz",
        outputs["mmwave_ocpid"],
    )

    print("[6/6] exporting augmented public IWR6843", flush=True)
    if outputs["iwr6843_fall_102"].is_file():
        summaries["iwr6843_fall_102"] = _existing_manifest(
            outputs["iwr6843_fall_102"]
        )
    else:
        summaries["iwr6843_fall_102"] = asdict(
            export_iwr6843_fall_sequence_npz(
                iwr6843_root,
                outputs["iwr6843_fall_102"],
                extractor=extractor,
            )
        )
    _assert_parallel(
        clean_root / "iwr6843_fall_sequence_auxiliary_v1.npz",
        outputs["iwr6843_fall_102"],
    )

    report = {
        "export_version": "radar_domain_augmented_sources_v1",
        "augmentation_spec": extractor.augmentation_spec(),
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
        "summaries": summaries,
        "parallel_clean_contract_check": "passed",
        "validation_and_test_usage": "clean exports only",
        "deployment_eligible": False,
    }
    report_path = output_root / "augmentation_export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _assert_parallel(clean_path: Path, augmented_path: Path) -> None:
    with np.load(clean_path, allow_pickle=False) as clean, np.load(
        augmented_path, allow_pickle=False
    ) as augmented:
        if clean["features"].shape != augmented["features"].shape:
            raise ValueError(f"feature shape mismatch: {clean_path.name}")
        for key in ("labels", "split", "source_files"):
            if not np.array_equal(clean[key], augmented[key]):
                raise ValueError(f"parallel {key} mismatch: {clean_path.name}")
        if "window_end_seconds" in clean.files and not np.allclose(
            clean["window_end_seconds"],
            augmented["window_end_seconds"],
            equal_nan=True,
        ):
            raise ValueError(f"parallel endpoint mismatch: {clean_path.name}")


def _existing_manifest(output_path: Path) -> dict[str, object]:
    manifest = output_path.with_suffix(".manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError(f"existing output lacks manifest: {output_path}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic raw-point augmentation")
    parser.add_argument("--dguha-root", required=True, type=Path)
    parser.add_argument("--mmradpose-root", required=True, type=Path)
    parser.add_argument("--radhar-root", required=True, type=Path)
    parser.add_argument("--ocpid-root", required=True, type=Path)
    parser.add_argument("--iwr6843-root", required=True, type=Path)
    parser.add_argument("--clean-processed-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    export_domain_augmented_sources(
        dguha_root=args.dguha_root,
        mmradpose_root=args.mmradpose_root,
        radhar_root=args.radhar_root,
        ocpid_root=args.ocpid_root,
        iwr6843_root=args.iwr6843_root,
        clean_processed_root=args.clean_processed_root,
        output_directory=args.output_directory,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
