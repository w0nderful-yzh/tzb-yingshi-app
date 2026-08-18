from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.dataset.iwr6843_fall_v1 import (
    NONFALL_ACTIONS,
    parse_iwr6843_fall_csv,
)
from radar_module.preprocess.pointcloud_sequence import PointCloudSequenceBuilder


SEQUENCE_VERSION = "radar_point_sequence_v2"
FEATURE_NAMES = ("x_m", "y_m", "z_m", "radial_velocity_mps", "snr")
DGUHA_MODE = "DGUHA_POINT_PREFALL_IWR_ADAPTATION_V1"
IWR_MODE = "IWR6843_NONFALL_REPRESENTATION_V1"
IWR_ACTIONS = ("bow", "squat", "walk")
IWR_SPLIT_BY_SUBJECT = {"Areeb": "train", "Raffay": "train", "Towsif": "validation"}
IWR_FALL102_SNR_TO_DB_SCALE = 0.1


def build_adaptation_datasets(
    dguha_source: str | Path,
    iwr_source: str | Path,
    dguha_output: str | Path,
    iwr_output: str | Path,
    audit_output: str | Path,
    *,
    replay_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    dguha_source = Path(dguha_source).resolve()
    iwr_source = Path(iwr_source).resolve()
    dguha_output = Path(dguha_output).resolve()
    iwr_output = Path(iwr_output).resolve()
    audit_output = Path(audit_output).resolve()
    if not dguha_source.is_file():
        raise FileNotFoundError(dguha_source)
    gathered = _resolve_gathered_data(iwr_source)

    dguha_summary = _export_dguha(dguha_source, dguha_output)
    iwr_summary = _export_iwr(gathered, iwr_output)
    audit = _coordinate_snr_audit(
        dguha_output,
        iwr_output,
        tuple(Path(value).resolve() for value in replay_paths),
    )
    audit["artifacts"] = {
        "dguha": dguha_summary,
        "iwr6843_nonfall": iwr_summary,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit


def _export_dguha(source: Path, destination: Path) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as data:
        required = {
            "points", "point_mask", "frame_mask", "labels", "split", "subject_id",
            "source_files", "window_end_seconds", "seconds_to_onset", "label_source",
            "prediction_horizon_seconds", "feature_names", "sequence_version",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"DGUHA source is incomplete: {missing}")
        old_points = np.asarray(data["points"], dtype=np.float32)
        point_mask = np.asarray(data["point_mask"], dtype=np.bool_)
        frame_mask = np.asarray(data["frame_mask"], dtype=np.bool_)
        old_names = tuple(str(value) for value in data["feature_names"])
        expected = ("x_m", "y_m", "z_m", "radial_velocity_mps", "snr", "snr_present")
        if old_names != expected or old_points.shape[-1] != 6:
            raise ValueError("DGUHA source point feature contract is incompatible")
        points = old_points[..., :5].copy()
        # DGUHA/RadHAR ROS exports use x as forward range and y as lateral.
        # The deployment/Fall-102 convention is x lateral and y forward.
        points[..., [0, 1]] = points[..., [1, 0]]
        points[~point_mask] = 0.0
        snr_available = point_mask & (old_points[..., 5] > 0.5)
        labels = np.asarray(data["labels"], dtype=np.int8)
        splits = np.asarray(data["split"])
        subjects = np.asarray(data["subject_id"])
        source_files = np.asarray(data["source_files"])
        window_end = np.asarray(data["window_end_seconds"], dtype=np.float32)
        seconds_to_onset = np.asarray(data["seconds_to_onset"], dtype=np.float32)
        label_source = np.asarray(data["label_source"])
        horizon = np.asarray(data["prediction_horizon_seconds"], dtype=np.float32)

    sample_id = np.asarray([
        hashlib.sha256(f"{path}|{float(end):.6f}".encode("utf-8")).hexdigest()[:20]
        for path, end in zip(source_files, window_end)
    ])
    if len(set(sample_id.tolist())) != len(sample_id):
        raise ValueError("DGUHA sample_id collision")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(
        destination,
        points=points,
        point_mask=point_mask,
        frame_mask=frame_mask,
        snr_available=snr_available,
        labels=labels,
        split=splits,
        subject_id=subjects,
        source_files=source_files,
        window_end_seconds=window_end,
        seconds_to_onset=seconds_to_onset,
        label_source=label_source,
        sample_id=sample_id,
        prediction_horizon_seconds=horizon,
        sequence_version=np.asarray(SEQUENCE_VERSION),
        feature_names=np.asarray(FEATURE_NAMES),
        dataset_mode=np.asarray(DGUHA_MODE),
        coordinate_contract=np.asarray("x=lateral,y=forward,z=vertical; source x/y swapped"),
        snr_contract=np.asarray("missing; value zero and snr_available=false"),
        kinect_used_as_model_input=np.asarray(False),
        deployment_eligible=np.asarray(False),
    )
    summary = {
        "path": str(destination),
        "sha256": _sha256(destination),
        "source_sha256": _sha256(source),
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "split_counts": dict(Counter(str(v) for v in splits)),
        "positive_by_split": {
            name: int(np.sum((splits == name) & (labels == 1)))
            for name in sorted(set(splits.tolist()))
        },
        "label_source_counts": dict(Counter(str(v) for v in label_source)),
        "unique_subjects": int(len(set(subjects.tolist()))),
        "unique_recordings": int(len(set(source_files.tolist()))),
        "feature_names": list(FEATURE_NAMES),
        "coordinate_transform": "source(x=forward,y=lateral) -> canonical(x=lateral,y=forward)",
        "snr_available_fraction": float(snr_available.sum() / max(point_mask.sum(), 1)),
        "sample_id_sha256": _array_sha256(sample_id),
    }
    _write_manifest(destination, summary)
    return summary


def _export_iwr(source_root: Path, destination: Path) -> dict[str, Any]:
    builder = PointCloudSequenceBuilder()
    points_list: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    snr_masks: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    subjects: list[str] = []
    actions: list[str] = []
    sources: list[str] = []
    end_frames: list[int] = []
    sample_ids: list[str] = []
    files = sorted(source_root.glob("Not/*.csv"))
    if not files:
        raise FileNotFoundError(f"no nonfall IWR6843 files under {source_root}")
    for path in files:
        stem = path.stem.split("_")
        subject, action = stem[0], stem[1]
        if subject not in IWR_SPLIT_BY_SUBJECT or action not in NONFALL_ACTIONS:
            raise ValueError(f"unexpected Fall-102 nonfall file: {path.name}")
        frames, _ = parse_iwr6843_fall_csv(path)
        for end_index in range(builder.time_steps - 1, len(frames)):
            sequence = builder.transform(frames[: end_index + 1], end_timestamp=frames[end_index].timestamp)
            point5 = sequence.values[..., :5].copy()
            # Fall-102 stores the uint16 side-info count. TI's official
            # detected-points parser converts this field to dB with * 0.1.
            point5[..., 4] *= IWR_FALL102_SNR_TO_DB_SCALE
            point5[~sequence.point_mask] = 0.0
            snr_available = sequence.point_mask & (sequence.values[..., 5] > 0.5)
            relative = path.relative_to(source_root).as_posix()
            sample_id = hashlib.sha256(
                f"{relative}|frame={end_index}".encode("utf-8")
            ).hexdigest()[:20]
            points_list.append(point5)
            point_masks.append(sequence.point_mask)
            frame_masks.append(sequence.frame_mask)
            snr_masks.append(snr_available)
            labels.append(IWR_ACTIONS.index(action))
            splits.append(IWR_SPLIT_BY_SUBJECT[subject])
            subjects.append(subject)
            actions.append(action)
            sources.append(relative)
            end_frames.append(end_index)
            sample_ids.append(sample_id)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("IWR6843 sample_id collision")
    points = np.stack(points_list).astype(np.float32, copy=False)
    point_mask = np.stack(point_masks)
    frame_mask = np.stack(frame_masks)
    snr_available = np.stack(snr_masks)
    labels_array = np.asarray(labels, dtype=np.int8)
    splits_array = np.asarray(splits)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(
        destination,
        points=points,
        point_mask=point_mask,
        frame_mask=frame_mask,
        snr_available=snr_available,
        labels=labels_array,
        split=splits_array,
        subject_id=np.asarray(subjects),
        action=np.asarray(actions),
        action_names=np.asarray(IWR_ACTIONS),
        source_files=np.asarray(sources),
        window_end_frame=np.asarray(end_frames, dtype=np.int16),
        sample_id=np.asarray(sample_ids),
        sequence_version=np.asarray(SEQUENCE_VERSION),
        feature_names=np.asarray(FEATURE_NAMES),
        dataset_mode=np.asarray(IWR_MODE),
        coordinate_contract=np.asarray("x=lateral,y=forward,z=vertical; no axis transform"),
        snr_contract=np.asarray("Fall-102 raw side-info count * 0.1 = dB; availability explicit"),
        fall_recordings_included=np.asarray(False),
        prediction_labels_used=np.asarray(False),
        deployment_eligible=np.asarray(False),
    )
    summary = {
        "path": str(destination),
        "sha256": _sha256(destination),
        "source_root": str(source_root),
        "source_file_count": len(files),
        "sample_count": len(labels),
        "split_counts": dict(Counter(splits)),
        "subject_counts": dict(Counter(subjects)),
        "action_counts_recordings": dict(Counter(path.stem.split("_")[1] for path in files)),
        "action_counts_windows": dict(Counter(actions)),
        "feature_names": list(FEATURE_NAMES),
        "coordinate_transform": "none; source already canonical x=lateral,y=forward",
        "snr_available_fraction": float(snr_available.sum() / max(point_mask.sum(), 1)),
        "snr_scale_to_db": IWR_FALL102_SNR_TO_DB_SCALE,
        "snr_scale_evidence": "TI official parseTLVs.py detected-points side info: snr * 0.1",
        "fall_recordings_included": False,
        "sample_id_sha256": _array_sha256(np.asarray(sample_ids)),
    }
    _write_manifest(destination, summary)
    return summary


def _coordinate_snr_audit(
    dguha_path: Path,
    iwr_path: Path,
    replay_paths: tuple[Path, ...],
) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for name, path in (("dguha_canonical", dguha_path), ("fall102_nonfall", iwr_path)):
        with np.load(path, allow_pickle=False) as data:
            points = np.asarray(data["points"], dtype=np.float32)
            mask = np.asarray(data["point_mask"], dtype=np.bool_)
            snr_mask = np.asarray(data["snr_available"], dtype=np.bool_)
        valid = points[mask]
        domains[name] = _point_stats(valid, snr_mask[mask])
    replay_entries: list[dict[str, Any]] = []
    for path in replay_paths:
        if not path.is_file():
            replay_entries.append({"path": str(path), "exists": False})
            continue
        rows: list[list[float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                for point in payload.get("points", ()):
                    rows.append([
                        float(point["x"]), float(point["y"]), float(point["z"]),
                        float(point.get("velocity", 0.0)), float(point.get("snr", 0.0)),
                    ])
        values = np.asarray(rows, dtype=np.float32).reshape((-1, 5))
        replay_entries.append({
            "path": str(path), "exists": True, "stats": _point_stats(values, np.ones(len(values), bool))
        })
    fall_snr = domains["fall102_nonfall"]["snr"]
    replay_snr = [entry["stats"]["snr"] for entry in replay_entries if entry.get("exists")]
    snr_ratio = None
    if replay_snr and fall_snr["median"] > 0:
        snr_ratio = replay_snr[0]["median"] / fall_snr["median"]
    return {
        "audit_version": "point_iwr6843_contract_audit_v1",
        "status": "PASS_WITH_EXPLICIT_SNR_MISSINGNESS",
        "canonical_coordinate_contract": {
            "feature_order": list(FEATURE_NAMES),
            "x": "lateral (m)", "y": "forward/range (m)", "z": "vertical (m)",
            "velocity": "radial velocity (m/s)",
            "dguha_transform": "swap source x and y",
            "fall102_transform": "none",
            "live_transform": "none",
            "evidence": "canonicalized axis absolute medians and deployment JSONL geometry",
        },
        "snr_contract": {
            "dguha": "unavailable; normalized input forced to zero using snr_available mask",
            "fall102": "CSV snr column, numeric and available",
            "live": "decoded TLV SNR, numeric and available",
            "fall102_vs_live_unit_compatibility": "aligned to dB using TI official detected-points side-info scale 0.1",
            "scale_evidence": "radar_toolbox_2_20_00_05/tools/visualizers/Applications_Visualizer/common/parseTLVs.py",
            "first_replay_to_fall102_median_ratio": snr_ratio,
        },
        "domains": domains,
        "replays": replay_entries,
    }


def _point_stats(values: np.ndarray, snr_available: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"point_count": 0}
    result: dict[str, Any] = {"point_count": int(len(values))}
    for index, name in enumerate(FEATURE_NAMES[:4]):
        column = values[:, index]
        result[name] = _quantiles(column)
    selected_snr = values[snr_available, 4]
    result["snr_available_fraction"] = float(np.mean(snr_available))
    result["snr"] = _quantiles(selected_snr) if len(selected_snr) else None
    result["absolute_axis_medians"] = {
        name: float(np.median(np.abs(values[:, index]))) for index, name in enumerate(FEATURE_NAMES[:3])
    }
    return result


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)), "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _resolve_gathered_data(path: Path) -> Path:
    candidates = (path, path / "GatheredData", path / "mmwave-radar-fall-detection-main" / "GatheredData")
    for candidate in candidates:
        if (candidate / "Not").is_dir():
            return candidate
    raise FileNotFoundError(f"cannot locate Fall-102 GatheredData below {path}")


def _save_npz(destination: Path, **arrays: np.ndarray) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(destination: Path, payload: dict[str, Any]) -> None:
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the first-stage PointNet IWR6843 adaptation manifests.")
    parser.add_argument("--dguha-source", required=True, type=Path)
    parser.add_argument("--iwr-source", required=True, type=Path)
    parser.add_argument("--dguha-output", required=True, type=Path)
    parser.add_argument("--iwr-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--replay", action="append", default=[], type=Path)
    args = parser.parse_args()
    result = build_adaptation_datasets(
        args.dguha_source, args.iwr_source, args.dguha_output, args.iwr_output,
        args.audit_output, replay_paths=tuple(args.replay),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
