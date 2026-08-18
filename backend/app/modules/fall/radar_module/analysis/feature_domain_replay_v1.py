from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from radar_module.acquisition.ti_reader import JsonlReplayAdapter, TiRadarReader
from radar_module.contracts import RadarFrame, Room
from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.relative_temporal_features_v3 import (
    FEATURE_NAMES_V3,
    FEATURE_VERSION_V3,
    transform_v2_values_to_v3,
)
from radar_module.preprocess.hybrid_temporal_features_v4 import (
    FEATURE_NAMES_V4,
    FEATURE_VERSION_V4,
    transform_v2_values_to_v4,
)
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    RadarTemporalFeatureExtractorV2,
    TemporalDataQuality,
)


def evaluate_replay_feature_domain(
    replay_path: str | Path,
    checkpoint_path: str | Path,
    *,
    confirmation_windows: int = 3,
) -> dict[str, object]:
    replay = Path(replay_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    feature_version = str(checkpoint["feature_version"])
    expected_names = {
        FEATURE_VERSION_V2: FEATURE_NAMES_V2,
        FEATURE_VERSION_V3: FEATURE_NAMES_V3,
        FEATURE_VERSION_V4: FEATURE_NAMES_V4,
    }.get(feature_version)
    if expected_names is None:
        raise ValueError("checkpoint feature version is unsupported")
    if tuple(checkpoint["feature_names"]) != expected_names:
        raise ValueError("checkpoint feature order is incompatible")

    adapter = JsonlReplayAdapter(replay, speed=100_000.0, loop=False)
    reader = TiRadarReader(adapter, device_id="feature-domain-replay", room=Room.BATHROOM)
    extractor = RadarTemporalFeatureExtractorV2()
    frames: deque[RadarFrame] = deque()
    values: list[np.ndarray] = []
    qualities: Counter[str] = Counter()
    last_evaluation: datetime | None = None
    frame_count = 0
    reader.start()
    try:
        while not adapter.finished:
            frame = reader.read()
            if frame is None:
                continue
            frame_count += 1
            frames.append(frame)
            while frames and (frame.timestamp - frames[0].timestamp).total_seconds() > 2.2:
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
            feature_values = np.asarray(window.values, dtype=np.float32)
            if feature_version == FEATURE_VERSION_V3:
                feature_values = transform_v2_values_to_v3(feature_values)
            elif feature_version == FEATURE_VERSION_V4:
                feature_values = transform_v2_values_to_v4(feature_values)
            values.append(feature_values)
    finally:
        reader.stop()

    if not values:
        raise ValueError("replay produced no valid feature windows")
    raw = np.stack(values).astype(np.float32, copy=False)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = ((raw - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    model = TemporalBinaryModel(
        architecture=str(checkpoint["model_architecture"]),
        input_size=len(expected_names),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    score_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 512):
            score_batches.append(
                torch.sigmoid(model(torch.from_numpy(normalized[start : start + 512])))
                .numpy()
                .astype(np.float64)
            )
    scores = np.concatenate(score_batches)
    threshold = float(checkpoint["decision_threshold"])
    confirmed_runs = _confirmed_run_count(
        scores >= threshold, confirmation_windows=confirmation_windows
    )
    absolute_normalized = np.abs(normalized.astype(np.float64))
    return {
        "analysis_version": "feature_domain_replay_v1",
        "replay_file": str(replay),
        "replay_sha256": _sha256(replay),
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "feature_version": feature_version,
        "threshold": threshold,
        "frame_count": frame_count,
        "valid_window_count": int(len(scores)),
        "quality_counts": dict(qualities),
        "score_distribution": _describe(scores),
        "above_threshold_window_fraction": float(np.mean(scores >= threshold)),
        "confirmed_run_count": confirmed_runs,
        "normalized_feature_absolute_distribution": _describe(
            absolute_normalized.reshape(-1)
        ),
        "normalized_feature_fraction_above_5sigma": float(
            np.mean(absolute_normalized > 5.0)
        ),
        "normalized_feature_fraction_above_10sigma": float(
            np.mean(absolute_normalized > 10.0)
        ),
        "deployment_eligible": False,
    }


def _confirmed_run_count(high: np.ndarray, *, confirmation_windows: int) -> int:
    runs = 0
    length = 0
    for value in np.asarray(high, dtype=bool):
        if value:
            length += 1
            if length == confirmation_windows:
                runs += 1
        else:
            length = 0
    return runs


def _describe(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint score saturation and normalized feature shift on JSONL."
    )
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--confirmation-windows", type=int, default=3)
    args = parser.parse_args()
    report = evaluate_replay_feature_domain(
        args.replay,
        args.checkpoint,
        confirmation_windows=args.confirmation_windows,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
