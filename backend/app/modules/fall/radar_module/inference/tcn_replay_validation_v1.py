from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from radar_module.acquisition.ti_reader import JsonlReplayAdapter, TiRadarReader
from radar_module.contracts import RadarFrame, Room
from radar_module.inference.tcn_live_v1 import RadarTcnLivePredictorV1
from radar_module.model.temporal_models_v3 import TemporalBinaryModel
from radar_module.preprocess.temporal_features_v2 import (
    FEATURE_NAMES_V2,
    RadarTemporalFeatureExtractorV2,
)


@dataclass(frozen=True, slots=True)
class ReplayConsistencySummaryV1:
    schema_version: str
    replay_file: str
    replay_sha256: str
    checkpoint_file: str
    checkpoint_sha256: str
    frame_count: int
    emitted_result_count: int
    unknown_result_count: int
    compared_window_count: int
    maximum_absolute_score_difference: float
    mean_absolute_score_difference: float
    state_mismatch_count: int
    tolerance: float
    consistent: bool
    shadow_only: bool = True
    formal_alerts_enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_tcn_jsonl_replay(
    replay_path: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    output_jsonl_path: str | Path | None = None,
    confirmation_windows: int = 3,
    tolerance: float = 1e-6,
    device: str | torch.device = "cpu",
) -> ReplayConsistencySummaryV1:
    """Compare live sequential TCN scores with independent batched inference."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    replay_file = Path(replay_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    predictor = RadarTcnLivePredictorV1(
        checkpoint_file,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        confirmation_windows=confirmation_windows,
        device=device,
    )
    adapter = JsonlReplayAdapter(replay_file, speed=100_000.0, loop=False)
    reader = TiRadarReader(
        adapter,
        device_id="tcn-jsonl-replay-v1",
        room=Room.BATHROOM,
    )

    frames: list[RadarFrame] = []
    emitted: list[tuple[int, dict[str, object]]] = []
    reader.start()
    try:
        while not adapter.finished:
            frame = reader.read()
            if frame is None:
                continue
            frames.append(frame)
            result = predictor.consume(frame)
            if result is not None:
                emitted.append((len(frames) - 1, result.to_dict()))
    finally:
        reader.stop()

    valid_entries = [
        (frame_index, payload)
        for frame_index, payload in emitted
        if bool(payload["score_valid"])
    ]
    batch_scores = _infer_independent_batch(
        frames,
        [frame_index for frame_index, _ in valid_entries],
        checkpoint_file,
        device=device,
    )
    batch_states = _derive_confirmed_states(
        batch_scores,
        threshold=predictor.threshold,
        confirmation_windows=confirmation_windows,
    )

    differences: list[float] = []
    state_mismatch_count = 0
    batch_by_frame: dict[int, tuple[float, str]] = {}
    for (frame_index, live_payload), batch_score, batch_state in zip(
        valid_entries, batch_scores, batch_states, strict=True
    ):
        difference = abs(float(live_payload["pre_fall_score"]) - batch_score)
        differences.append(difference)
        if live_payload["risk_state"] != batch_state:
            state_mismatch_count += 1
        batch_by_frame[frame_index] = (batch_score, batch_state)

    maximum_difference = max(differences, default=0.0)
    mean_difference = float(np.mean(differences)) if differences else 0.0
    summary = ReplayConsistencySummaryV1(
        schema_version="radar_tcn_replay_consistency_v1",
        replay_file=str(replay_file),
        replay_sha256=_sha256(replay_file),
        checkpoint_file=str(checkpoint_file),
        checkpoint_sha256=predictor.checkpoint_sha256,
        frame_count=len(frames),
        emitted_result_count=len(emitted),
        unknown_result_count=sum(
            not bool(payload["score_valid"]) for _, payload in emitted
        ),
        compared_window_count=len(valid_entries),
        maximum_absolute_score_difference=maximum_difference,
        mean_absolute_score_difference=mean_difference,
        state_mismatch_count=state_mismatch_count,
        tolerance=float(tolerance),
        consistent=(
            bool(valid_entries)
            and maximum_difference <= tolerance
            and state_mismatch_count == 0
        ),
    )
    if output_jsonl_path is not None:
        _write_validation_artifacts(
            Path(output_jsonl_path).resolve(),
            emitted,
            batch_by_frame,
            summary,
        )
    return summary


def _infer_independent_batch(
    frames: Sequence[RadarFrame],
    frame_indices: Sequence[int],
    checkpoint_path: Path,
    *,
    device: str | torch.device,
) -> list[float]:
    if not frame_indices:
        return []
    torch_device = torch.device(device)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=torch_device, weights_only=True
    )
    extractor = RadarTemporalFeatureExtractorV2()
    windows = [
        extractor.transform(
            frames[: frame_index + 1],
            end_timestamp=frames[frame_index].timestamp,
        ).values
        for frame_index in frame_indices
    ]
    values = np.asarray(windows, dtype=np.float32)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = ((values - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )
    model = TemporalBinaryModel(
        architecture=str(checkpoint["model_architecture"]),
        input_size=len(FEATURE_NAMES_V2),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch_device)
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(
            model(torch.from_numpy(normalized).to(torch_device))
        )
    return [float(value) for value in scores.detach().cpu().tolist()]


def _derive_confirmed_states(
    scores: Sequence[float],
    *,
    threshold: float,
    confirmation_windows: int,
) -> list[str]:
    consecutive = 0
    states: list[str] = []
    for score in scores:
        if score >= threshold:
            consecutive += 1
            states.append(
                "IMMINENT" if consecutive >= confirmation_windows else "WATCH"
            )
        else:
            consecutive = 0
            states.append("NORMAL")
    return states


def _write_validation_artifacts(
    output_path: Path,
    emitted: Sequence[tuple[int, dict[str, object]]],
    batch_by_frame: dict[int, tuple[float, str]],
    summary: ReplayConsistencySummaryV1,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for frame_index, live_payload in emitted:
            batch = batch_by_frame.get(frame_index)
            record = {
                "frame_index": frame_index,
                "tcn_prediction": live_payload,
                "independent_batch_score": batch[0] if batch else None,
                "independent_batch_state": batch[1] if batch else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate sequential TCN live inference against batch inference."
    )
    parser.add_argument("--replay", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirmation-windows", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = validate_tcn_jsonl_replay(
        args.replay,
        args.checkpoint,
        expected_checkpoint_sha256=args.sha256,
        output_jsonl_path=args.output,
        confirmation_windows=args.confirmation_windows,
        tolerance=args.tolerance,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
