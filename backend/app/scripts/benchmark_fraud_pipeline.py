"""Offline replay benchmark for the fraud decision pipeline.

Reads a desensitized JSONL manifest of conversation samples, replays each one
through the production decision path (rule matching + calibrated classifier +
S0-S5 state machine) and reports both per-stage latency (P50/P95/max) and
accuracy metrics (per-label precision/recall/F1, state accuracy, confusion).

The benchmark is intentionally self-contained and runs without ASR models, a
database or network access, so it can produce a repeatable CPU baseline. WAV
end-to-end replay (VAD + SenseVoice/Paraformer) is layered on top later; see
backend/tests/fixtures/fraud_audio/README.md.

Usage:
    python -m app.scripts.benchmark_fraud_pipeline \
        --manifest path/to/manifest.jsonl \
        --output report.json

VAD parameter comparison (local, non-committed desensitized WAVs):
    python -m app.scripts.benchmark_fraud_pipeline \
        --manifest path/to/manifest.jsonl \
        --wav-dir path/to/local/wavs \
        --vad-silence-end-ms 500
    # rerun with 600 / 700 and diff the report's "vad" section

Manifest line format (one JSON object per line):
    {
      "id": "sample-001",
      "turns": [{"text": "..."}, {"text": "..."}],
      "expected_labels": ["credential_request"],
      "expected_state": "S4_ACTION_INDUCEMENT",
      "scenario": "fake_customer_service",
      "expected_turns": 2
    }

`turns` may be omitted in favour of a single `text` field. Samples may carry a
`split_group` so the harness can warn about train/eval leakage when the same
group appears in both the training data and the manifest.

WAV replay mode: for every WAV in --wav-dir named {sample_id}.wav, the VAD
segmenter runs with the given parameters; expected_turns (when present) is the
ground truth for over-segmentation / merged-sentence rates. No real transcripts
are required: segment timing and call counts are reported without ASR inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.modules.fraud.evidence_labels import CLASSIFIER_LABELS
from app.modules.fraud.risk_engine import build_risk_snapshot
from app.modules.fraud.speech_risk import build_speech_events
from app.modules.fraud.text_classifier import DEFAULT_DATA_DIR, get_default_classifier
from app.modules.fraud.voice_activity import FRAME_MS, VoiceActivitySegmenter

TURN_GAP_MS = 1_500


@dataclass(slots=True)
class SampleResult:
    sample_id: str
    scenario: str
    evidence_extract_ms: float
    state_machine_ms: float
    total_ms: float
    predicted_state: str
    expected_state: str | None
    predicted_labels: list[str]
    expected_labels: list[str]
    split_group: str | None


@dataclass(slots=True)
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * ratio
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _latency_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "p50_ms": round(_percentile(ordered, 0.50), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "max_ms": round(ordered[-1], 3) if ordered else 0.0,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _data_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "turns" not in record and "text" not in record:
                raise ValueError(f"Manifest {path}:{line_number} needs 'turns' or 'text'")
            samples.append(record)
    if not samples:
        raise ValueError(f"Manifest {path} contains no samples")
    return samples


def _replay_sample(sample: dict[str, Any], index: int) -> SampleResult:
    raw_turns = sample.get("turns")
    if raw_turns is None:
        raw_turns = [{"text": str(sample.get("text", ""))}]
    texts = [str(turn.get("text", "")).strip() for turn in raw_turns]
    texts = [text for text in texts if text]

    segments: list[dict[str, Any]] = []
    cursor_ms = 0
    for text in texts:
        duration_ms = max(len(text) * 200, 1_000)
        segments.append(
            {
                "start_ms": cursor_ms,
                "end_ms": cursor_ms + duration_ms,
                "text": text,
                "transcript_status": "FINAL",
            }
        )
        cursor_ms += duration_ms + TURN_GAP_MS

    extract_start = time.monotonic_ns()
    speech_events = build_speech_events(segments)
    evidence_extract_ms = (time.monotonic_ns() - extract_start) / 1_000_000

    machine_start = time.monotonic_ns()
    snapshot = build_risk_snapshot(
        session_id=str(sample.get("id") or f"sample-{index:04d}"),
        device_id="benchmark-device",
        speech_events=speech_events,
        visual_events=[],
        elder_alone=bool(sample.get("elder_alone", True)),
    )
    state_machine_ms = (time.monotonic_ns() - machine_start) / 1_000_000

    predicted_labels = sorted(
        {
            str(item.get("kind"))
            for item in snapshot.evidence_chain
            if item.get("source") in {"speech", "llm"}
        }
    )
    return SampleResult(
        sample_id=str(sample.get("id") or f"sample-{index:04d}"),
        scenario=str(sample.get("scenario", "unknown")),
        evidence_extract_ms=round(evidence_extract_ms, 3),
        state_machine_ms=round(state_machine_ms, 3),
        total_ms=round(evidence_extract_ms + state_machine_ms, 3),
        predicted_state=str(snapshot.state),
        expected_state=sample.get("expected_state"),
        predicted_labels=predicted_labels,
        expected_labels=[str(label) for label in sample.get("expected_labels", [])],
        split_group=sample.get("split_group"),
    )


def _label_metrics(results: list[SampleResult]) -> dict[str, Any]:
    counts: dict[str, _Counts] = {label: _Counts() for label in CLASSIFIER_LABELS}
    for result in results:
        predicted = set(result.predicted_labels)
        expected = set(result.expected_labels)
        for label in CLASSIFIER_LABELS:
            bucket = counts[label]
            if label in predicted and label in expected:
                bucket.tp += 1
            elif label in predicted and label not in expected:
                bucket.fp += 1
            elif label not in predicted and label in expected:
                bucket.fn += 1
    per_label: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for label in CLASSIFIER_LABELS:
        bucket = counts[label]
        total_tp += bucket.tp
        total_fp += bucket.fp
        total_fn += bucket.fn
        precision = bucket.tp / (bucket.tp + bucket.fp) if bucket.tp + bucket.fp else 0.0
        recall = bucket.tp / (bucket.tp + bucket.fn) if bucket.tp + bucket.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": bucket.tp + bucket.fn,
        }
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "per_label": per_label,
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
        },
    }


def _state_metrics(results: list[SampleResult]) -> dict[str, Any]:
    scored = [r for r in results if r.expected_state]
    correct = sum(1 for r in scored if r.predicted_state == r.expected_state)
    confusion: dict[str, dict[str, int]] = {}
    for result in scored:
        row = confusion.setdefault(str(result.expected_state), {})
        row[result.predicted_state] = row.get(result.predicted_state, 0) + 1
    return {
        "scored_samples": len(scored),
        "state_accuracy": round(correct / len(scored), 4) if scored else 0.0,
        "confusion": confusion,
    }


def _leakage_warnings(manifest_groups: set[str]) -> list[str]:
    train_path = DEFAULT_DATA_DIR / "train.jsonl"
    if not train_path.exists():
        return []
    train_groups: set[str] = set()
    with train_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            group = json.loads(line).get("split_group")
            if group:
                train_groups.add(str(group))
    overlap = sorted(manifest_groups & train_groups)
    if not overlap:
        return []
    return [
        f"split_group '{group}' appears in both training data and manifest" for group in overlap
    ]


def _run_vad_replay(
    *,
    wav_dir: Path,
    samples: list[dict[str, Any]],
    silence_end_ms: int,
) -> dict[str, Any]:
    """Segment local desensitized WAVs with the configured VAD parameters.

    Reports FINAL output time (last segment end), over-segmentation and merged
    sentence rates versus expected_turns, and SenseVoice call count per minute.
    Requires no ASR models; timing is based on VAD segmentation only.
    """
    expected_by_id = {str(sample.get("id")): sample for sample in samples}
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        return {"skipped": True, "reason": f"no WAV files in {wav_dir}"}

    results: list[dict[str, Any]] = []
    total_duration_s = 0.0
    total_segments = 0
    for wav_path in wavs:
        sample = expected_by_id.get(wav_path.stem, {})
        with wave.open(str(wav_path), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        if channels != 1 or width != 2 or rate != 16_000:
            continue
        duration_s = len(frames) / (2 * rate)
        segmenter = VoiceActivitySegmenter(silence_end_ms=silence_end_ms)
        segments = 0
        final_output_ms = 0
        for offset in range(0, len(frames), rate * 2 * FRAME_MS // 1_000):
            frame = frames[offset : offset + rate * 2 * FRAME_MS // 1_000]
            if len(frame) < rate * 2 * FRAME_MS // 1_000:
                continue
            for segment in segmenter.consume(frame):
                segments += 1
                final_output_ms = max(final_output_ms, segment.start_offset_ms)
        for segment in segmenter.flush():
            segments += 1
            final_output_ms = max(final_output_ms, segment.start_offset_ms)
        expected_turns = int(sample.get("expected_turns") or 1)
        results.append(
            {
                "sample_id": wav_path.stem,
                "duration_s": round(duration_s, 2),
                "segments": segments,
                "expected_turns": expected_turns,
                "over_segmentation": max(0, segments - expected_turns),
                "merged_sentences": max(0, expected_turns - segments),
                "final_output_ms": final_output_ms,
            }
        )
        total_duration_s += duration_s
        total_segments += segments

    total_minutes = max(total_duration_s / 60.0, 1e-6)
    return {
        "skipped": False,
        "silence_end_ms": silence_end_ms,
        "sample_count": len(results),
        "total_duration_s": round(total_duration_s, 2),
        "over_segmentation_total": sum(item["over_segmentation"] for item in results),
        "merged_sentences_total": sum(item["merged_sentences"] for item in results),
        "sensevoice_calls_per_minute": round(total_segments / total_minutes, 2),
        "final_output_p95_ms": _latency_stats([item["final_output_ms"] for item in results])[
            "p95_ms"
        ],
        "samples": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the fraud decision pipeline")
    parser.add_argument("--manifest", required=True, help="Path to JSONL sample manifest")
    parser.add_argument("--output", help="Write JSON report to this path (default: stdout)")
    parser.add_argument(
        "--wav-dir",
        help="Directory of local desensitized {sample_id}.wav files for VAD replay",
    )
    parser.add_argument(
        "--vad-silence-end-ms",
        type=int,
        default=700,
        help="VAD silence threshold in ms (must be a 20 ms multiple); compare 500/600/700",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    samples = _load_manifest(manifest_path)
    classifier = get_default_classifier()

    results = [_replay_sample(sample, index) for index, sample in enumerate(samples, start=1)]

    report = {
        "environment": {
            "device": "cpu",
            "classifier_model": classifier.model_name,
            "git_commit": _git_commit(),
            "train_data_digest": _data_digest(DEFAULT_DATA_DIR / "train.jsonl"),
        },
        "sample_count": len(results),
        "latency": {
            "evidence_extract": _latency_stats([r.evidence_extract_ms for r in results]),
            "state_machine": _latency_stats([r.state_machine_ms for r in results]),
            "total_decision": _latency_stats([r.total_ms for r in results]),
        },
        "accuracy": {
            "labels": _label_metrics(results),
            "states": _state_metrics(results),
        },
        "leakage_warnings": _leakage_warnings(
            {str(s["split_group"]) for s in samples if s.get("split_group")}
        ),
        "samples": [
            {
                "id": r.sample_id,
                "scenario": r.scenario,
                "evidence_extract_ms": r.evidence_extract_ms,
                "state_machine_ms": r.state_machine_ms,
                "total_ms": r.total_ms,
                "predicted_state": r.predicted_state,
                "expected_state": r.expected_state,
                "predicted_labels": r.predicted_labels,
                "expected_labels": r.expected_labels,
            }
            for r in results
        ],
    }
    if args.wav_dir:
        report["vad"] = _run_vad_replay(
            wav_dir=Path(args.wav_dir),
            samples=samples,
            silence_end_ms=args.vad_silence_end_ms,
        )

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"report written to {args.output} ({len(results)} samples)")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
