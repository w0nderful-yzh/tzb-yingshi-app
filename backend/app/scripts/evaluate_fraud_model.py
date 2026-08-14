"""Offline evaluation of the fraud evidence pipeline.

Replays a fixed evaluation set through rule extraction, the lightweight
classifier and the fusion layer, then reports per-label and conversation-level
metrics. Run from the backend directory:

    uv run python -m app.scripts.evaluate_fraud_model \
        [--eval backend/evaluation/fraud/eval_public.jsonl] \
        [--threshold 0.2] [--report-dir backend/evaluation/fraud/reports]

Private, un-committable eval sets can be dropped into
backend/evaluation/private/ and are picked up automatically.

The report is JSON with: per-label P/R/F1 (rule / classifier / fusion),
micro/macro F1, hard-negative false positive list, state machine confusion
matrix for conversation groups, model version, data hash, thresholds and the
environment (CPU-only device type by default). No transcripts, IDs or keys are
written into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.modules.fraud.evidence_labels import EVIDENCE_LABELS
from app.modules.fraud.fraud_evidence import extract_speech_evidence
from app.modules.fraud.fraud_state_machine import FraudProcessStateMachine
from app.modules.fraud.speech_risk import build_speech_events
from app.modules.fraud.text_classifier import (
    DEFAULT_DATA_DIR,
    get_default_classifier,
    load_examples,
)

LABELS = tuple(EVIDENCE_LABELS)
REPORT_DIR = Path(__file__).resolve().parents[2] / "evaluation/fraud/reports"
PRIVATE_DIR = Path(__file__).resolve().parents[2] / "evaluation/private"
DEFAULT_EVAL = Path(__file__).resolve().parents[2] / "evaluation/fraud/eval_public.jsonl"


@dataclass(slots=True)
class _LabelStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def to_report(self) -> dict[str, float | int]:
        return {
            "support": self.tp + self.fn,
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f1(), 4),
        }


def _as_record(value: dict[str, Any]) -> dict[str, Any]:
    labels = list(value.get("labels") or [])
    unknown = sorted(set(labels).difference(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {', '.join(unknown)}")
    text = str(value.get("text", "")).strip()
    if not text:
        raise ValueError("Empty text")
    return {
        "text": text,
        "labels": labels,
        "source": str(value.get("source", "curated")),
        "scenario": str(value.get("scenario", "")),
        "split_group": str(value.get("split_group", "")),
        "asr_noisy": bool(value.get("asr_noisy", False)),
        "expected_state": value.get("expected_state"),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [_as_record(json.loads(line)) for line in path.read_text("utf-8").splitlines() if line]


def _evidence_kinds(fusion: list[dict[str, Any]]) -> set[str]:
    return {str(item["kind"]) for item in fusion}


def _run_eval(
    records: list[dict[str, Any]],
    classifier: Any,
    threshold: float,
) -> tuple[dict[str, dict[str, _LabelStats]], list[dict[str, Any]], list[dict[str, Any]]]:
    rule_stats = {label: _LabelStats() for label in LABELS}
    classifier_stats = {label: _LabelStats() for label in LABELS}
    fusion_stats = {label: _LabelStats() for label in LABELS}
    hard_neg_false_positives: list[dict[str, Any]] = []
    missed_strong_actions: list[dict[str, Any]] = []

    for record in records:
        expected = set(record["labels"])
        built = build_speech_events(
            [
                {
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "text": record["text"],
                    "transcript_status": "FINAL",
                }
            ],
            classifier=classifier,
        )[0]
        rule_evidence = extract_speech_evidence(built)
        predictions = built["classifier_predictions"]
        fusion = built["evidence_observations"]

        for kind, stats in rule_stats.items():
            present = kind in {str(item["kind"]) for item in rule_evidence}
            if present and kind in expected:
                stats.tp += 1
            elif present:
                stats.fp += 1
            elif kind in expected:
                stats.fn += 1
        classifier_kinds = {str(item["kind"]) for item in predictions}
        for kind, stats in classifier_stats.items():
            if kind in classifier_kinds and kind in expected:
                stats.tp += 1
            elif kind in classifier_kinds:
                stats.fp += 1
            elif kind in expected:
                stats.fn += 1
        fusion_kinds = _evidence_kinds(fusion)
        for kind, stats in fusion_stats.items():
            if kind in fusion_kinds and kind in expected:
                stats.tp += 1
            elif kind in fusion_kinds:
                stats.fp += 1
            elif kind in expected:
                stats.fn += 1

        if not expected:
            fp_kinds = sorted(fusion_kinds)
            if fp_kinds:
                hard_neg_false_positives.append(
                    {
                        "split_group": record["split_group"],
                        "scenario": record["scenario"],
                        "asr_noisy": record["asr_noisy"],
                        "fusion_false_positives": fp_kinds,
                    }
                )
        strong_expected = {
            "credential_request",
            "remote_control_instruction",
            "money_instruction",
        }.intersection(expected)
        if strong_expected and not strong_expected.intersection(fusion_kinds):
            missed_strong_actions.append(
                {
                    "split_group": record["split_group"],
                    "scenario": record["scenario"],
                    "asr_noisy": record["asr_noisy"],
                    "expected": sorted(strong_expected),
                    "fusion_kinds": sorted(fusion_kinds),
                }
            )

    return (
        {
            "rule": rule_stats,
            "classifier": classifier_stats,
            "fusion": fusion_stats,
        },
        hard_neg_false_positives,
        missed_strong_actions,
    )


def _macro_micro(
    stats: dict[str, _LabelStats],
) -> dict[str, float]:
    total_tp = sum(item.tp for item in stats.values())
    total_fp = sum(item.fp for item in stats.values())
    total_fn = sum(item.fn for item in stats.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    macro_f1 = sum(item.f1() for item in stats.values()) / len(stats)
    return {
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
    }


def _conversation_eval(
    records: list[dict[str, Any]],
    classifier: Any,
) -> tuple[dict[str, list[dict[str, str]]], float]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["split_group"] and record["expected_state"]:
            groups.setdefault(record["split_group"], []).append(record)
    matrix: dict[str, list[dict[str, str]]] = {}
    correct = 0
    total = 0
    for _group, turns in groups.items():
        machine = FraudProcessStateMachine(memory_ms=120_000)
        for turn in turns:
            built = build_speech_events(
                [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "text": turn["text"],
                        "transcript_status": "FINAL",
                    }
                ],
                classifier=classifier,
            )[0]
            for evidence in built["evidence_observations"]:
                machine.consume(dict(evidence))
        expected = str(turns[-1].get("expected_state"))
        actual = machine.current_state
        matrix.setdefault(expected, []).append({"expected": expected, "actual": actual})
        total += 1
        correct += 1 if actual == expected else 0
    return matrix, round(correct / total, 4) if total else 1.0


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _data_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((DEFAULT_DATA_DIR / "train.jsonl", Path(DEFAULT_EVAL))):
        if path.exists():
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in [args.eval, *(sorted(PRIVATE_DIR.glob("*.jsonl")) if PRIVATE_DIR.exists() else [])]:
        records.extend(load_records(path))
    if not records:
        print("No evaluation records found", file=sys.stderr)
        return 2

    classifier = get_default_classifier()
    stats, hard_negatives, missed_strong = _run_eval(records, classifier, args.threshold)
    matrix, state_accuracy = _conversation_eval(records, classifier)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "environment": {"device_type": "cpu", "language": "python"},
        "model": {
            "name": classifier.model_name,
            "classifier_threshold": args.threshold,
        },
        "data": {
            "train_records": len(load_examples(DEFAULT_DATA_DIR / "train.jsonl")),
            "eval_records": len(records),
            "eval_files": [str(path.name) for path in [args.eval] if path.exists()],
            "train_data_sha256": _data_sha256(),
        },
        "metrics": {
            method: {
                "per_label": {label: stats[label].to_report() for label in LABELS},
                "overall": _macro_micro(stats),
            }
            for method, stats in stats.items()
        },
        "state_machine": {
            "confusion_matrix": matrix,
            "accuracy": state_accuracy,
        },
        "hard_negative_false_positives": hard_negatives,
        "missed_strong_actions": missed_strong,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"fraud_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    print(f"Evaluated {len(records)} records, report: {report_path}")
    for method, value in report["metrics"].items():
        overall = value["overall"]
        print(
            f"  [{method}] micro_f1={overall['micro_f1']} macro_f1={overall['macro_f1']} "
            f"strong_action_recall="
            f"{_strong_action_recall(value['per_label'])}"
        )
    print(
        f"  [state] accuracy={state_accuracy} "
        f"hard_neg_fp={len(hard_negatives)} missed_strong={len(missed_strong)}"
    )
    return 0


def _strong_action_recall(per_label: dict[str, Any]) -> float:
    stats = per_label
    recalls = [
        float(stats[label]["recall"])
        for label in ("credential_request", "remote_control_instruction", "money_instruction")
    ]
    return round(sum(recalls) / len(recalls), 4)


if __name__ == "__main__":
    raise SystemExit(main())
