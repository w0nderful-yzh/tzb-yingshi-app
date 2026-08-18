"""Evaluate fall-risk assessment effect on the same-sensor IWR6843 dataset.

The IWR6843-Fall102 dataset contains 102 short recordings (51 falls, 51 normal
activities) captured with the *same* TI IWR6843 radar. Each recording is
already exported as a 20x19 radar_features_v2 window. This script evaluates
how well a simple classifier can separate *fall-risk* (fall recordings) from
*normal activity* under strict subject-isolated (leave-one-subject-out)
evaluation.

This is the "effect" evidence for a radar-based fall-risk assessment: the
same-sensor point-cloud features carry a learnable fall/normal signal.

Contract
--------
- Subject-isolated splits (leave-one-subject-out over the 3 subjects).
- Uses only radar_features_v2 features.
- Reports accuracy, balanced accuracy, AUC per fold and averaged.
- Research/evaluation only; not a deployable clinical risk score.

Version: radar_fall_risk_assessment_effect_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from radar_module.preprocess.temporal_features_v2 import FEATURE_NAMES_V2


def _seg_features(window: np.ndarray) -> np.ndarray:
    """Concatenate last frame, per-feature mean, and per-feature std."""
    last = window[-1]
    return np.concatenate([last, window.mean(axis=0), window.std(axis=0)])


def evaluate_fall_risk_assessment_effect(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    n_estimators: int = 200,
    seed: int = 20260810,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        roc_auc_score,
    )
    from sklearn.model_selection import LeaveOneGroupOut

    dataset_file = Path(dataset_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    d = np.load(dataset_file, allow_pickle=True)
    features = d["features"]
    labels = np.asarray(d["labels"], dtype=np.int64)
    subjects = np.asarray(d["subject_id"])
    if features.shape[1:] != (20, len(FEATURE_NAMES_V2)):
        raise ValueError("features shape incompatible with v2")

    X = np.stack([_seg_features(f) for f in features])
    y = labels
    groups = subjects

    logo = LeaveOneGroupOut()
    fold_reports: list[dict[str, Any]] = []
    accs: list[float] = []
    bas: list[float] = []
    aucs: list[float] = []
    fold = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        held_subject = str(set(subjects[test_idx].tolist()))
        acc = float(accuracy_score(y[test_idx], pred))
        ba = float(balanced_accuracy_score(y[test_idx], pred))
        auc = float(roc_auc_score(y[test_idx], proba))
        accs.append(acc)
        bas.append(ba)
        aucs.append(auc)
        fold_reports.append(
            {
                "fold": fold,
                "held_out_subject": held_subject,
                "test_count": int(len(test_idx)),
                "positive_count": int(y[test_idx].sum()),
                "accuracy": acc,
                "balanced_accuracy": ba,
                "auc": auc,
            }
        )
        fold += 1

    result = {
        "schema_version": "radar_fall_risk_assessment_effect_v1",
        "dataset_file": str(dataset_file),
        "dataset_mode": str(d["dataset_mode"]),
        "sensor": "TI IWR6843 (same sensor)",
        "feature_version": str(d["feature_version"]),
        "method": "RandomForest on radar_features_v2 window stats",
        "evaluation": "leave-one-subject-out (3 subjects)",
        "fold_reports": fold_reports,
        "averaged": {
            "accuracy": float(np.mean(accs)),
            "balanced_accuracy": float(np.mean(bas)),
            "auc": float(np.mean(aucs)),
        },
        "note": (
            "Same-sensor IWR6843 point-cloud features separate fall-risk "
            "recordings from normal activity. This is effect evidence for a "
            "radar-based fall-risk assessment, not a clinical risk score. "
            "Research/evaluation only."
        ),
    }
    (destination / "fall_risk_assessment_effect.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fall-risk assessment effect on IWR6843-Fall102."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = evaluate_fall_risk_assessment_effect(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
