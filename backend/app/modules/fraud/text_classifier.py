from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .evidence_labels import CLASSIFIER_LABELS

DEFAULT_DATA_DIR = Path(__file__).with_name("data")
MODEL_NAME = "char_tfidf_logreg_evidence_v1"


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            labels = list(record.get("labels", []))
            unknown = sorted(set(labels).difference(CLASSIFIER_LABELS))
            if not text:
                raise ValueError(f"Empty text at {path}:{line_number}")
            if unknown:
                raise ValueError(f"Unknown labels at {path}:{line_number}: {', '.join(unknown)}")
            records.append({"text": text, "labels": labels})
    if not records:
        raise ValueError(f"No classifier examples found in {path}")
    return records


class LightweightEvidenceClassifier:
    """Small multi-label Chinese evidence classifier using character n-grams."""

    def __init__(self) -> None:
        self.model_name = MODEL_NAME
        self.vectorizer: Any = None
        self.model: Any = None
        self.binarizer: Any = None

    def fit(self, examples: list[dict[str, Any]]) -> LightweightEvidenceClassifier:
        try:
            from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
                TfidfVectorizer,
            )
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
            from sklearn.multiclass import OneVsRestClassifier  # type: ignore[import-untyped]
            from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The lightweight classifier requires scikit-learn. "
                "Run: pip install -r requirements-models.txt"
            ) from exc

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=1,
            max_features=12_000,
            sublinear_tf=True,
        )
        self.binarizer = MultiLabelBinarizer(classes=CLASSIFIER_LABELS)
        features = self.vectorizer.fit_transform([item["text"] for item in examples])
        targets = self.binarizer.fit_transform([item["labels"] for item in examples])
        self.model = OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=500,
                random_state=42,
            )
        )
        self.model.fit(features, targets)
        return self

    def predict_scores(self, text: str) -> dict[str, float]:
        if self.vectorizer is None or self.model is None or self.binarizer is None:
            raise RuntimeError("Classifier must be fitted before prediction")
        normalized = " ".join(str(text).split())
        if not normalized:
            return {label: 0.0 for label in CLASSIFIER_LABELS}
        probabilities = self.model.predict_proba(self.vectorizer.transform([normalized]))[0]
        return {
            label: round(float(probability), 4)
            for label, probability in zip(self.binarizer.classes_, probabilities, strict=True)
        }

    def predict(self, text: str, *, threshold: float = 0.2) -> list[dict[str, Any]]:
        scores = self.predict_scores(text)
        return [
            {"kind": kind, "confidence": confidence, "source": "text_classifier"}
            for kind, confidence in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if confidence >= threshold
        ]


@lru_cache(maxsize=1)
def get_default_classifier() -> LightweightEvidenceClassifier:
    examples = load_examples(DEFAULT_DATA_DIR / "train.jsonl")
    return LightweightEvidenceClassifier().fit(examples)
