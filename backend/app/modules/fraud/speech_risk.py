from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from .evidence_fusion import fuse_speech_evidence
from .fraud_evidence import extract_speech_evidence
from .text_classifier import LightweightEvidenceClassifier, get_default_classifier

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.json")


def _matched_category_terms(text: str, config: dict[str, Any]) -> list[str]:
    matches = [term for term in config.get("terms", []) if term.lower() in text]
    for pattern in config.get("patterns", []):
        matches.extend(match.group(0) for match in re.finditer(pattern, text))
    return list(dict.fromkeys(matches))


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        return cast(dict[str, Any], json.load(source))


def match_speech_categories(text: str, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return deterministic observations without producing a risk score."""
    rules = rules or load_rules()
    normalized = text.lower()
    matched = {
        category: terms
        for category, config in rules["categories"].items()
        if (terms := _matched_category_terms(normalized, config))
    }
    warning_terms = rules.get("context_rules", {}).get("anti_fraud_warning", [])
    context_adjustments = (
        ["anti_fraud_warning"] if any(term.lower() in normalized for term in warning_terms) else []
    )
    return {
        "matched_terms": matched,
        "matched_categories": sorted(matched),
        "context_adjustments": context_adjustments,
    }


def build_speech_events(
    segments: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
    classifier: LightweightEvidenceClassifier | None = None,
    event_id_offset: int = 0,
) -> list[dict[str, Any]]:
    classifier = classifier or get_default_classifier()
    events: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=event_id_offset + 1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        observations = match_speech_categories(text, rules)
        event = {
            "event_id": f"speech-{index:03d}",
            "start_ms": int(segment["start_ms"]),
            "end_ms": int(segment["end_ms"]),
            "text": text,
            "language": segment.get("language"),
            "emotion": segment.get("emotion"),
            "audio_events": list(segment.get("audio_events") or []),
            "transcript_status": str(segment.get("transcript_status", "FINAL")),
            "risk_tags": observations["matched_categories"],
            "matched_terms": observations["matched_terms"],
            "context_adjustments": observations["context_adjustments"],
            "classifier_model": classifier.model_name,
        }
        rule_evidence = extract_speech_evidence(event)
        predictions = classifier.predict(text, threshold=0.2)
        event["classifier_predictions"] = predictions
        event["evidence_observations"] = fuse_speech_evidence(event, rule_evidence, predictions)
        events.append(event)
    return events
