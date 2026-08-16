from __future__ import annotations

from typing import Any

from .evidence_labels import EVIDENCE_LABELS

CLASSIFIER_MIN_CONFIDENCE = {
    "weak": 0.50,
    "medium": 0.45,
    "strong": 0.40,
    "protective": 0.40,
}


def _downgrade_strength(strength: str, confidence: float) -> str:
    if confidence >= 0.5 or strength in {"weak", "protective"}:
        return strength
    return {"strong": "medium", "medium": "weak"}.get(strength, strength)


def _classifier_evidence(event: dict[str, Any], kind: str, confidence: float) -> dict[str, Any]:
    metadata = EVIDENCE_LABELS[kind]
    strength = _downgrade_strength(str(metadata["strength"]), confidence)
    return {
        "evidence_id": f"ev-{event['event_id']}-{kind}",
        "kind": kind,
        "stage": metadata["stage"],
        "strength": strength,
        "polarity": "protective" if kind == "protective_warning" else "supporting",
        "source": "speech",
        "start_ms": int(event["start_ms"]),
        "end_ms": int(event["end_ms"]),
        "speech_event_id": str(event["event_id"]),
        "text": str(event.get("text", "")),
        "reason": f"轻量文本分类器识别为：{metadata['description']}。",
        "confidence": round(confidence, 4),
        "rule_confidence": 0.0,
        "classifier_confidence": round(confidence, 4),
        "detectors": ["text_classifier"],
        "fusion_method": "classifier_only",
        "transcript_status": event.get("transcript_status", "FINAL"),
    }


def fuse_speech_evidence(
    event: dict[str, Any],
    rule_evidence: list[dict[str, Any]],
    classifier_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse deterministic rule evidence with classifier probabilities."""
    rule_by_kind = {str(item["kind"]): dict(item) for item in rule_evidence}
    classifier_scores = {
        str(item["kind"]): float(item["confidence"])
        for item in classifier_predictions
        if item.get("kind") in EVIDENCE_LABELS
    }

    protective_confidence = classifier_scores.get("protective_warning", 0.0)
    protective: dict[str, Any] | None = None
    if "protective_warning" in rule_by_kind or protective_confidence >= 0.40:
        protective_item = rule_by_kind.get("protective_warning") or _classifier_evidence(
            event, "protective_warning", protective_confidence
        )
        rule_confidence = float(protective_item.get("confidence", 0.0))
        fused_confidence = min(
            0.99,
            max(rule_confidence, protective_confidence)
            + 0.04 * min(rule_confidence, protective_confidence),
        )
        protective = {
            **protective_item,
            "confidence": round(fused_confidence, 4),
            "rule_confidence": round(rule_confidence, 4),
            "classifier_confidence": round(protective_confidence, 4),
            "detectors": [
                detector
                for detector, present in (
                    ("rule", "protective_warning" in rule_by_kind),
                    ("text_classifier", protective_confidence > 0),
                )
                if present
            ],
            "fusion_method": "protective_override",
        }

    fused: list[dict[str, Any]] = []
    for kind in EVIDENCE_LABELS:
        if kind == "protective_warning":
            continue
        rule_item = rule_by_kind.get(kind)
        classifier_confidence = classifier_scores.get(kind, 0.0)
        if rule_item is None:
            strength = str(EVIDENCE_LABELS[kind]["strength"])
            if classifier_confidence < CLASSIFIER_MIN_CONFIDENCE[strength]:
                continue
            fused.append(_classifier_evidence(event, kind, classifier_confidence))
            continue

        rule_confidence = float(rule_item.get("confidence", 0.0))
        detectors = ["rule"]
        fusion_method = "rule_only"
        fused_confidence = rule_confidence
        if classifier_confidence > 0:
            detectors.append("text_classifier")
            fusion_method = "rule_classifier_agreement"
            fused_confidence = min(
                0.99,
                max(rule_confidence, classifier_confidence)
                + 0.05 * min(rule_confidence, classifier_confidence),
            )
        rule_item.update(
            {
                "confidence": round(fused_confidence, 4),
                "rule_confidence": round(rule_confidence, 4),
                "classifier_confidence": round(classifier_confidence, 4),
                "detectors": detectors,
                "fusion_method": fusion_method,
            }
        )
        fused.append(rule_item)

    if protective is not None:
        fused.insert(0, protective)
    return fused
