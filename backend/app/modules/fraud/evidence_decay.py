"""Per-stage evidence windows and confidence decay.

Phase 5 of the latency-accuracy roadmap: replace the single 120 s memory
window with staged windows per evidence stage. Evidence older than its stage
window expires (marked `expired`), and within the window the confidence decays
towards the end so stale contact or probing evidence cannot keep a session in
a high state. The state machine keeps consuming standard evidence; raw and
decayed confidence are both retained in the evidence chain for explainability.
"""

from __future__ import annotations

from typing import Any

STAGE_WINDOWS_MS: dict[str, int] = {
    "contact": 300_000,
    "probing": 180_000,
    "action": 120_000,
    "control": 120_000,
    "protective": 300_000,
    "context": 120_000,
}

STAGE_DECAY_STRENGTH: dict[str, float] = {
    "contact": 0.7,
    "probing": 0.5,
    "action": 0.2,
    "control": 0.5,
    "protective": 0.0,
    "context": 0.5,
}

MAX_WINDOW_MS = max(STAGE_WINDOWS_MS.values())


def stage_window_ms(stage: str) -> int:
    return STAGE_WINDOWS_MS.get(str(stage), MAX_WINDOW_MS)


def apply_stage_windows(
    evidence: list[dict[str, Any]],
    *,
    at_ms: int,
) -> list[dict[str, Any]]:
    """Mark expired evidence and attach decayed confidence.

    Mutates and returns the same list so the chain keeps every item for
    explainability. Expired items get `used_for_transition=False`; live items
    get `decayed_confidence` (raw `confidence` is preserved) plus `window_ms`.
    """
    for item in evidence:
        if not isinstance(item, dict):
            continue
        window_ms = stage_window_ms(str(item.get("stage", "context")))
        item["window_ms"] = window_ms
        end_ms = int(item.get("end_ms", at_ms))
        age_ms = max(0, at_ms - end_ms)
        raw_confidence = float(item.get("confidence", 1.0))
        if age_ms > window_ms:
            item["expired"] = True
            item["decayed_confidence"] = round(0.0, 4)
            item["used_for_transition"] = False
            continue
        strength = STAGE_DECAY_STRENGTH.get(str(item.get("stage", "context")), 0.5)
        fraction = age_ms / window_ms if window_ms else 0.0
        decayed = raw_confidence * (1.0 - strength * fraction)
        item["expired"] = False
        item["decayed_confidence"] = round(decayed, 4)
    return evidence
