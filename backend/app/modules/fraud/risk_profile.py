"""Recent cross-session risk profile (Phase 6, data-gated).

A new session always starts at S0; the recent profile is converted into
context evidence with time decay so it can lower composition thresholds
without letting a new session start at a high state. The profile never
contains full transcripts — only state, evidence kinds, occurrence time and
the desensitized summary digest.
"""

from __future__ import annotations

from typing import Any

from app.modules.fraud.ports import RecentFraudContext


def recent_context_evidence(context: RecentFraudContext, *, at_ms: int) -> list[dict[str, Any]]:
    """Shape a RecentFraudContext into weak context evidence.

    The evidence is weak, time-decayed by the standard staged window for the
    context stage, and never alone advances the state machine (weak evidence
    only combines with other signals). `used_for_transition` stays False so it
    cannot push S4/S5 by itself.
    """
    if context.recent_risk_events <= 0:
        return []
    if context.last_occurred_at is None:
        return []
    end_ms = round(context.last_occurred_at.timestamp() * 1000)
    kinds = ", ".join(context.last_kinds) if context.last_kinds else "unknown"
    return [
        {
            "evidence_id": "ev-context-recent-risk",
            "kind": "recent_risk_profile",
            "stage": "context",
            "strength": "weak",
            "polarity": "supporting",
            "source": "risk_profile",
            "start_ms": at_ms - 24 * 60 * 60_000,
            "end_ms": end_ms,
            "text": (
                f"近 24 小时该设备出现 {context.recent_risk_events} 次风险事件，"
                f"最近一次等级 {context.last_risk_level or 'unknown'}（{kinds}）。"
            ),
            "reason": "近期风险画像仅作为组合门槛参考，不单独推动状态升级。",
            "confidence": 0.6,
            "used_for_transition": False,
        }
    ]
