from __future__ import annotations

from typing import Any, TypedDict


class _StateInfo(TypedDict):
    index: int
    label: str
    score: int
    level: str
    decision: str


STATE_INFO: dict[str, _StateInfo] = {
    "S0_NORMAL": {
        "index": 0,
        "label": "正常交流",
        "score": 0,
        "level": "low",
        "decision": "normal",
    },
    "S1_OBSERVING": {
        "index": 1,
        "label": "待观察",
        "score": 20,
        "level": "low",
        "decision": "observe",
    },
    "S2_TRUST_BUILDING": {
        "index": 2,
        "label": "可疑身份接触",
        "score": 42,
        "level": "medium",
        "decision": "verify",
    },
    "S3_INFORMATION_PROBING": {
        "index": 3,
        "label": "敏感信息试探",
        "score": 58,
        "level": "medium",
        "decision": "warn",
    },
    "S4_ACTION_INDUCEMENT": {
        "index": 4,
        "label": "敏感操作诱导",
        "score": 75,
        "level": "high",
        "decision": "block",
    },
    "S5_CRITICAL_CONTROL": {
        "index": 5,
        "label": "高危控制与执行",
        "score": 95,
        "level": "critical",
        "decision": "intervene",
    },
}

CONTACT_KINDS = {"identity_claim", "benefit_lure", "emergency_pretext"}
ACTION_KINDS = {"credential_request", "remote_control_instruction", "money_instruction"}
MIN_TRANSITION_CONFIDENCE = {"weak": 0.4, "medium": 0.45, "strong": 0.5}


class FraudProcessStateMachine:
    """Replayable evidence state machine for one conversation or visit session."""

    def __init__(self, *, elder_alone: bool = False, memory_ms: int = 120_000):
        self.elder_alone = elder_alone
        self.memory_ms = memory_ms
        self.current_state = "S0_NORMAL"
        self.evidence_chain: list[dict[str, Any]] = []
        self.state_history: list[dict[str, Any]] = []

    @property
    def state_index(self) -> int:
        return int(STATE_INFO[self.current_state]["index"])

    def _recent(self, at_ms: int) -> list[dict[str, Any]]:
        lower_bound = at_ms - self.memory_ms
        return [
            item
            for item in self.evidence_chain
            if int(item["end_ms"]) >= lower_bound
            and item.get("polarity") != "protective"
            and item.get("used_for_transition", True)
        ]

    def _advance(self, target: str, evidence: dict[str, Any], reason: str) -> None:
        if int(STATE_INFO[target]["index"]) <= self.state_index:
            return
        previous = self.current_state
        self.current_state = target
        self.state_history.append(
            {
                "transition_id": f"transition-{len(self.state_history) + 1:03d}",
                "from": previous,
                "to": target,
                "at_ms": int(evidence["end_ms"]),
                "trigger_evidence_id": evidence["evidence_id"],
                "trigger_kind": evidence["kind"],
                "reason": reason,
            }
        )

    def _critical_reason(self, recent: list[dict[str, Any]]) -> str | None:
        actions = [
            item
            for item in recent
            if item["kind"] in ACTION_KINDS and item.get("strength") == "strong"
        ]
        if not actions:
            return None
        if any(
            item["kind"] == "secrecy_control" and item.get("strength") == "strong"
            for item in recent
        ):
            return "敏感操作叠加保密或隔离家属要求。"
        if self.elder_alone:
            return "老人独处时出现明确的资金、凭证或远控操作指令。"
        if any(
            item["kind"] == "visitor_presence" and item.get("escalation_eligible")
            for item in recent
        ):
            return "明确操作指令与连续摄像头的异常访客情境重叠。"
        if len({item["kind"] for item in actions}) >= 2:
            return "短时间内连续出现多种敏感操作指令。"
        return None

    def consume(self, evidence: dict[str, Any]) -> None:
        evidence = dict(evidence)
        strength = str(evidence.get("strength", "weak"))
        minimum_confidence = MIN_TRANSITION_CONFIDENCE.get(strength, 1.0)
        evidence["used_for_transition"] = (
            bool(evidence.get("used_for_transition", True))
            and evidence.get("polarity") != "protective"
            and float(evidence.get("confidence", 1.0)) >= minimum_confidence
        )
        self.evidence_chain.append(evidence)
        if not evidence["used_for_transition"]:
            return

        recent = self._recent(int(evidence["end_ms"]))
        kinds = {item["kind"] for item in recent}
        weak_kinds = {
            item["kind"]
            for item in recent
            if item.get("strength") == "weak" and item.get("stage") != "context"
        }

        if evidence["kind"] in ACTION_KINDS and strength == "strong":
            self._advance(
                "S4_ACTION_INDUCEMENT",
                evidence,
                "出现明确的转账、取现、凭证提交或远程控制指令。",
            )
        elif evidence["kind"] in CONTACT_KINDS and "phone_call_active" in kinds:
            self._advance(
                "S2_TRUST_BUILDING",
                evidence,
                "通话场景中出现可疑身份、利益诱导或紧急剧本。",
            )
        elif evidence["kind"] == "sensitive_info_request" and strength in {
            "medium",
            "strong",
        }:
            self._advance(
                "S3_INFORMATION_PROBING",
                evidence,
                "对话开始索要身份、账户或资产信息。",
            )
        elif (
            evidence["kind"] == "amount_request"
            and strength in {"medium", "strong"}
            and CONTACT_KINDS.intersection(kinds)
        ):
            self._advance(
                "S2_TRUST_BUILDING",
                evidence,
                "可疑身份或紧急剧本之后出现具体金额。",
            )
        elif len(CONTACT_KINDS.intersection(kinds)) >= 2 and any(
            item["kind"] in CONTACT_KINDS and item.get("strength") == "medium" for item in recent
        ):
            self._advance(
                "S2_TRUST_BUILDING",
                evidence,
                "多个身份、利益或紧急剧本证据连续出现。",
            )
        elif evidence.get("strength") == "medium" or len(weak_kinds) >= 2:
            self._advance(
                "S1_OBSERVING",
                evidence,
                "多个弱证据或一个中等证据在近期对话中出现。",
            )

        critical_reason = self._critical_reason(recent)
        if critical_reason:
            self._advance("S5_CRITICAL_CONTROL", evidence, critical_reason)

    def snapshot(self) -> dict[str, Any]:
        info = STATE_INFO[self.current_state]
        return {
            "state": self.current_state,
            "state_index": info["index"],
            "state_label": info["label"],
            "score": info["score"],
            "level": info["level"],
            "decision": info["decision"],
            "evidence_chain": list(self.evidence_chain),
            "state_history": list(self.state_history),
            "transition_reason": (
                self.state_history[-1]["reason"]
                if self.state_history
                else "尚未形成可升级的证据链。"
            ),
            "next_stage_conditions": self._next_stage_conditions(),
        }

    def _next_stage_conditions(self) -> list[str]:
        return {
            "S0_NORMAL": ["第二个独立弱证据", "亲属出事等中等证据"],
            "S1_OBSERVING": ["可疑身份后的金额或信息试探", "明确操作指令"],
            "S2_TRUST_BUILDING": ["身份或账户信息索要", "转账、验证码或远控指令"],
            "S3_INFORMATION_PROBING": ["转账、验证码或远控指令"],
            "S4_ACTION_INDUCEMENT": ["要求保密", "老人独处", "异常访客", "紧急施压"],
            "S5_CRITICAL_CONTROL": ["已达到高危干预条件"],
        }[self.current_state]
