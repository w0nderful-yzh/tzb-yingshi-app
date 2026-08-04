from __future__ import annotations

import re
from typing import Any

_CREDENTIAL_REQUEST_PATTERNS = [
    re.compile(
        r"(?:告诉|提供|发送|发给|报给|输入|提交|念给|给我|说一下)"
        r"[^\u3002！？]{0,12}(?:验证码|密码|身份证号|银行卡号)"
    ),
    re.compile(
        r"(?:验证码|密码|身份证号|银行卡号)"
        r"[^\u3002！？]{0,12}(?:告诉|提供|发给|报给|输入|提交|念给|给我|是多少)"
    ),
]

_REMOTE_CONTROL_PATTERN = re.compile(
    r"(?:打开|开启|下载|安装|点击|共享|允许|同意)"
    r"[^\u3002！？]{0,12}(?:屏幕共享|远程控制|远程软件|会议软件)"
)

_MONEY_INSTRUCTION_PATTERNS = [
    re.compile(
        r"(?:马上|立即|赶紧|立刻|现在|必须|务必|请|先|快点|需要)"
        r"[^\u3002！？]{0,18}(?:转账|汇款|取钱|取现|打款|付款|充值)"
    ),
    re.compile(
        r"(?:把|将)[^\u3002！？]{0,12}(?:钱|资金|存款|现金)"
        r"[^\u3002！？]{0,10}(?:转|汇|打|存)"
    ),
    re.compile(r"(?:转账|汇款|取钱|取现|打款|付款|充值)(?:给|到|至|一下|过去|过来)"),
]


def _make_evidence(
    event: dict[str, Any],
    kind: str,
    stage: str,
    strength: str,
    reason: str,
    *,
    confidence: float = 0.8,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    return {
        "evidence_id": f"ev-{event_id}-{kind}",
        "kind": kind,
        "stage": stage,
        "strength": strength,
        "polarity": "supporting" if strength != "protective" else "protective",
        "source": "speech",
        "start_ms": int(event["start_ms"]),
        "end_ms": int(event["end_ms"]),
        "speech_event_id": event_id,
        "text": str(event.get("text", "")),
        "reason": reason,
        "confidence": confidence,
        "matched_terms": event.get("matched_terms", {}),
    }


def _matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def extract_speech_evidence(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert keyword observations into semantic evidence without deciding risk."""
    text = str(event.get("text", ""))
    tags = set(event.get("risk_tags", []))
    if "anti_fraud_warning" in event.get("context_adjustments", []):
        return [
            _make_evidence(
                event,
                "protective_warning",
                "protective",
                "protective",
                "当前语句是防诈提醒或否定性建议，不支持风险升级。",
                confidence=0.95,
            )
        ]

    evidence: list[dict[str, Any]] = []
    if "identity_impersonation" in tags:
        evidence.append(
            _make_evidence(
                event,
                "identity_claim",
                "contact",
                "weak",
                "出现机构、熟人或权威身份表述。",
            )
        )
    if {"health_sales", "reward_or_refund"}.intersection(tags):
        evidence.append(
            _make_evidence(
                event,
                "benefit_lure",
                "contact",
                "weak",
                "出现补贴、收益、保健或养老优惠等利益诱因。",
            )
        )
    if "family_emergency" in tags:
        evidence.append(
            _make_evidence(
                event,
                "emergency_pretext",
                "contact",
                "medium",
                "以亲属事故、被抓或抢救等紧急情境建立压力。",
                confidence=0.9,
            )
        )

    if "credential" in tags:
        credential_request = _matches_any(_CREDENTIAL_REQUEST_PATTERNS, text)
        remote_instruction = bool(_REMOTE_CONTROL_PATTERN.search(text))
        if remote_instruction:
            evidence.append(
                _make_evidence(
                    event,
                    "remote_control_instruction",
                    "action",
                    "strong",
                    "明确要求开启屏幕共享或远程控制。",
                    confidence=0.98,
                )
            )
        elif credential_request and re.search(r"验证码|密码", text):
            evidence.append(
                _make_evidence(
                    event,
                    "credential_request",
                    "action",
                    "strong",
                    "明确索要验证码、密码或账户凭证。",
                    confidence=0.96,
                )
            )
        elif credential_request:
            evidence.append(
                _make_evidence(
                    event,
                    "sensitive_info_request",
                    "probing",
                    "medium",
                    "明确索要身份证号、银行卡号等敏感账户信息。",
                    confidence=0.9,
                )
            )
        else:
            evidence.append(
                _make_evidence(
                    event,
                    "credential_mention",
                    "probing",
                    "weak",
                    "仅提及账户凭证，未识别到明确索要动作。",
                    confidence=0.55,
                )
            )

    if "amount_expression" in tags:
        evidence.append(
            _make_evidence(
                event,
                "amount_request",
                "probing",
                "medium",
                "对话出现具体金额，作为资产操作的中等证据。",
            )
        )

    if "money_operation" in tags:
        is_instruction = _matches_any(_MONEY_INSTRUCTION_PATTERNS, text)
        if "安全账户" in text and re.search(r"转|汇|打|存", text):
            is_instruction = True
        evidence.append(
            _make_evidence(
                event,
                "money_instruction" if is_instruction else "money_topic",
                "action" if is_instruction else "probing",
                "strong" if is_instruction else "weak",
                (
                    "明确要求转账、取现、汇款或其他资金操作。"
                    if is_instruction
                    else "仅提及资金或银行操作，未识别到明确指令。"
                ),
                confidence=0.95 if is_instruction else 0.5,
            )
        )

    if "secrecy" in tags:
        evidence.append(
            _make_evidence(
                event,
                "secrecy_control",
                "control",
                "strong",
                "要求对家人保密、不报警或不挂断通话。",
                confidence=0.98,
            )
        )
    if "urgency" in tags:
        evidence.append(
            _make_evidence(
                event,
                "urgency_pressure",
                "control",
                "weak",
                "出现立即执行、后果威胁或时限压力。",
            )
        )
    return evidence


def context_evidence(
    *,
    elder_alone: bool,
    visitor_track_ids: list[int],
    source_mode: str,
    end_ms: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if elder_alone:
        evidence.append(
            {
                "evidence_id": "ev-context-elder-alone",
                "kind": "elder_alone",
                "stage": "context",
                "strength": "weak",
                "polarity": "supporting",
                "source": "context",
                "start_ms": 0,
                "end_ms": end_ms,
                "text": "当前设置为老人独处。",
                "reason": "老人独处是高风险操作的脆弱性语境。",
                "confidence": 1.0,
            }
        )
    if visitor_track_ids:
        eligible = source_mode == "continuous_camera"
        evidence.append(
            {
                "evidence_id": "ev-context-visitor",
                "kind": "visitor_presence",
                "stage": "context",
                "strength": "weak",
                "polarity": "supporting",
                "source": "video",
                "start_ms": 0,
                "end_ms": end_ms,
                "text": f"风险时段检测到重叠新轨迹：{visitor_track_ids}。",
                "reason": (
                    "连续摄像头中的重叠新轨迹可作为入户情境。"
                    if eligible
                    else "当前为录像文件，轨迹仅展示，不单独推动高危状态。"
                ),
                "confidence": 0.9 if eligible else 0.45,
                "track_ids": visitor_track_ids,
                "escalation_eligible": eligible,
            }
        )
    return evidence
