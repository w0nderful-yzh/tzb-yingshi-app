import asyncio
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

FraudEvidenceKind = Literal[
    "identity_claim",
    "benefit_lure",
    "emergency_pretext",
    "sensitive_info_request",
    "amount_request",
    "credential_request",
    "remote_control_instruction",
    "money_instruction",
    "secrecy_control",
    "urgency_pressure",
    "protective_warning",
]


class FraudLlmFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: FraudEvidenceKind
    quote: str = Field(min_length=1, max_length=300)
    reason: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class FraudLlmReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["INCONCLUSIVE", "SUSPICIOUS", "HIGH_RISK", "PROTECTIVE"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=500)
    findings: list[FraudLlmFinding] = Field(default_factory=list, max_length=12)


@dataclass(frozen=True, slots=True)
class FraudLlmReviewRequest:
    review_id: str
    session_id: str
    device_id: str
    current_state: str
    at_ms: int
    transcript_segments: tuple[dict[str, Any], ...]
    evidence_chain: tuple[dict[str, Any], ...]
    visual_inputs: tuple[dict[str, Any], ...] = ()


class FraudLlmJudge(Protocol):
    @property
    def model_name(self) -> str: ...

    async def review(self, request: FraudLlmReviewRequest) -> FraudLlmReview: ...


class FraudLlmReviewQueue:
    def __init__(self, *, maxsize: int) -> None:
        self._queue: asyncio.Queue[FraudLlmReviewRequest] = asyncio.Queue(maxsize=maxsize)
        self._submitted: set[str] = set()

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def submit_nowait(self, request: FraudLlmReviewRequest) -> bool:
        if request.review_id in self._submitted:
            return False
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            return False
        self._submitted.add(request.review_id)
        return True

    async def get(self) -> FraudLlmReviewRequest:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


_KIND_STAGE = {
    "identity_claim": "contact",
    "benefit_lure": "contact",
    "emergency_pretext": "contact",
    "sensitive_info_request": "probing",
    "amount_request": "probing",
    "credential_request": "action",
    "remote_control_instruction": "action",
    "money_instruction": "action",
    "secrecy_control": "control",
    "urgency_pressure": "control",
    "protective_warning": "protective",
}


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def review_to_evidence(
    request: FraudLlmReviewRequest,
    review: FraudLlmReview,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    if review.verdict == "INCONCLUSIVE":
        return []
    transcript = "".join(
        _normalized_text(segment.get("text")) for segment in request.transcript_segments
    )
    evidence: list[dict[str, Any]] = []
    for index, finding in enumerate(review.findings, start=1):
        quote = _normalized_text(finding.quote)
        if not quote or quote not in transcript:
            continue
        matching_segment = next(
            (
                segment
                for segment in request.transcript_segments
                if quote in _normalized_text(segment.get("text"))
            ),
            None,
        )
        start_ms = (
            int(matching_segment["start_ms"])
            if matching_segment is not None
            else int(request.transcript_segments[0]["start_ms"])
        )
        end_ms = (
            int(matching_segment["end_ms"])
            if matching_segment is not None
            else int(request.transcript_segments[-1]["end_ms"])
        )
        protective = finding.kind == "protective_warning"
        if review.verdict == "PROTECTIVE" and not protective:
            continue
        evidence.append(
            {
                "evidence_id": f"ev-llm-{request.review_id[:16]}-{index:02d}",
                "kind": finding.kind,
                "stage": _KIND_STAGE[finding.kind],
                "strength": "protective" if protective else "medium",
                "polarity": "protective" if protective else "supporting",
                "source": "llm",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": finding.quote,
                "reason": finding.reason,
                "confidence": min(review.confidence, finding.confidence, 0.85),
                "llm_model": model_name,
                "llm_verdict": review.verdict,
                "llm_summary": review.summary,
            }
        )
    return evidence
