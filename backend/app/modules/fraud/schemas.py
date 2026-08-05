from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VisualBoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "VisualBoundingBox":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding box maximums must not be less than minimums")
        return self


class VisualEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_event_id: str
    message_id: str | None = None
    request_id: str | None = None
    device_id: str
    occurred_at: datetime
    received_at: datetime
    source: Literal["ys7"]
    event_type: Literal["phone_call", "people_count", "person_detected"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    people_count: int | None = Field(default=None, ge=0)
    boxes: list[VisualBoundingBox] = Field(default_factory=list)
    image_url: str | None = None
    raw_event_ref: str


class FraudAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    source_event_id: str = Field(min_length=1, max_length=256)
    device_id: str = Field(min_length=1, max_length=256)
    occurred_at: datetime
    ended_at: datetime
    text: str = Field(min_length=1, max_length=2_000)
    elder_alone: bool = False
    language: str | None = Field(default=None, max_length=32)
    emotion: str | None = Field(default=None, max_length=32)
    audio_events: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("occurred_at", "ended_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("speech event times must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_interval(self) -> "FraudAnalyzeRequest":
        if self.ended_at < self.occurred_at:
            raise ValueError("ended_at must not be before occurred_at")
        return self


class FraudRiskSnapshot(BaseModel):
    session_id: str
    device_id: str
    state: str
    state_index: int
    state_label: str
    score: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    occurred_at: datetime
    transition_reason: str
    next_stage_conditions: list[str]
    evidence_chain: list[dict[str, Any]]
    state_history: list[dict[str, Any]]


class FraudAnalyzeData(BaseModel):
    status: Literal["accepted", "duplicate"]
    speech_event: dict[str, Any]
    risk: FraudRiskSnapshot


class FraudAudioChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=256)
    started_at: datetime
    elder_alone: bool = False

    @field_validator("started_at")
    @classmethod
    def require_started_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audio chunk started_at must include a timezone")
        return value


class TranscriptSegment(BaseModel):
    source_event_id: str
    occurred_at: datetime
    ended_at: datetime
    text: str
    language: str | None = None
    emotion: str | None = None
    audio_events: list[str] = Field(default_factory=list)


class FraudAudioChunkData(BaseModel):
    status: Literal["accepted", "duplicate"]
    chunk_id: str
    duration_ms: int
    transcript_segments: list[TranscriptSegment]
    risk: FraudRiskSnapshot | None
