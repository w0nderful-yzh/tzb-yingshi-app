from datetime import datetime, timezone
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class RiskModule(str, Enum):
    FALL = "FALL"
    MENTAL_STATE = "MENTAL_STATE"
    FRAUD = "FRAUD"
    DEVICE = "DEVICE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EventSource(str, Enum):
    SIMULATION = "SIMULATION"
    ALGORITHM = "ALGORITHM"
    EZVIZ = "EZVIZ"


class RiskEventStatus(str, Enum):
    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    FALSE_ALARM = "FALSE_ALARM"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=200)
    value: int | float | str | bool | None = None
    unit: str | None = Field(default=None, max_length=32)


class RiskEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^1\.0$")
    event_id: str = Field(min_length=1, max_length=80)
    session_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    module: RiskModule
    event_type: str = Field(min_length=1, max_length=64)
    occurred_at: datetime
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    summary: str = Field(min_length=1, max_length=500)
    evidence: list[EvidenceItem]
    recommended_action: str | None = Field(default=None, max_length=500)
    snapshot_path: str | None = Field(default=None, max_length=500)
    clip_path: str | None = Field(default=None, max_length=500)
    model_version: str = Field(min_length=1, max_length=64)
    source: EventSource

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone offset")
        return value


class RiskEventStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RiskEventStatus
    handling_note: str | None = Field(default=None, max_length=500)


class RiskEventBulkDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("event_ids")
    @classmethod
    def event_ids_must_be_unique_and_valid(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 80 for value in normalized):
            raise ValueError("event_ids must contain non-empty IDs up to 80 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("event_ids must not contain duplicates")
        return normalized


class RiskEventDeleteResponse(BaseModel):
    deleted_count: int = Field(ge=0)


class RiskEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    schema_version: str = "1.0"
    event_id: str
    session_id: str
    device_id: str
    module: RiskModule
    event_type: str
    occurred_at: datetime
    received_at: datetime
    risk_score: float
    risk_level: RiskLevel
    summary: str
    evidence: list[EvidenceItem] = Field(
        validation_alias=AliasChoices("evidence", "evidence_json")
    )
    recommended_action: str | None
    snapshot_path: str | None
    clip_path: str | None
    model_version: str
    source: EventSource
    status: RiskEventStatus
    handled_at: datetime | None
    handling_note: str | None
    updated_at: datetime

    @field_validator(
        "occurred_at",
        "received_at",
        "handled_at",
        "updated_at",
        mode="before",
    )
    @classmethod
    def database_times_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
