from datetime import datetime, timezone
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.modules.fall.multimodal_engine.schemas.risk_event import RiskModule


class MonitoringMode(str, Enum):
    SIMULATION = "SIMULATION"
    FILE = "FILE"
    LIVE = "LIVE"


class MonitoringStatus(str, Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class MonitoringSessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    mode: MonitoringMode = MonitoringMode.SIMULATION
    device_id: str = Field(min_length=1, max_length=64)
    enabled_modules: list[RiskModule]
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("started_at")
    @classmethod
    def started_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("started_at must include a timezone offset")
        return value


class MonitoringSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str = Field(validation_alias=AliasChoices("session_id", "id"))
    mode: MonitoringMode
    status: MonitoringStatus
    device_id: str
    enabled_modules: list[RiskModule]
    started_at: datetime
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @field_validator(
        "started_at",
        "ended_at",
        "created_at",
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
