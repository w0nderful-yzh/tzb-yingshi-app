from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class GuardSessionState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"


class GuardCapabilityState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class GuardSessionStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, min_length=8, max_length=128)


class GuardCapabilityStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: GuardCapabilityState
    enabled_for_session: bool
    detail: str


class MultimodalGuardSessionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "multimodal_guard_session_v1"
    session_id: str | None = None
    active: bool = False
    state: GuardSessionState = GuardSessionState.STOPPED
    camera_analysis: GuardCapabilityStatus
    radar_worker: GuardCapabilityStatus
    radar_participation: GuardCapabilityStatus
    fusion: GuardCapabilityStatus
    reason_codes: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
