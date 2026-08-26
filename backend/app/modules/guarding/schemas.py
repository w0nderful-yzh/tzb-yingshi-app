from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class LifecycleState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class CapabilityStatus(BaseModel):
    state: LifecycleState
    enabled: bool
    detail: str


class GuardianSessionStatus(BaseModel):
    session_id: str | None = None
    active: bool = False
    state: LifecycleState = LifecycleState.STOPPED
    camera_analysis: CapabilityStatus
    fraud_monitoring: CapabilityStatus
    psychology_observation: CapabilityStatus
    radar_worker: CapabilityStatus
    radar_participation: CapabilityStatus
    fusion: CapabilityStatus
    camera_preview_managed_by_guard: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    updated_at: datetime
