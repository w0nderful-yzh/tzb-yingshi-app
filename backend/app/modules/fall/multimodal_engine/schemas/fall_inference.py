from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FallInferenceJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FallInferenceSystemStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    model_version: str
    input_format: str
    execution_mode: str
    project_dir_exists: bool
    python_exists: bool
    checkpoints_found: int
    ezviz_live_capture_ready: bool = False
    note: str


class FallInferenceJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: FallInferenceJobStatus
    session_id: str
    device_id: str
    filename: str
    record_non_alert_test_event: bool
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    event_id: str | None = None
    error: str | None = Field(default=None, max_length=2000)
