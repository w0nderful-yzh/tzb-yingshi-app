from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Ys7SignalType(StrEnum):
    PHONE_CALL = "phone_call"
    PEOPLE_COUNT = "people_count"
    PERSON_DETECTED = "person_detected"


class Ys7Box(BaseModel):
    model_config = ConfigDict(extra="allow")

    xyxy: tuple[float, float, float, float] | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "Ys7Box":
        x1, y1, x2, y2 = self.coordinates
        if x2 < x1 or y2 < y1:
            raise ValueError("box maximums must not be less than minimums")
        return self

    @property
    def coordinates(self) -> tuple[float, float, float, float]:
        if self.xyxy is not None:
            return self.xyxy
        if self.x1 is None or self.y1 is None or self.x2 is None or self.y2 is None:
            raise ValueError("box must provide xyxy or x1/y1/x2/y2")
        return self.x1, self.y1, self.x2, self.y2


class Ys7Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str | None = None
    request_id: str | None = None
    source_event_id: str
    device_id: str
    occurred_at: datetime
    received_at: datetime
    event_type: Ys7SignalType
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    people_count: int | None = Field(default=None, ge=0)
    boxes: list[Ys7Box] = Field(default_factory=list)
    image_url: str | None = None
