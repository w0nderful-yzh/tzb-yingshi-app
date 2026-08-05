import json
from datetime import UTC, datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.infrastructure.external.ys7.models import Ys7Box, Ys7Signal, Ys7SignalType


class _Ys7Payload(BaseModel):
    model_config = ConfigDict(extra="allow")

    message_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("messageId", "message_id"),
    )
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("requestId", "request_id"),
    )
    event_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("eventId", "event_id"),
    )
    device_id: str = Field(
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("deviceId", "device_id", "deviceSerial", "sn"),
    )
    occurred_at: datetime = Field(
        validation_alias=AliasChoices("timestamp", "occurredAt", "occurred_at", "eventTime")
    )
    event_type: Ys7SignalType = Field(validation_alias=AliasChoices("eventType", "event_type"))
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    people_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("peopleCount", "people_count"),
    )
    boxes: list[Ys7Box] = Field(default_factory=list)
    image_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("imageUrl", "image_url", "picUrl"),
    )

    @field_validator("occurred_at", mode="before")
    @classmethod
    def parse_event_time(cls, value: object) -> object:
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, (int, float)):
            seconds = float(value) / 1000 if value > 10_000_000_000 else float(value)
            return datetime.fromtimestamp(seconds, tz=UTC)
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamp must include a timezone")
        return value

    @field_validator("event_type", mode="before")
    @classmethod
    def normalize_event_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_")
        return value


class Ys7EventParser:
    def parse(
        self,
        raw_payload: dict[str, object],
        *,
        received_at: datetime | None = None,
    ) -> Ys7Signal:
        payload = self._unwrap_payload(raw_payload)
        parsed = _Ys7Payload.model_validate(payload)
        event_id = parsed.event_id or parsed.message_id or parsed.request_id
        if event_id is None:
            raise ValueError("eventId, messageId, or requestId is required")
        return Ys7Signal(
            message_id=parsed.message_id,
            request_id=parsed.request_id,
            source_event_id=event_id,
            device_id=parsed.device_id,
            occurred_at=parsed.occurred_at,
            received_at=received_at or datetime.now(UTC),
            event_type=parsed.event_type,
            confidence=parsed.confidence,
            people_count=parsed.people_count,
            boxes=parsed.boxes,
            image_url=parsed.image_url,
        )

    def _unwrap_payload(self, raw_payload: dict[str, object]) -> dict[str, object]:
        body = raw_payload.get("body")
        if isinstance(body, dict):
            return {**raw_payload, **body}
        if isinstance(body, str):
            try:
                decoded = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValueError("YS7 body must contain valid JSON") from exc
            if not isinstance(decoded, dict):
                raise ValueError("YS7 body JSON must be an object")
            return {**raw_payload, **decoded}
        return raw_payload
