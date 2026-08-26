from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UnifiedDataPacket(BaseModel):
    """不同数据源进入算法适配层前使用的最小内部数据契约。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str = Field(min_length=1, max_length=80)
    session_id: str = Field(min_length=1, max_length=64)
    source_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    modality: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )
    timestamp: datetime
    data: dict[str, Any]

    @field_validator("packet_id", "session_id", "source_id", "device_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier must not be blank")
        return normalized

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value
