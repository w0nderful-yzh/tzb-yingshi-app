from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.modules.fall.multimodal_engine.schemas.risk_event import EvidenceItem, RiskLevel, RiskModule


class AdapterContext(BaseModel):
    """运行一次算法适配器所需的会话上下文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)


class AlgorithmFinding(BaseModel):
    """成员算法原始输出经Adapter归一化后的内部结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    module: RiskModule
    event_type: str = Field(min_length=1, max_length=64)
    occurred_at: datetime
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    summary: str = Field(min_length=1, max_length=500)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommended_action: str | None = Field(default=None, max_length=500)
    snapshot_path: str | None = Field(default=None, max_length=500)
    clip_path: str | None = Field(default=None, max_length=500)
    model_version: str = Field(min_length=1, max_length=64)

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone offset")
        return value
