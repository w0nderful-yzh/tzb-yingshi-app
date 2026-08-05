from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserInfo(BaseModel):
    user_id: str
    role: Literal["elder", "family"]
    name: str
    bound_family_count: int = 0
    font_size: str = "extra_large"
    voice_assist_enabled: bool = True


class TodayStats(BaseModel):
    event_count: int = 0
    active_hours: float = 0.0
    call_screened: int = 0


class SafetyStatus(BaseModel):
    overall: Literal["safe", "attention", "danger"]
    overall_label: str
    active_event_count: int
    highest_active_level: str | None
    devices_online: int
    devices_total: int
    checked_at: datetime
    today: TodayStats


class DeviceItem(BaseModel):
    device_id: str
    name: str
    room: str = ""
    online: bool
    signal: Literal["good", "weak", "offline"]
    last_seen_at: datetime | None = None


class DeviceListData(BaseModel):
    devices: list[DeviceItem]


class LiveUrlData(BaseModel):
    url: str
    protocol: str
    expires_in: int


class LiveSdkSessionData(BaseModel):
    app_key: str
    access_token: str
    device_serial: str
    channel_no: int
    expires_in: int


class RiskEventItem(BaseModel):
    event_id: str
    type: str
    level: str
    title: str
    summary: str
    device_id: str = ""
    occurred_at: datetime
    status: str
    version: int
    evidence_image_url: str | None = None


class EventListData(BaseModel):
    events: list[RiskEventItem]
    next_cursor: str | None = None


class ReasonItem(BaseModel):
    key: str
    label: str
    value: str


class AnalysisData(BaseModel):
    confidence: float = 0.0
    reasons: list[ReasonItem]
    disclaimer: str


class NotificationItem(BaseModel):
    target: str
    channel: str
    sent_at: datetime | None = None
    ack: bool = False


class EscalationData(BaseModel):
    auto_call_at: datetime | None = None
    status: str = "pending"


class EventDetailData(BaseModel):
    event_id: str
    type: str
    level: str
    status: str
    version: int
    device_id: str = ""
    occurred_at: datetime
    evidence_image_url: str | None = None
    analysis: AnalysisData
    notifications: list[NotificationItem]
    escalation: EscalationData


class SosRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trigger: str = Field(default="long_press", max_length=32)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value


class SosResult(BaseModel):
    event_id: str
    status: Literal["recorded", "duplicate"]
    notified_contacts: int


class ConfirmRequest(BaseModel):
    action: Literal["im_ok", "need_help"]
    version: int | None = Field(default=None, ge=1)


class StatusPatchRequest(BaseModel):
    status: Literal["acknowledged", "resolved", "false_alarm"]
    note: str = Field(default="", max_length=1_000)
    version: int | None = Field(default=None, ge=1)


class EmptyData(BaseModel):
    pass


class ContactItem(BaseModel):
    order: int
    name: str
    relation: str
    phone: str
    channels: list[str]


class ContactsData(BaseModel):
    contacts: list[ContactItem]


class ElderItem(BaseModel):
    elder_id: str
    name: str
    relation: str
    overall: Literal["safe", "attention", "danger"]
    last_active_at: datetime | None = None
    pending_event_count: int


class EldersData(BaseModel):
    elders: list[ElderItem]


class StatsBucket(BaseModel):
    period: str
    reminder: int = 0
    warning: int = 0
    emergency: int = 0


class EventsStatsData(BaseModel):
    buckets: list[StatsBucket]


class ActivityData(BaseModel):
    hours: list[float]
