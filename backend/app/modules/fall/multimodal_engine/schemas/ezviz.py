from typing import Literal

from pydantic import BaseModel, Field


class EzvizDeviceResponse(BaseModel):
    device_name: str
    device_id: str
    online: bool
    source: Literal["EZVIZ"] = "EZVIZ"


class EzvizDeviceListApiResponse(BaseModel):
    devices: list[EzvizDeviceResponse]
    total: int
    page_start: int
    page_size: int


class EzvizPlayConfigResponse(BaseModel):
    device_id: str
    channel_no: int
    protocol: Literal["EZOPEN"] = "EZOPEN"
    play_url: str
    access_token: str = Field(repr=False)
    expires_at: int | None = None
    source: Literal["EZVIZ"] = "EZVIZ"
