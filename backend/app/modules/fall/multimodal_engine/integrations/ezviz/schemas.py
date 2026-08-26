from pydantic import BaseModel, ConfigDict, Field


class EzvizTokenData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    access_token: str = Field(alias="accessToken", repr=False)
    expire_time: int = Field(alias="expireTime", gt=0)


class EzvizTokenResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | int
    msg: str = ""
    data: EzvizTokenData | None = None


class EzvizDevice(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    device_serial: str = Field(alias="deviceSerial")
    device_name: str | None = Field(default=None, alias="deviceName")
    device_type: str | None = Field(default=None, alias="deviceType")
    status: int | None = None
    defence: int | None = None
    device_version: str | None = Field(default=None, alias="deviceVersion")
    add_time: int | None = Field(default=None, alias="addTime")
    update_time: int | None = Field(default=None, alias="updateTime")


class EzvizPageInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total: int = 0
    page: int = 0
    size: int = 0


class EzvizDeviceListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | int
    msg: str = ""
    data: list[EzvizDevice] | None = None
    page: EzvizPageInfo | None = None


class EzvizDeviceListResult(BaseModel):
    devices: list[EzvizDevice]
    total: int
    page_start: int
    page_size: int


class EzvizPlayAddressData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    address_id: str | None = Field(default=None, alias="id")
    url: str = Field(min_length=1)
    expire_time: int | str | None = Field(default=None, alias="expireTime")


class EzvizPlayAddressResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | int
    msg: str = ""
    data: EzvizPlayAddressData | None = None


class EzvizPlayConfigResult(BaseModel):
    device_id: str
    channel_no: int
    play_url: str
    access_token: str = Field(repr=False)
    expires_at: int | str | None = None
