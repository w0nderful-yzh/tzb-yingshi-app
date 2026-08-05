import time
from typing import Any, Literal, Protocol, cast

import httpx

TOKEN_URL = "https://open.ys7.com/api/lapp/token/get"
LIVE_ADDRESS_URL = "https://open.ys7.com/api/lapp/v2/live/address/get"
PROTOCOL_CODES = {"hls": 2, "rtmp": 3, "flv": 4}


class Ys7ApiError(RuntimeError):
    pass


class Ys7LiveAddressProvider(Protocol):
    async def get_live_address(
        self,
        *,
        device_serial: str,
        channel_no: int,
        protocol: Literal["hls", "rtmp", "flv"],
        quality: int,
    ) -> str: ...


class Ys7SdkCredentialProvider(Protocol):
    async def get_access_token(self) -> str: ...


class Ys7ApiClient:
    def __init__(
        self,
        *,
        app_key: str | None,
        app_secret: str | None,
        access_token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._static_access_token = access_token
        self._cached_access_token: str | None = None
        self._access_token_expires_at_ms = 0
        self._transport = transport

    async def get_live_address(
        self,
        *,
        device_serial: str,
        channel_no: int,
        protocol: Literal["hls", "rtmp", "flv"],
        quality: int,
    ) -> str:
        token = await self.get_access_token()
        response = await self._post(
            LIVE_ADDRESS_URL,
            {
                "accessToken": token,
                "deviceSerial": device_serial,
                "channelNo": channel_no,
                "protocol": PROTOCOL_CODES[protocol],
                "quality": quality,
            },
        )
        if str(response.get("code")) == "10002" and self._static_access_token is None:
            self._cached_access_token = None
            token = await self.get_access_token()
            response = await self._post(
                LIVE_ADDRESS_URL,
                {
                    "accessToken": token,
                    "deviceSerial": device_serial,
                    "channelNo": channel_no,
                    "protocol": PROTOCOL_CODES[protocol],
                    "quality": quality,
                },
            )
        self._require_success(response, operation="get live address")
        data = response.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("url"), str):
            raise Ys7ApiError("YS7 live address response did not contain data.url")
        return cast(str, data["url"])

    async def get_access_token(self) -> str:
        if self._static_access_token:
            return self._static_access_token
        now_ms = round(time.time() * 1000)
        if self._cached_access_token and self._access_token_expires_at_ms > now_ms + 60_000:
            return self._cached_access_token
        if not self._app_key or not self._app_secret:
            raise Ys7ApiError("YS7 AppKey/AppSecret or access token is not configured")
        response = await self._post(
            TOKEN_URL,
            {"appKey": self._app_key, "appSecret": self._app_secret},
        )
        self._require_success(response, operation="get access token")
        data = response.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("accessToken"), str):
            raise Ys7ApiError("YS7 token response did not contain an access token")
        self._cached_access_token = cast(str, data["accessToken"])
        expire_time = data.get("expireTime")
        self._access_token_expires_at_ms = (
            int(expire_time) if expire_time is not None else now_ms + 3_600_000
        )
        return self._cached_access_token

    async def _post(self, url: str, data: dict[str, object]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=10.0,
            ) as client:
                response = await client.post(url, data=data)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise Ys7ApiError("YS7 API request failed") from exc
        if not isinstance(payload, dict):
            raise Ys7ApiError("YS7 API returned an invalid response")
        return cast(dict[str, Any], payload)

    @staticmethod
    def _require_success(payload: dict[str, Any], *, operation: str) -> None:
        if str(payload.get("code")) != "200":
            code = str(payload.get("code", "unknown"))
            raise Ys7ApiError(f"YS7 {operation} failed with code {code}")
