from urllib.parse import quote, urlsplit

import httpx
from pydantic import ValidationError

from app.modules.fall.multimodal_engine.core.config import Settings, get_settings
from app.modules.fall.multimodal_engine.integrations.ezviz.auth import EzvizApiError, EzvizAuth
from app.modules.fall.multimodal_engine.integrations.ezviz.schemas import (
    EzvizDeviceListResponse,
    EzvizDeviceListResult,
    EzvizPlayAddressResponse,
    EzvizPlayConfigResult,
)


class EzvizClient:
    """萤石开放平台服务器API的最小同步客户端。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
        auth: EzvizAuth | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_url = self._settings.ezviz_base_url.rstrip("/")
        self._device_verify_code = (
            self._settings.ezviz_device_verify_code.get_secret_value().strip()
        )
        self._http_client = http_client or httpx.Client(
            timeout=self._settings.ezviz_request_timeout_seconds,
            # 不继承HTTP_PROXY/HTTPS_PROXY。比赛电脑的代理软件关闭后常
            # 留下失效环境变量，会让萤石API误报502而本机服务仍正常。
            trust_env=False,
        )
        self._owns_http_client = http_client is None
        self.auth = auth or EzvizAuth.from_settings(
            self._settings,
            http_client=self._http_client,
        )

    def list_devices(
        self,
        *,
        page_start: int = 0,
        page_size: int = 10,
    ) -> EzvizDeviceListResult:
        if page_start < 0:
            raise ValueError("page_start must be greater than or equal to 0")
        if not 1 <= page_size <= 50:
            raise ValueError("page_size must be between 1 and 50")

        access_token = self.auth.get_access_token()
        try:
            response = self._http_client.post(
                f"{self._base_url}/device/list",
                data={
                    "accessToken": access_token,
                    "pageStart": page_start,
                    "pageSize": page_size,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EzvizApiError("Failed to request the EZVIZ device list") from exc

        try:
            payload = EzvizDeviceListResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise EzvizApiError("EZVIZ returned an invalid device list response") from exc

        if str(payload.code) != "200":
            raise EzvizApiError(
                payload.msg or "EZVIZ rejected the device list request",
                code=payload.code,
            )

        devices = payload.data or []
        page = payload.page
        return EzvizDeviceListResult(
            devices=devices,
            total=page.total if page is not None else len(devices),
            page_start=page.page if page is not None else page_start,
            page_size=page.size if page is not None else page_size,
        )

    def get_play_config(
        self,
        device_id: str,
        *,
        channel_no: int = 1,
    ) -> EzvizPlayConfigResult:
        device_id = device_id.strip()
        if not device_id:
            raise ValueError("device_id must not be empty")
        if channel_no < 1:
            raise ValueError("channel_no must be greater than or equal to 1")

        access_token = self.auth.get_access_token()
        request_data: dict[str, str | int] = {
            "accessToken": access_token,
            "deviceSerial": device_id,
            "channelNo": channel_no,
            "protocol": 1,
            # The browser player uses software/WASM decoding.  Request the
            # fluent stream instead of the 4K-capable HD stream so long-running
            # duty-station playback does not stall under decoder pressure.
            "quality": 2,
        }
        if self._device_verify_code:
            request_data["code"] = self._device_verify_code

        try:
            response = self._http_client.post(
                f"{self._base_url}/v2/live/address/get",
                data=request_data,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EzvizApiError("Failed to request the EZVIZ play address") from exc

        try:
            payload = EzvizPlayAddressResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise EzvizApiError("EZVIZ returned an invalid play response") from exc

        if str(payload.code) == "60019" and self._device_verify_code:
            encoded_verify_code = quote(self._device_verify_code, safe="")
            encoded_device_id = quote(device_id, safe="")
            return EzvizPlayConfigResult(
                device_id=device_id,
                channel_no=channel_no,
                play_url=(
                    f"ezopen://{encoded_verify_code}@open.ys7.com/"
                    f"{encoded_device_id}/{channel_no}.live"
                ),
                access_token=access_token,
                expires_at=None,
            )

        if str(payload.code) != "200" or payload.data is None:
            raise EzvizApiError(
                payload.msg or "EZVIZ rejected the play address request",
                code=payload.code,
            )

        return EzvizPlayConfigResult(
            device_id=device_id,
            channel_no=channel_no,
            play_url=payload.data.url,
            access_token=access_token,
            expires_at=payload.data.expire_time,
        )

    def get_standard_live_address(
        self,
        device_id: str,
        *,
        channel_no: int = 1,
        protocol: int = 3,
        quality: int = 2,
        expire_seconds: int = 120,
    ) -> EzvizPlayConfigResult:
        """Request a server-decodable HLS/RTMP/HTTP-FLV live address.

        Browser playback keeps using EZOPEN through ``get_play_config``.  This
        method is intentionally separate because EZOPEN is a private player
        protocol and cannot be passed to OpenCV/FFmpeg as a normal URL.
        """

        device_id = device_id.strip()
        if not device_id:
            raise ValueError("device_id must not be empty")
        if channel_no < 1:
            raise ValueError("channel_no must be greater than or equal to 1")
        if protocol not in {2, 3, 4}:
            raise ValueError("standard live protocol must be 2 (HLS), 3 (RTMP), or 4 (FLV)")
        if quality not in {1, 2}:
            raise ValueError("quality must be 1 (HD) or 2 (fluent)")
        if not 30 <= expire_seconds <= 604800:
            raise ValueError("expire_seconds must be between 30 and 604800")

        access_token = self.auth.get_access_token()
        request_data: dict[str, str | int] = {
            "accessToken": access_token,
            "deviceSerial": device_id,
            "channelNo": channel_no,
            "protocol": protocol,
            "quality": quality,
            "type": 1,
            "expireTime": expire_seconds,
        }
        if self._device_verify_code:
            request_data["code"] = self._device_verify_code

        try:
            response = self._http_client.post(
                f"{self._base_url}/v2/live/address/get",
                data=request_data,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EzvizApiError("Failed to request an EZVIZ standard live address") from exc

        try:
            payload = EzvizPlayAddressResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise EzvizApiError("EZVIZ returned an invalid standard stream response") from exc

        if str(payload.code) != "200" or payload.data is None:
            raise EzvizApiError(
                payload.msg or "EZVIZ rejected the standard live address request",
                code=payload.code,
            )

        scheme = urlsplit(payload.data.url).scheme.lower()
        allowed_schemes = {
            2: {"http", "https"},
            3: {"rtmp", "rtmps"},
            4: {"http", "https"},
        }
        if scheme not in allowed_schemes[protocol]:
            raise EzvizApiError(
                f"EZVIZ returned an unsupported {scheme or 'unknown'} stream URL",
                code="UNSUPPORTED_STREAM_SCHEME",
            )

        return EzvizPlayConfigResult(
            device_id=device_id,
            channel_no=channel_no,
            play_url=payload.data.url,
            access_token=access_token,
            expires_at=payload.data.expire_time,
        )

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> "EzvizClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
