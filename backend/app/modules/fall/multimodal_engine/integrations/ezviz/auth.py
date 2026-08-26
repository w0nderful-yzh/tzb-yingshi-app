from collections.abc import Callable
from threading import Lock
from time import time

import httpx
from pydantic import ValidationError

from app.modules.fall.multimodal_engine.core.config import Settings, get_settings
from app.modules.fall.multimodal_engine.integrations.ezviz.schemas import EzvizTokenData, EzvizTokenResponse


class EzvizIntegrationError(RuntimeError):
    pass


class EzvizConfigurationError(EzvizIntegrationError):
    pass


class EzvizApiError(EzvizIntegrationError):
    def __init__(self, message: str, *, code: str | int | None = None) -> None:
        super().__init__(message)
        self.code = code


class EzvizAuth:
    """获取并在当前后端进程内缓存萤石AccessToken。"""

    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        base_url: str,
        request_timeout_seconds: float = 10.0,
        refresh_skew_seconds: int = 60,
        http_client: httpx.Client | None = None,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._app_key = app_key.strip()
        self._app_secret = app_secret.strip()
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client or httpx.Client(
            timeout=request_timeout_seconds,
            # 萤石设备链路必须在代理/VPN开关变化后仍可用。系统代理只
            # 服务浏览器访问，不应把open.ys7.com转发到本机失效代理端口。
            trust_env=False,
        )
        self._owns_http_client = http_client is None
        self._refresh_skew_ms = refresh_skew_seconds * 1000
        self._now_ms = now_ms or (lambda: int(time() * 1000))
        self._cached_token: EzvizTokenData | None = None
        self._cache_lock = Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> "EzvizAuth":
        settings = settings or get_settings()
        return cls(
            app_key=settings.ezviz_app_key.get_secret_value(),
            app_secret=settings.ezviz_app_secret.get_secret_value(),
            base_url=settings.ezviz_base_url,
            request_timeout_seconds=settings.ezviz_request_timeout_seconds,
            refresh_skew_seconds=settings.ezviz_token_refresh_skew_seconds,
            http_client=http_client,
        )

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        with self._cache_lock:
            if not force_refresh and self._token_is_usable(self._cached_token):
                return self._cached_token.access_token

            self._validate_configuration()
            self._cached_token = self._request_access_token()
            return self._cached_token.access_token

    def clear_cached_token(self) -> None:
        with self._cache_lock:
            self._cached_token = None

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def _token_is_usable(self, token: EzvizTokenData | None) -> bool:
        if token is None:
            return False
        return self._now_ms() + self._refresh_skew_ms < token.expire_time

    def _validate_configuration(self) -> None:
        if not self._app_key or not self._app_secret:
            raise EzvizConfigurationError(
                "EZVIZ_APP_KEY and EZVIZ_APP_SECRET must be configured locally"
            )

    def _request_access_token(self) -> EzvizTokenData:
        try:
            response = self._http_client.post(
                f"{self._base_url}/token/get",
                data={
                    "appKey": self._app_key,
                    "appSecret": self._app_secret,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EzvizApiError("Failed to request an EZVIZ access token") from exc

        try:
            payload = EzvizTokenResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise EzvizApiError("EZVIZ returned an invalid token response") from exc

        if str(payload.code) != "200" or payload.data is None:
            raise EzvizApiError(
                payload.msg or "EZVIZ rejected the token request",
                code=payload.code,
            )
        return payload.data
