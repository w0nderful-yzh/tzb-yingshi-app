from datetime import datetime
from typing import Any

import httpx

from app.modules.fall.multimodal_engine.schemas.risk_event import RiskEventInput, RiskEventResponse


class EventPublishError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class EventPublisher:
    """通过既有HTTP入口发布算法事件，不直接访问Repository或MySQL。"""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    def publish(self, event: RiskEventInput) -> RiskEventResponse:
        try:
            response = self._client.post(
                "/api/algorithm/events",
                json=event.model_dump(mode="json"),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise EventPublishError(
                f"risk event was rejected: {detail}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise EventPublishError(f"risk event publish failed: {exc}") from exc

        try:
            return RiskEventResponse.model_validate(
                self._parse_response_datetimes(response.json())
            )
        except (ValueError, TypeError) as exc:
            raise EventPublishError("risk event API returned an invalid response") from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "EventPublisher":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return f"HTTP {response.status_code}"
        if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
            return payload["detail"]
        return f"HTTP {response.status_code}"

    @staticmethod
    def _parse_response_datetimes(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        parsed = dict(payload)
        for field in (
            "occurred_at",
            "received_at",
            "handled_at",
            "updated_at",
        ):
            value = parsed.get(field)
            if isinstance(value, str):
                parsed[field] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed
