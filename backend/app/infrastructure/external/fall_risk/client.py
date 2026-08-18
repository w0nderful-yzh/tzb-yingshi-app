"""HTTP adapter for the existing multimodal prototype and per-room Radar services."""

from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.modules.fall.ports import FallRiskSourceError
from app.modules.fall.source_schemas import (
    CameraLedSourceSnapshot,
    RadarLatestResponse,
    RadarOnlySourceSnapshot,
)


class HttpFallRiskSource:
    def __init__(
        self,
        *,
        camera_base_url: str | None,
        radar_room_base_urls: Mapping[str, str],
        timeout_seconds: float,
        api_key: str | None = None,
        camera_led_path: str = "/api/multimodal/camera-led-associated/latest",
        radar_only_path: str = "/api/radar/latest",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client_options: dict[str, Any] = {
            "timeout": timeout_seconds,
            "headers": headers,
            "transport": transport,
        }
        self._camera_client = (
            httpx.AsyncClient(
                base_url=camera_base_url.rstrip("/") + "/",
                **client_options,
            )
            if camera_base_url
            else None
        )
        self._radar_clients = {
            room_id: httpx.AsyncClient(
                base_url=base_url.rstrip("/") + "/",
                **client_options,
            )
            for room_id, base_url in radar_room_base_urls.items()
        }
        self._camera_led_path = camera_led_path
        self._radar_only_path = radar_only_path

    async def get_camera_led_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> CameraLedSourceSnapshot:
        del elder_id  # The real prototype endpoint has no elder query parameter.
        if self._camera_client is None:
            raise FallRiskSourceError("camera-led fall-risk upstream is not configured")
        payload = await self._get(
            self._camera_client,
            self._camera_led_path,
        )
        snapshot = self._validate(CameraLedSourceSnapshot, payload)
        if snapshot.radar.room is not None:
            self._require_room(snapshot.radar.room, room_id)
        return snapshot

    async def get_radar_only_risk(
        self,
        *,
        elder_id: str,
        room_id: str,
    ) -> RadarOnlySourceSnapshot:
        del elder_id  # Room selection is performed by choosing the Radar service base URL.
        client = self._radar_clients.get(room_id)
        if client is None:
            raise FallRiskSourceError("radar-only fall-risk upstream is not configured for room")
        payload = await self._get(client, self._radar_only_path)
        envelope = self._validate(RadarLatestResponse, payload)
        snapshot = (
            envelope.calibrated_tcn_prediction or envelope.tcn_prediction or envelope.tcn_baseline
        )
        if snapshot is None:
            raise FallRiskSourceError("radar upstream did not return a Radar TCN result")
        self._require_room(snapshot.room, room_id)
        return snapshot

    async def close(self) -> None:
        clients = [*self._radar_clients.values()]
        if self._camera_client is not None:
            clients.append(self._camera_client)
        for client in clients:
            await client.aclose()

    @staticmethod
    async def _get(
        client: httpx.AsyncClient,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        try:
            response = await client.get(path.lstrip("/"), params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise FallRiskSourceError("fall-risk upstream request failed") from exc
        if not isinstance(payload, dict):
            raise FallRiskSourceError("fall-risk upstream returned a non-object payload")
        return payload

    @staticmethod
    def _validate[ModelT: BaseModel](
        model: type[ModelT],
        payload: Mapping[str, Any],
    ) -> ModelT:
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise FallRiskSourceError("fall-risk upstream payload is invalid") from exc

    @staticmethod
    def _require_room(actual: str, expected: str) -> None:
        if actual != expected:
            raise FallRiskSourceError("fall-risk upstream returned the wrong room")
