"""HTTP client for the read-only psychology latest-assessment projection."""

from typing import Any

import httpx
from pydantic import ValidationError

from app.modules.psychology.ports import PsychologySourceError
from app.modules.psychology.source_schemas import PsychologySourceSnapshot


class HttpPsychologySource:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        api_key: str | None = None,
        latest_path: str = "/api/psychology/assessments/latest",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_seconds,
            headers=headers,
            transport=transport,
        )
        self._latest_path = latest_path

    async def get_latest_assessment(self, *, subject_key: str) -> PsychologySourceSnapshot:
        try:
            response = await self._client.get(
                self._latest_path.lstrip("/"),
                params={"subject_key": subject_key},
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PsychologySourceError("psychology upstream request failed") from exc
        if not isinstance(payload, dict):
            raise PsychologySourceError("psychology upstream returned a non-object payload")
        try:
            snapshot = PsychologySourceSnapshot.model_validate(payload)
        except ValidationError as exc:
            raise PsychologySourceError("psychology upstream payload is invalid") from exc
        if snapshot.subject_key != subject_key:
            raise PsychologySourceError("psychology upstream returned the wrong subject")
        return snapshot

    async def close(self) -> None:
        await self._client.aclose()

