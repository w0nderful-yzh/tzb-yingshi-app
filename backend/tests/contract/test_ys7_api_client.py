from urllib.parse import parse_qs

import httpx
import pytest

from app.infrastructure.external.ys7.api_client import Ys7ApiClient, Ys7ApiError


@pytest.mark.asyncio
async def test_app_credentials_are_exchanged_and_cached_for_live_address() -> None:
    token_requests = 0
    live_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_requests, live_requests
        form = parse_qs(request.content.decode())
        if request.url.path.endswith("/token/get"):
            token_requests += 1
            assert form == {"appKey": ["app-key"], "appSecret": ["app-secret"]}
            return httpx.Response(
                200,
                json={
                    "code": "200",
                    "msg": "success",
                    "data": {
                        "accessToken": "access-token",
                        "expireTime": 4_102_444_800_000,
                    },
                },
            )
        live_requests += 1
        assert form["accessToken"] == ["access-token"]
        assert form["deviceSerial"] == ["camera-01"]
        assert form["channelNo"] == ["1"]
        assert form["protocol"] == ["4"]
        assert form["quality"] == ["2"]
        return httpx.Response(
            200,
            json={
                "code": "200",
                "msg": "success",
                "data": {"url": "https://stream.invalid/live.flv"},
            },
        )

    client = Ys7ApiClient(
        app_key="app-key",
        app_secret="app-secret",
        transport=httpx.MockTransport(handler),
    )

    first = await client.get_live_address(
        device_serial="camera-01",
        channel_no=1,
        protocol="flv",
        quality=2,
    )
    second = await client.get_live_address(
        device_serial="camera-01",
        channel_no=1,
        protocol="flv",
        quality=2,
    )

    assert first == second == "https://stream.invalid/live.flv"
    assert token_requests == 1
    assert live_requests == 2


@pytest.mark.asyncio
async def test_access_token_can_be_reused_for_android_sdk_session() -> None:
    token_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_requests
        token_requests += 1
        assert request.url.path.endswith("/token/get")
        return httpx.Response(
            200,
            json={
                "code": "200",
                "data": {
                    "accessToken": "sdk-access-token",
                    "expireTime": 4_102_444_800_000,
                },
            },
        )

    client = Ys7ApiClient(
        app_key="app-key",
        app_secret="app-secret",
        transport=httpx.MockTransport(handler),
    )

    assert await client.get_access_token() == "sdk-access-token"
    assert await client.get_access_token() == "sdk-access-token"
    assert token_requests == 1


@pytest.mark.asyncio
async def test_static_access_token_skips_token_exchange() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/live/address/get")
        assert parse_qs(request.content.decode())["accessToken"] == ["static-token"]
        return httpx.Response(
            200,
            json={"code": "200", "data": {"url": "rtmp://stream.invalid/live"}},
        )

    client = Ys7ApiClient(
        app_key=None,
        app_secret=None,
        access_token="static-token",
        transport=httpx.MockTransport(handler),
    )

    address = await client.get_live_address(
        device_serial="camera-01",
        channel_no=1,
        protocol="rtmp",
        quality=1,
    )

    assert address == "rtmp://stream.invalid/live"


@pytest.mark.asyncio
async def test_device_alarm_list_uses_expected_query_window() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/alarm/device/list")
        form = parse_qs(request.content.decode())
        assert form == {
            "accessToken": ["static-token"],
            "deviceSerial": ["camera-01"],
            "startTime": ["1700000000000"],
            "endTime": ["1700000120000"],
            "status": ["2"],
            "pageStart": ["0"],
            "pageSize": ["50"],
        }
        return httpx.Response(
            200,
            json={
                "code": "200",
                "data": [
                    {
                        "alarmId": "alarm-01",
                        "alarmType": 10000,
                        "alarmStart": "2026-08-07 12:00:00",
                    }
                ],
            },
        )

    client = Ys7ApiClient(
        app_key=None,
        app_secret=None,
        access_token="static-token",
        transport=httpx.MockTransport(handler),
    )

    alarms = await client.list_device_alarms(
        device_serial="camera-01",
        start_time_ms=1_700_000_000_000,
        end_time_ms=1_700_000_120_000,
        page_size=50,
    )

    assert alarms[0]["alarmId"] == "alarm-01"


@pytest.mark.asyncio
async def test_ys7_error_exposes_only_error_code() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "20014", "msg": "device offline"})

    client = Ys7ApiClient(
        app_key=None,
        app_secret=None,
        access_token="secret-token",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(Ys7ApiError, match="code 20014") as raised:
        await client.get_live_address(
            device_serial="camera-01",
            channel_no=1,
            protocol="flv",
            quality=2,
        )

    assert "secret-token" not in str(raised.value)
