import unittest
from urllib.parse import parse_qs

import httpx
from fastapi.testclient import TestClient

from app.modules.fall.multimodal_engine.api.ezviz import get_ezviz_client
from app.modules.fall.multimodal_engine.core.config import Settings
from app.modules.fall.multimodal_engine.integrations.ezviz import EzvizAuth, EzvizClient, EzvizConfigurationError
from app.modules.fall.multimodal_engine.integrations.ezviz.schemas import EzvizPlayConfigResult
from app.modules.fall.multimodal_engine.main import app


BASE_URL = "https://open.ys7.com/api/lapp"
NOW_MS = 1_700_000_000_000


class FakePlayClient:
    def get_play_config(
        self,
        device_id: str,
        *,
        channel_no: int = 1,
    ) -> EzvizPlayConfigResult:
        return EzvizPlayConfigResult(
            device_id=device_id,
            channel_no=channel_no,
            play_url=f"ezopen://open.ys7.com/{device_id}/{channel_no}.live",
            access_token="mock-access-token",
            expires_at=NOW_MS + 3_600_000,
        )


class UnconfiguredPlayClient:
    def get_play_config(self, *_: object, **__: object) -> EzvizPlayConfigResult:
        raise EzvizConfigurationError("missing credentials")


class EzvizPhase6C1Test(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_integration_client_requests_ezopen_play_address(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            form = parse_qs(request.content.decode("utf-8"))
            requests.append(request.url.path)
            if request.url.path.endswith("/token/get"):
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "msg": "OK",
                        "data": {
                            "accessToken": "mock-access-token",
                            "expireTime": NOW_MS + 7_200_000,
                        },
                    },
                )
            if request.url.path.endswith("/v2/live/address/get"):
                self.assertEqual(form["accessToken"], ["mock-access-token"])
                self.assertEqual(form["deviceSerial"], ["TEST123"])
                self.assertEqual(form["channelNo"], ["1"])
                self.assertEqual(form["protocol"], ["1"])
                self.assertEqual(form["quality"], ["2"])
                self.assertNotIn("code", form)
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "msg": "OK",
                        "data": {
                            "id": "address-id",
                            "url": "ezopen://open.ys7.com/TEST123/1.live",
                            "expireTime": NOW_MS + 3_600_000,
                        },
                    },
                )
            return httpx.Response(404)

        settings = Settings(
            _env_file=None,
            ezviz_base_url=BASE_URL,
            ezviz_app_key="test-app-key",
            ezviz_app_secret="test-app-secret",
        )
        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            auth = EzvizAuth(
                app_key="test-app-key",
                app_secret="test-app-secret",
                base_url=BASE_URL,
                http_client=http_client,
                now_ms=lambda: NOW_MS,
            )
            client = EzvizClient(
                settings,
                http_client=http_client,
                auth=auth,
            )
            result = client.get_play_config("TEST123")

        self.assertEqual(
            requests,
            ["/api/lapp/token/get", "/api/lapp/v2/live/address/get"],
        )
        self.assertEqual(result.device_id, "TEST123")
        self.assertEqual(result.channel_no, 1)
        self.assertEqual(result.access_token, "mock-access-token")
        self.assertTrue(result.play_url.startswith("ezopen://"))

    def test_backend_returns_normalized_play_config(self) -> None:
        app.dependency_overrides[get_ezviz_client] = FakePlayClient

        with TestClient(app) as client:
            response = client.get(
                "/api/ezviz/devices/TEST123/play-config",
                params={"channel_no": 1},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["device_id"], "TEST123")
        self.assertEqual(payload["channel_no"], 1)
        self.assertEqual(payload["protocol"], "EZOPEN")
        self.assertEqual(payload["source"], "EZVIZ")
        self.assertEqual(payload["access_token"], "mock-access-token")
        self.assertTrue(payload["play_url"].startswith("ezopen://"))

    def test_encrypted_device_uses_locally_configured_verify_code(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            form = parse_qs(request.content.decode("utf-8"))
            if request.url.path.endswith("/token/get"):
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "msg": "OK",
                        "data": {
                            "accessToken": "mock-access-token",
                            "expireTime": NOW_MS + 7_200_000,
                        },
                    },
                )
            if request.url.path.endswith("/v2/live/address/get"):
                self.assertEqual(form["code"], ["ABC123"])
                self.assertEqual(form["quality"], ["2"])
                return httpx.Response(
                    200,
                    json={"code": "60019", "msg": "加密已开启", "data": None},
                )
            return httpx.Response(404)

        settings = Settings(
            _env_file=None,
            ezviz_base_url=BASE_URL,
            ezviz_app_key="test-app-key",
            ezviz_app_secret="test-app-secret",
            ezviz_device_verify_code="ABC123",
        )
        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            auth = EzvizAuth(
                app_key="test-app-key",
                app_secret="test-app-secret",
                base_url=BASE_URL,
                http_client=http_client,
                now_ms=lambda: NOW_MS,
            )
            client = EzvizClient(settings, http_client=http_client, auth=auth)
            result = client.get_play_config("TEST123")

        self.assertEqual(
            result.play_url,
            "ezopen://ABC123@open.ys7.com/TEST123/1.live",
        )
        self.assertEqual(result.access_token, "mock-access-token")
        self.assertIsNone(result.expires_at)

    def test_unconfigured_credentials_return_clear_service_error(self) -> None:
        app.dependency_overrides[get_ezviz_client] = UnconfiguredPlayClient

        with TestClient(app) as client:
            response = client.get(
                "/api/ezviz/devices/TEST123/play-config"
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "EZVIZ AppKey/AppSecret are not configured on the backend",
        )


if __name__ == "__main__":
    unittest.main()
