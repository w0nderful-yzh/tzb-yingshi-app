import json
import unittest
from urllib.parse import parse_qs

import httpx

from app.modules.fall.multimodal_engine.core.config import Settings
from app.modules.fall.multimodal_engine.integrations.ezviz.auth import (
    EzvizApiError,
    EzvizAuth,
    EzvizConfigurationError,
)
from app.modules.fall.multimodal_engine.integrations.ezviz.client import EzvizClient


BASE_URL = "https://open.ys7.com/api/lapp"
NOW_MS = 1_700_000_000_000


class EzvizPhase6ATest(unittest.TestCase):
    def setUp(self) -> None:
        self.token_requests = 0
        self.device_requests = 0

        def handler(request: httpx.Request) -> httpx.Response:
            form = parse_qs(request.content.decode("utf-8"))

            if request.url.path.endswith("/token/get"):
                self.token_requests += 1
                self.assertEqual(form["appKey"], ["test-app-key"])
                self.assertEqual(form["appSecret"], ["test-app-secret"])
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "msg": "OK",
                        "data": {
                            "accessToken": "test-access-token",
                            "expireTime": NOW_MS + 3_600_000,
                        },
                    },
                )

            if request.url.path.endswith("/device/list"):
                self.device_requests += 1
                self.assertEqual(form["accessToken"], ["test-access-token"])
                self.assertEqual(form["pageStart"], ["0"])
                self.assertEqual(form["pageSize"], ["10"])
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "msg": "OK",
                        "data": [
                            {
                                "deviceSerial": "TEST123",
                                "deviceName": "客厅摄像机",
                                "deviceType": "CS-C6N",
                                "status": 1,
                                "defence": 1,
                                "deviceVersion": "V1.0.0",
                            }
                        ],
                        "page": {"total": 1, "page": 0, "size": 10},
                    },
                )

            return httpx.Response(404, json={"message": "not found"})

        self.http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.settings = Settings(
            _env_file=None,
            ezviz_base_url=BASE_URL,
            ezviz_app_key="test-app-key",
            ezviz_app_secret="test-app-secret",
        )
        self.auth = EzvizAuth(
            app_key="test-app-key",
            app_secret="test-app-secret",
            base_url=BASE_URL,
            http_client=self.http_client,
            now_ms=lambda: NOW_MS,
        )

    def tearDown(self) -> None:
        self.http_client.close()

    def test_device_list_uses_one_cached_token_for_repeated_requests(self) -> None:
        client = EzvizClient(
            self.settings,
            http_client=self.http_client,
            auth=self.auth,
        )

        first = client.list_devices()
        second = client.list_devices()

        self.assertEqual(first.total, 1)
        self.assertEqual(first.devices[0].device_serial, "TEST123")
        self.assertEqual(first.devices[0].device_name, "客厅摄像机")
        self.assertEqual(second.devices[0].status, 1)
        self.assertEqual(self.token_requests, 1)
        self.assertEqual(self.device_requests, 2)

    def test_force_refresh_bypasses_token_cache(self) -> None:
        self.assertEqual(self.auth.get_access_token(), "test-access-token")
        self.assertEqual(
            self.auth.get_access_token(force_refresh=True),
            "test-access-token",
        )
        self.assertEqual(self.token_requests, 2)

    def test_missing_credentials_fail_before_network_call(self) -> None:
        auth = EzvizAuth(
            app_key="",
            app_secret="",
            base_url=BASE_URL,
            http_client=self.http_client,
        )

        with self.assertRaises(EzvizConfigurationError):
            auth.get_access_token()

        self.assertEqual(self.token_requests, 0)

    def test_api_error_preserves_platform_error_code(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps(
                    {"code": "10001", "msg": "parameter error"}
                ).encode("utf-8"),
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            auth = EzvizAuth(
                app_key="test-app-key",
                app_secret="test-app-secret",
                base_url=BASE_URL,
                http_client=http_client,
            )

            with self.assertRaises(EzvizApiError) as context:
                auth.get_access_token()

        self.assertEqual(context.exception.code, "10001")


if __name__ == "__main__":
    unittest.main()
