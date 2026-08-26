import unittest

from fastapi.testclient import TestClient

from app.modules.fall.multimodal_engine.api.ezviz import get_ezviz_client
from app.modules.fall.multimodal_engine.integrations.ezviz import EzvizConfigurationError, EzvizDevice
from app.modules.fall.multimodal_engine.integrations.ezviz.schemas import EzvizDeviceListResult
from app.modules.fall.multimodal_engine.main import app


class FakeEzvizClient:
    def list_devices(
        self,
        *,
        page_start: int = 0,
        page_size: int = 10,
    ) -> EzvizDeviceListResult:
        return EzvizDeviceListResult(
            devices=[
                EzvizDevice(
                    deviceSerial="TEST-ONLINE",
                    deviceName="客厅摄像机",
                    status=1,
                ),
                EzvizDevice(
                    deviceSerial="TEST-OFFLINE",
                    deviceName="卧室摄像机",
                    status=0,
                ),
            ],
            total=2,
            page_start=page_start,
            page_size=page_size,
        )


class UnconfiguredEzvizClient:
    def list_devices(self, **_: int) -> EzvizDeviceListResult:
        raise EzvizConfigurationError("missing credentials")


class EzvizPhase6BApiTest(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_device_list_is_exposed_through_backend_api(self) -> None:
        app.dependency_overrides[get_ezviz_client] = FakeEzvizClient

        with TestClient(app) as client:
            response = client.get(
                "/api/ezviz/devices",
                params={"page_start": 0, "page_size": 10},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["devices"][0]["device_name"], "客厅摄像机")
        self.assertEqual(payload["devices"][0]["device_id"], "TEST-ONLINE")
        self.assertTrue(payload["devices"][0]["online"])
        self.assertFalse(payload["devices"][1]["online"])
        self.assertEqual(payload["devices"][0]["source"], "EZVIZ")

    def test_unconfigured_credentials_return_clear_service_error(self) -> None:
        app.dependency_overrides[get_ezviz_client] = UnconfiguredEzvizClient

        with TestClient(app) as client:
            response = client.get("/api/ezviz/devices")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "EZVIZ AppKey/AppSecret are not configured on the backend",
        )


if __name__ == "__main__":
    unittest.main()
