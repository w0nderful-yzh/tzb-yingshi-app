import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from urllib.parse import parse_qs
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.modules.fall.multimodal_engine.api.ezviz import get_ezviz_client
from app.modules.fall.multimodal_engine.api.fall_inference import get_fall_inference_service
from app.modules.fall.multimodal_engine.core.config import Settings
from app.modules.fall.multimodal_engine.integrations.ezviz import EzvizAuth, EzvizClient
from app.modules.fall.multimodal_engine.integrations.ezviz.schemas import EzvizPlayConfigResult
from app.modules.fall.multimodal_engine.integrations.ezviz.live_capture import (
    EzvizLiveCapture,
    EzvizLiveCaptureError,
)
from app.modules.fall.multimodal_engine.main import app
from app.modules.fall.multimodal_engine.schemas.fall_inference import (
    FallInferenceJobResponse,
    FallInferenceJobStatus,
    FallInferenceSystemStatus,
)


BASE_URL = "https://open.ys7.com/api/lapp"
NOW_MS = 1_700_000_000_000


class EzvizStandardLiveAddressTest(unittest.TestCase):
    def test_requests_rtmp_address_for_server_side_capture(self) -> None:
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
                self.assertEqual(form["deviceSerial"], ["TEST123"])
                self.assertEqual(form["channelNo"], ["1"])
                self.assertEqual(form["protocol"], ["3"])
                self.assertEqual(form["quality"], ["2"])
                self.assertEqual(form["type"], ["1"])
                self.assertEqual(form["expireTime"], ["120"])
                return httpx.Response(
                    200,
                    json={
                        "code": "200",
                        "msg": "OK",
                        "data": {
                            "id": "address-id",
                            "url": "rtmp://example.invalid/live/signed-stream",
                            "expireTime": NOW_MS + 120_000,
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
            client = EzvizClient(settings, http_client=http_client, auth=auth)
            result = client.get_standard_live_address("TEST123")

        self.assertTrue(result.play_url.startswith("rtmp://"))
        self.assertEqual(result.device_id, "TEST123")

    def test_rejects_browser_only_ezopen_address(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
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
            return httpx.Response(
                200,
                json={
                    "code": "200",
                    "msg": "OK",
                    "data": {
                        "url": "ezopen://open.ys7.com/TEST123/1.live",
                    },
                },
            )

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
            client = EzvizClient(settings, http_client=http_client, auth=auth)
            with self.assertRaisesRegex(Exception, "unsupported ezopen"):
                client.get_standard_live_address("TEST123")


class EzvizLiveCaptureTest(unittest.TestCase):
    def test_signed_url_is_sent_over_stdin_not_command_line(self) -> None:
        secret_url = "rtmp://example.invalid/live?token=do-not-log"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python.exe"
            script = root / "capture_cli.py"
            output = root / "capture.mp4"
            python.write_bytes(b"python")
            script.write_text("# capture", encoding="utf-8")

            def fake_runner(command, **kwargs):
                self.assertNotIn(secret_url, command)
                self.assertEqual(kwargs["input"], f"{secret_url}\n")
                output.write_bytes(b"video")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "frames": 250,
                            "fps": 25.0,
                            "width": 1280,
                            "height": 720,
                        }
                    ),
                    stderr="",
                )

            capture = EzvizLiveCapture(
                python_executable=python,
                capture_script=script,
                process_runner=fake_runner,
            )
            report = capture.capture(secret_url, output, duration_seconds=10)

        self.assertEqual(report["frames"], 250)

    def test_ezopen_is_rejected_before_process_start(self) -> None:
        capture = EzvizLiveCapture(python_executable=Path("missing-python"))
        with self.assertRaisesRegex(EzvizLiveCaptureError, "browser-only"):
            capture.capture(
                "ezopen://open.ys7.com/TEST123/1.live",
                Path("capture.mp4"),
                duration_seconds=10,
            )


class FakeLiveFallService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.background_call = None

    def system_status(self) -> FallInferenceSystemStatus:
        return FallInferenceSystemStatus(
            ready=True,
            model_version="test-model",
            input_format="test-input",
            execution_mode="test",
            project_dir_exists=True,
            python_exists=True,
            checkpoints_found=6,
            ezviz_live_capture_ready=True,
            note="ready",
        )

    def allocate_live_capture(self):
        job_dir = self.root / "live-job"
        job_dir.mkdir()
        return "live-job", job_dir / "ezviz-live.mp4", job_dir

    def create_job(self, **kwargs):
        return FallInferenceJobResponse(
            job_id=kwargs["job_id"],
            status=FallInferenceJobStatus.QUEUED,
            session_id=kwargs["session_id"],
            device_id=kwargs["device_id"],
            filename=kwargs["filename"],
            record_non_alert_test_event=kwargs["record_non_alert_test_event"],
            created_at=datetime.now(timezone.utc),
        )

    def run_live_job(self, job_id, **kwargs):
        self.background_call = (job_id, kwargs)


class FakeStandardStreamClient:
    def get_standard_live_address(self, device_id, **kwargs):
        del kwargs
        return EzvizPlayConfigResult(
            device_id=device_id,
            channel_no=1,
            play_url="rtmp://example.invalid/live/signed",
            access_token="mock-access-token",
        )


class EzvizLiveFallApiTest(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_live_job_uses_existing_job_contract_without_exposing_stream_url(self) -> None:
        with TemporaryDirectory() as directory:
            service = FakeLiveFallService(Path(directory))
            app.dependency_overrides[get_fall_inference_service] = lambda: service
            app.dependency_overrides[get_ezviz_client] = FakeStandardStreamClient
            monitoring_session = SimpleNamespace(
                id="session-1",
                device_id="camera-anchor",
                enabled_modules=["FALL"],
            )
            with patch(
                "app.modules.fall.multimodal_engine.api.fall_inference.MonitoringService.get_current_session",
                return_value=monitoring_session,
            ):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/fall-inference/ezviz-live-jobs",
                        params={"device_id": "CAMERA-1", "capture_seconds": 10},
                    )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["filename"], "ezviz-live-10s.mp4")
        self.assertNotIn("rtmp://", response.text)
        self.assertEqual(service.background_call[0], "live-job")
        self.assertEqual(service.background_call[1]["capture_seconds"], 10)


if __name__ == "__main__":
    unittest.main()
