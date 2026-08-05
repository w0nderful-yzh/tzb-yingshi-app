import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

import pytest

from app.modules.fraud.audio import validate_wav_chunk
from app.modules.fraud.schemas import FraudAudioChunkRequest
from app.workers.ys7_media_stream_worker import Ys7MediaStreamWorker


class FakeAddressProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def get_live_address(
        self,
        *,
        device_serial: str,
        channel_no: int,
        protocol: str,
        quality: int,
    ) -> str:
        self.calls += 1
        assert device_serial == "camera-stream"
        assert channel_no == 1
        assert protocol == "flv"
        assert quality == 2
        return "https://stream.invalid/live.flv"


class FakePcmStreamSource:
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    async def stream(self, url: str, *, chunk_ms: int) -> AsyncIterator[bytes]:
        assert url == "https://stream.invalid/live.flv"
        assert chunk_ms == 5_000
        yield b"\x00\x00" * (16_000 * 5)
        await self.closed.wait()

    async def close(self) -> None:
        self.closed.set()


class CapturingAudioService:
    def __init__(self) -> None:
        self.calls: list[tuple[FraudAudioChunkRequest, bytes]] = []
        self.called = asyncio.Event()

    async def analyze_chunk(
        self,
        metadata: FraudAudioChunkRequest,
        audio: bytes,
    ) -> None:
        self.calls.append((metadata, audio))
        self.called.set()


@pytest.mark.asyncio
async def test_media_worker_converts_pcm_and_calls_fraud_audio_service() -> None:
    address_provider = FakeAddressProvider()
    stream_source = FakePcmStreamSource()
    audio_service = CapturingAudioService()
    worker = Ys7MediaStreamWorker(
        address_provider=address_provider,
        stream_source=stream_source,
        fraud_audio_service=audio_service,  # type: ignore[arg-type]
        device_serial="camera-stream",
        channel_no=1,
        protocol="flv",
        quality=2,
        chunk_ms=5_000,
        queue_maxsize=2,
        elder_alone=True,
    )

    await worker.start()
    await asyncio.wait_for(audio_service.called.wait(), timeout=1.0)
    await worker.stop()

    metadata, wav_audio = audio_service.calls[0]
    assert metadata.device_id == "camera-stream"
    assert metadata.elder_alone is True
    assert metadata.chunk_id == "stream-000000001"
    assert isinstance(metadata.started_at, datetime)
    assert validate_wav_chunk(wav_audio) == 5_000
    assert worker.chunks_processed == 1
    assert worker.running is False
    assert address_provider.calls == 1


def test_media_status_endpoint_defaults_to_disabled(client: object) -> None:
    response = client.get("/api/v1/integrations/ys7/media/status")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "enabled": False,
        "running": False,
        "connected": False,
        "session_id": None,
        "queue_depth": 0,
        "chunks_processed": 0,
        "chunks_dropped": 0,
        "reconnect_attempts": 0,
        "last_error": None,
    }
