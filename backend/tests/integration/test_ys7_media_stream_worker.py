import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.modules.fraud.audio import RelativeTranscriptSegment, validate_wav_chunk
from app.modules.fraud.audio_service import FraudAudioService
from app.modules.fraud.ports import FraudRiskEventWrite
from app.modules.fraud.schemas import FraudAudioChunkRequest
from app.modules.fraud.service import FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore
from app.modules.fraud.voice_activity import FRAME_MS, SAMPLE_RATE
from app.workers.ys7_media_stream_worker import (
    Ys7MediaStreamWorker,
    _QueuedStreamingPcm,
)


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

    async def stream(self, url: str, *, frame_ms: int) -> AsyncIterator[bytes]:
        assert url == "https://stream.invalid/live.flv"
        assert frame_ms == FRAME_MS
        frame_samples = SAMPLE_RATE * FRAME_MS // 1_000
        speech_frame = b"\x01\x00" * frame_samples
        silence_frame = b"\x00\x00" * frame_samples
        for _ in range(25):
            yield speech_frame
        for _ in range(35):
            yield silence_frame
        await self.closed.wait()

    async def close(self) -> None:
        self.closed.set()


class CapturingAudioService:
    def __init__(self) -> None:
        self.calls: list[tuple[FraudAudioChunkRequest, bytes]] = []
        self.called = asyncio.Event()

    @property
    def streaming_enabled(self) -> bool:
        return False

    async def analyze_chunk(
        self,
        metadata: FraudAudioChunkRequest,
        audio: bytes,
    ) -> None:
        self.calls.append((metadata, audio))
        self.called.set()


class FakeFinalRecognizer:
    def transcribe_wav(
        self,
        audio: bytes,
        *,
        duration_ms: int,
    ) -> list[RelativeTranscriptSegment]:
        del audio
        return [
            RelativeTranscriptSegment(
                start_ms=0,
                end_ms=duration_ms,
                text="把短信验证码告诉我，不要告诉家人",
            )
        ]


class FakeStreamingSession:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe_pcm(self, pcm: bytes, *, is_final: bool) -> str:
        del pcm
        self.calls += 1
        return "把短信验证码告诉我" if is_final else "把短信验证码告诉"


class FakeStreamingRecognizer:
    def create_session(self) -> FakeStreamingSession:
        return FakeStreamingSession()


class CapturingRiskSink:
    def __init__(self) -> None:
        self.events: list[FraudRiskEventWrite] = []

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        self.events.append(event)

    async def retract_preliminary(
        self,
        *,
        source_event_id: str,
        reason: str,
    ) -> None:
        pass


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
        queue_maxsize=2,
        elder_alone=True,
        voice_detector=lambda frame: any(frame),
    )

    await worker.start()
    await asyncio.wait_for(audio_service.called.wait(), timeout=1.0)
    await worker.stop()

    metadata, wav_audio = audio_service.calls[0]
    assert metadata.device_id == "camera-stream"
    assert metadata.elder_alone is True
    assert metadata.chunk_id == "stream-000000001"
    assert isinstance(metadata.started_at, datetime)
    assert validate_wav_chunk(wav_audio) == 800
    assert worker.chunks_processed == 1
    assert worker.running is False
    assert address_provider.calls == 1


@pytest.mark.asyncio
async def test_media_worker_revises_streaming_partial_with_final_transcript() -> None:
    stream_source = FakePcmStreamSource()
    sink = CapturingRiskSink()
    session_service = FraudSessionService(
        visual_event_store=VisualEventStore(),
        risk_event_sink=sink,
    )
    audio_service = FraudAudioService(
        recognizer=FakeFinalRecognizer(),
        streaming_recognizer=FakeStreamingRecognizer(),
        fraud_session_service=session_service,
        max_chunk_bytes=10 * 1024 * 1024,
    )
    worker = Ys7MediaStreamWorker(
        address_provider=FakeAddressProvider(),
        stream_source=stream_source,
        fraud_audio_service=audio_service,
        device_serial="camera-stream",
        channel_no=1,
        protocol="flv",
        quality=2,
        queue_maxsize=32,
        elder_alone=True,
        voice_detector=lambda frame: any(frame),
    )

    await worker.start()
    for _ in range(200):
        if worker.chunks_processed == 1 and worker.partials_processed == 2:
            break
        await asyncio.sleep(0.01)
    session_id = worker.session_id
    await worker.stop()

    assert worker.partials_processed == 2
    assert worker.partials_failed == 0
    assert worker.chunks_processed == 1
    assert session_id is not None
    risk = await session_service.get_session(
        device_id="camera-stream",
        session_id=session_id,
    )
    assert risk is not None
    assert risk.state == "S5_CRITICAL_CONTROL"
    speech_event_ids = {
        item.get("speech_event_id")
        for item in risk.evidence_chain
        if item.get("source") == "speech"
    }
    assert speech_event_ids == {"speech-001"}
    assert len(sink.events) == 1


def test_media_status_endpoint_defaults_to_disabled(client: object) -> None:
    response = client.get("/api/v1/integrations/ys7/media/status")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "enabled": False,
        "source": "cloud",
        "running": False,
        "connected": False,
        "session_id": None,
        "queue_depth": 0,
        "chunks_processed": 0,
        "chunks_dropped": 0,
        "streaming_enabled": False,
        "partials_processed": 0,
        "partials_failed": 0,
        "reconnect_attempts": 0,
        "last_error": None,
        "models_ready": "DISABLED",
        "classifier_ready": True,
        "warmup_error": None,
    }


class _FakeAnalyzeData:
    def __init__(self, state: str, kinds: list[str]) -> None:
        self.risk = SimpleNamespace(state=state)
        self.speech_event = {"evidence_observations": [{"kind": kind} for kind in kinds]}


def _new_worker() -> Ys7MediaStreamWorker:
    return Ys7MediaStreamWorker(
        address_provider=FakeAddressProvider(),
        stream_source=FakePcmStreamSource(),
        fraud_audio_service=object(),  # type: ignore[arg-type]
        device_serial="camera-stream",
        channel_no=1,
        protocol="flv",
        quality=2,
        queue_maxsize=4,
        elder_alone=False,
    )


def _partial_job() -> _QueuedStreamingPcm:
    return _QueuedStreamingPcm(
        event_id="utterance-1",
        session_id="session-9",
        started_at=datetime.now(UTC),
        pcm=b"\x00" * 1_000,
        is_final=False,
    )


def test_partial_log_only_on_signal_change(caplog: pytest.LogCaptureFixture) -> None:
    worker = _new_worker()
    job = _partial_job()
    with caplog.at_level(logging.INFO, logger="app.workers.ys7_media_stream_worker"):
        worker._log_partial_signal(  # type: ignore[arg-type]
            job, _FakeAnalyzeData("S1_OBSERVING", ["identity_claim"]), 12.0
        )
        worker._log_partial_signal(  # type: ignore[arg-type]
            job, _FakeAnalyzeData("S1_OBSERVING", ["identity_claim"]), 30.0
        )
        worker._log_partial_signal(  # type: ignore[arg-type]
            job, _FakeAnalyzeData("S2_TRUST_BUILDING", ["identity_claim"]), 40.0
        )
        worker._log_partial_signal(  # type: ignore[arg-type]
            job,
            _FakeAnalyzeData(
                "S2_TRUST_BUILDING",
                ["identity_claim", "credential_request"],
            ),
            50.0,
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("fraud_partial")
    ]
    assert len(messages) == 3
    assert "wait=12ms" in messages[0]
    assert "state=S1_OBSERVING" in messages[0]
    assert "kinds=identity_claim" in messages[0]
    assert "state=S2_TRUST_BUILDING" in messages[1]
    assert "kinds=credential_request,identity_claim" in messages[2]
    assert "camera-stream" not in "\n".join(messages)
    assert "utterance-1" not in "\n".join(messages)


@pytest.mark.asyncio
async def test_worker_logs_partial_and_final_activity_without_raw_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = CapturingRiskSink()
    session_service = FraudSessionService(
        visual_event_store=VisualEventStore(),
        risk_event_sink=sink,
    )
    audio_service = FraudAudioService(
        recognizer=FakeFinalRecognizer(),
        streaming_recognizer=FakeStreamingRecognizer(),
        fraud_session_service=session_service,
        max_chunk_bytes=10 * 1024 * 1024,
    )
    worker = Ys7MediaStreamWorker(
        address_provider=FakeAddressProvider(),
        stream_source=FakePcmStreamSource(),
        fraud_audio_service=audio_service,
        device_serial="camera-stream",
        channel_no=1,
        protocol="flv",
        quality=2,
        queue_maxsize=32,
        elder_alone=True,
        voice_detector=lambda frame: any(frame),
    )
    with caplog.at_level(logging.INFO, logger="app.workers.ys7_media_stream_worker"):
        await worker.start()
        for _ in range(200):
            if worker.chunks_processed == 1 and worker.partials_processed >= 1:
                break
            await asyncio.sleep(0.01)
        await worker.stop()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(("fraud_partial", "fraud_final"))
    ]
    assert any(message.startswith("fraud_partial") for message in messages)
    assert any(message.startswith("fraud_final") for message in messages)
    joined = "\n".join(messages)
    assert "把短信" not in joined
    assert "camera-stream" not in joined
    assert "session-9" not in joined
