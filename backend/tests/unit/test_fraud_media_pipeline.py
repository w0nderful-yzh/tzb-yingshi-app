import asyncio
import wave
from io import BytesIO
from typing import Any

import pytest

from app.core.config import Settings
from app.modules.fraud.audio import RelativeTranscriptSegment
from app.modules.fraud.audio_service import FraudAudioService
from app.modules.fraud.ports import FraudRiskEventWrite
from app.modules.fraud.service import FraudSessionService
from app.modules.fraud.visual_event_store import VisualEventStore
from app.modules.fraud.voice_activity import VoiceActivitySegmenter
from app.workers.ys7_media_stream_worker import (
    Ys7MediaStreamWorker,
    pcm16_mono_to_wav,
)

FRAME_BYTES = 16_000 * 2 * 20 // 1_000


class FakeAddressProvider:
    async def get_live_address(self, **kwargs: Any) -> str:
        return "fake://live"


class FakePcmStreamSource:
    def __init__(self) -> None:
        self._frames: list[bytes] = [b"\x00" * FRAME_BYTES] * 8 + [b"\xff" * FRAME_BYTES] * 20
        self.closed = False

    async def stream(self, live_address: str, *, frame_ms: int) -> Any:
        for frame in self._frames:
            yield frame

    async def close(self) -> None:
        self.closed = True


class FakeFinalRecognizer:
    def transcribe_wav(self, audio: bytes, *, duration_ms: int) -> list[Any]:
        return [
            RelativeTranscriptSegment(
                start_ms=0,
                end_ms=duration_ms,
                text="把短信验证码告诉我",
                language="zh",
                emotion=None,
                audio_events=(),
            )
        ]


class FakeStreamingSession:
    def __init__(self) -> None:
        self._text = ""

    def transcribe_pcm(self, pcm: bytes, *, is_final: bool) -> str:
        self._text = "把短信验证码告诉我"
        return self._text


class FakeStreamingRecognizer:
    def create_session(self) -> FakeStreamingSession:
        return FakeStreamingSession()


class CapturingRiskSink:
    def __init__(self) -> None:
        self.events: list[FraudRiskEventWrite] = []

    async def upsert(self, event: FraudRiskEventWrite) -> None:
        self.events.append(event)

    async def retract_preliminary(self, *, source_event_id: str, reason: str) -> None:
        pass


def _silence(ms: int) -> bytes:
    return b"\x00" * (16_000 * 2 * ms // 1_000)


def _speech(ms: int) -> bytes:
    return b"\xff" * (16_000 * 2 * ms // 1_000)


def test_vad_segmenter_rejects_non_frame_aligned_timing() -> None:
    with pytest.raises(ValueError):
        VoiceActivitySegmenter(speech_start_ms=210, silence_end_ms=700)
    with pytest.raises(ValueError):
        VoiceActivitySegmenter(speech_start_ms=200, silence_end_ms=750)
    with pytest.raises(ValueError):
        VoiceActivitySegmenter(speech_start_ms=0, silence_end_ms=700)
    with pytest.raises(ValueError):
        VoiceActivitySegmenter(speech_start_ms=200, silence_end_ms=6_000)


def test_vad_segmenter_respects_silence_end_param() -> None:
    segmenter = VoiceActivitySegmenter(
        voice_detector=lambda frame: any(frame),
        speech_start_ms=200,
        silence_end_ms=500,
    )
    frames = _speech(1_000) + _silence(1_000) + _speech(200) + _silence(600)
    segments: list[Any] = []
    for offset in range(0, len(frames), FRAME_BYTES):
        segments.extend(segmenter.consume(frames[offset : offset + FRAME_BYTES]))
    segments.extend(segmenter.flush())
    assert len(segments) == 2
    assert segments[0].start_offset_ms == 0


def test_settings_reject_non_frame_aligned_vad_timing() -> None:
    with pytest.raises(ValueError):
        Settings(environment="test", _env_file=None, ys7_vad_silence_end_ms=705)
    with pytest.raises(ValueError):
        Settings(environment="test", _env_file=None, streaming_chunk_ms=610)


def test_settings_accept_frame_aligned_timing() -> None:
    settings = Settings(
        environment="test",
        _env_file=None,
        ys7_vad_silence_end_ms=500,
        streaming_chunk_ms=600,
    )
    assert settings.ys7_vad_silence_end_ms == 500
    assert settings.streaming_chunk_ms == 600


@pytest.mark.asyncio
async def test_worker_splits_streaming_and_final_queues() -> None:
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
        queue_maxsize=8,
        elder_alone=True,
        voice_detector=lambda frame: any(frame),
    )
    assert worker._streaming_queue is not worker._final_queue

    await worker.start()
    for _ in range(200):
        if worker.chunks_processed == 1 and worker.partials_processed >= 1:
            break
        await asyncio.sleep(0.01)
    await worker.stop()

    assert worker.chunks_processed == 1
    assert worker.partials_processed >= 1
    assert worker.partials_failed == 0
    assert worker.chunks_dropped == 0
    assert worker.partials_dropped == 0


@pytest.mark.asyncio
async def test_final_uses_same_turn_event_id_as_partial() -> None:
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
        queue_maxsize=8,
        elder_alone=True,
        voice_detector=lambda frame: any(frame),
    )
    await worker.start()
    for _ in range(200):
        if worker.chunks_processed == 1:
            break
        await asyncio.sleep(0.01)
    session_id = worker.session_id
    await worker.stop()

    risk = await session_service.get_session(
        device_id="camera-stream",
        session_id=session_id,
    )
    assert risk is not None
    speech_ids = {
        item.get("speech_event_id")
        for item in risk.evidence_chain
        if item.get("source") == "speech"
    }
    assert len(speech_ids) == 1


def test_streaming_chunk_size_follows_configured_chunk_ms() -> None:
    worker = Ys7MediaStreamWorker(
        address_provider=FakeAddressProvider(),
        stream_source=FakePcmStreamSource(),
        fraud_audio_service=object(),  # type: ignore[arg-type]
        device_serial="camera-stream",
        channel_no=1,
        protocol="flv",
        quality=2,
        queue_maxsize=4,
        elder_alone=False,
        streaming_chunk_ms=400,
    )
    assert worker._streaming_chunk_bytes == 16_000 * 2 * 400 // 1_000
    with pytest.raises(ValueError):
        Ys7MediaStreamWorker(
            address_provider=FakeAddressProvider(),
            stream_source=FakePcmStreamSource(),
            fraud_audio_service=object(),  # type: ignore[arg-type]
            device_serial="camera-stream",
            channel_no=1,
            protocol="flv",
            quality=2,
            queue_maxsize=4,
            elder_alone=False,
            streaming_chunk_ms=610,
        )


def test_wav_conversion_roundtrip() -> None:
    pcm = _speech(200)
    wav = pcm16_mono_to_wav(pcm)
    with wave.open(BytesIO(wav), "rb") as source:
        assert source.getframerate() == 16_000
        assert source.getnchannels() == 1
        assert source.getnframes() == len(pcm) // 2
