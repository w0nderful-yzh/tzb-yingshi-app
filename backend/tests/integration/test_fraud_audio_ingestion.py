import io
import wave
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.modules.fraud.audio import (
    RelativeTranscriptSegment,
    SpeechRecognitionUnavailableError,
)


class FakeSpeechRecognizer:
    def __init__(self) -> None:
        self.call_count = 0

    def transcribe_wav(
        self,
        audio: bytes,
        *,
        duration_ms: int,
    ) -> list[RelativeTranscriptSegment]:
        self.call_count += 1
        return [
            RelativeTranscriptSegment(
                start_ms=500,
                end_ms=min(2_500, duration_ms),
                text="把短信验证码告诉我，不要告诉家人",
                language="zh",
                emotion="ANGRY",
                audio_events=("speech",),
            )
        ]


class UnavailableSpeechRecognizer:
    def transcribe_wav(
        self,
        audio: bytes,
        *,
        duration_ms: int,
    ) -> list[RelativeTranscriptSegment]:
        raise SpeechRecognitionUnavailableError("SenseVoice dependencies are not installed")


class OverlappingSpeechRecognizer:
    def __init__(self) -> None:
        self.call_count = 0

    def transcribe_wav(
        self,
        audio: bytes,
        *,
        duration_ms: int,
    ) -> list[RelativeTranscriptSegment]:
        del audio, duration_ms
        self.call_count += 1
        start_ms, end_ms = (9_000, 9_800) if self.call_count == 1 else (0, 800)
        return [
            RelativeTranscriptSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                text="把短信验证码告诉我",
            )
        ]


def _wav_bytes(*, duration_ms: int = 3_000) -> bytes:
    target = io.BytesIO()
    sample_rate = 16_000
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * round(sample_rate * duration_ms / 1000))
    return target.getvalue()


@pytest.fixture
def audio_client() -> Iterator[tuple[TestClient, FakeSpeechRecognizer]]:
    recognizer = FakeSpeechRecognizer()
    settings = Settings(
        environment="test",
        sensevoice_enabled=True,
        database_enabled=False,
        _env_file=None,
    )
    app = create_app(settings, speech_recognizer=recognizer)
    with TestClient(app) as client:
        yield client, recognizer


def _post_chunk(client: TestClient, *, chunk_id: str = "chunk-001") -> object:
    return client.post(
        "/api/v1/fraud/audio/chunks",
        data={
            "session_id": "audio-session",
            "chunk_id": chunk_id,
            "device_id": "camera-audio",
            "started_at": "2026-08-04T12:00:00+08:00",
            "elder_alone": "true",
        },
        files={"audio": ("chunk.wav", _wav_bytes(), "audio/wav")},
    )


def test_audio_chunk_is_transcribed_with_absolute_time_and_analyzed(
    audio_client: tuple[TestClient, FakeSpeechRecognizer],
) -> None:
    client, recognizer = audio_client

    response = _post_chunk(client)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "accepted"
    assert data["duration_ms"] == 3_000
    assert data["transcript_segments"][0]["occurred_at"] == ("2026-08-04T12:00:00.500000+08:00")
    assert data["transcript_segments"][0]["language"] == "zh"
    assert data["transcript_segments"][0]["emotion"] == "ANGRY"
    assert data["transcript_segments"][0]["audio_events"] == ["speech"]
    assert data["risk"]["state"] == "S5_CRITICAL_CONTROL"
    assert recognizer.call_count == 1


def test_duplicate_chunk_does_not_run_sensevoice_twice(
    audio_client: tuple[TestClient, FakeSpeechRecognizer],
) -> None:
    client, recognizer = audio_client

    first = _post_chunk(client)
    second = _post_chunk(client)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["data"]["status"] == "duplicate"
    assert recognizer.call_count == 1


def test_overlapping_audio_segments_do_not_duplicate_transcript_evidence() -> None:
    recognizer = OverlappingSpeechRecognizer()
    settings = Settings(
        environment="test",
        sensevoice_enabled=True,
        database_enabled=False,
        _env_file=None,
    )
    app = create_app(settings, speech_recognizer=recognizer)
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/fraud/audio/chunks",
            data={
                "session_id": "overlap-session",
                "chunk_id": "overlap-001",
                "device_id": "camera-audio",
                "started_at": "2026-08-04T12:00:00+08:00",
            },
            files={"audio": ("chunk.wav", _wav_bytes(duration_ms=10_000), "audio/wav")},
        )
        second = client.post(
            "/api/v1/fraud/audio/chunks",
            data={
                "session_id": "overlap-session",
                "chunk_id": "overlap-002",
                "device_id": "camera-audio",
                "started_at": "2026-08-04T12:00:09+08:00",
            },
            files={"audio": ("chunk.wav", _wav_bytes(), "audio/wav")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(first.json()["data"]["transcript_segments"]) == 1
    assert second.json()["data"]["transcript_segments"] == []
    assert recognizer.call_count == 2


def test_invalid_wav_is_rejected_before_recognition(
    audio_client: tuple[TestClient, FakeSpeechRecognizer],
) -> None:
    client, recognizer = audio_client
    response = client.post(
        "/api/v1/fraud/audio/chunks",
        data={
            "session_id": "audio-session",
            "chunk_id": "bad-chunk",
            "device_id": "camera-audio",
            "started_at": "2026-08-04T12:00:00+08:00",
        },
        files={"audio": ("chunk.wav", b"invalid", "audio/wav")},
    )

    assert response.status_code == 422
    assert recognizer.call_count == 0


def test_audio_endpoint_isolated_when_sensevoice_is_disabled(client: TestClient) -> None:
    response = _post_chunk(client)

    assert response.status_code == 503


def test_missing_model_runtime_returns_service_unavailable() -> None:
    settings = Settings(
        environment="test",
        sensevoice_enabled=True,
        database_enabled=False,
        _env_file=None,
    )
    app = create_app(settings, speech_recognizer=UnavailableSpeechRecognizer())
    with TestClient(app) as client:
        response = _post_chunk(client)

    assert response.status_code == 503
    assert response.json()["message"] == "SenseVoice dependencies are not installed"
