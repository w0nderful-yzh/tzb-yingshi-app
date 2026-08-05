import io
import wave

import pytest

from app.infrastructure.external.sensevoice.recognizer import (
    extract_timed_segments,
    extract_transcript,
)
from app.modules.fraud.audio import InvalidAudioChunkError, validate_wav_chunk


def _wav_bytes(*, duration_ms: int = 1_000, sample_rate: int = 16_000) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * round(sample_rate * duration_ms / 1000))
    return target.getvalue()


def test_validate_wav_chunk_returns_duration() -> None:
    assert validate_wav_chunk(_wav_bytes(duration_ms=1_250)) == 1_250


def test_validate_wav_chunk_rejects_invalid_payload() -> None:
    with pytest.raises(InvalidAudioChunkError, match="valid WAV"):
        validate_wav_chunk(b"not-a-wave")


def test_validate_wav_chunk_rejects_long_audio() -> None:
    with pytest.raises(InvalidAudioChunkError, match="must not exceed"):
        validate_wav_chunk(_wav_bytes(duration_ms=15_100))


def test_extract_timed_segments_prefers_sentence_boundaries() -> None:
    result = [
        {
            "text": "combined",
            "sentence_info": [
                {"start": 100, "end": 900, "text": "第一句"},
                {"start": 1_100, "end": 1_800, "text": "第二句"},
            ],
        }
    ]

    segments = extract_timed_segments(result)

    assert [(item.start_ms, item.end_ms, item.text) for item in segments] == [
        (100, 900, "第一句"),
        (1_100, 1_800, "第二句"),
    ]


def test_extract_transcript_removes_sensevoice_rich_tags_without_funasr() -> None:
    text = extract_transcript([{"text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>马上转账"}])

    assert text == "马上转账"
