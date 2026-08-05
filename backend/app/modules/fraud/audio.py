import io
import wave
from dataclasses import dataclass
from typing import Protocol


class InvalidAudioChunkError(ValueError):
    pass


class SpeechRecognitionUnavailableError(RuntimeError):
    pass


class SpeechRecognitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RelativeTranscriptSegment:
    start_ms: int
    end_ms: int
    text: str


class SpeechRecognizer(Protocol):
    def transcribe_wav(
        self,
        audio: bytes,
        *,
        duration_ms: int,
    ) -> list[RelativeTranscriptSegment]: ...


def validate_wav_chunk(audio: bytes, *, max_duration_ms: int = 15_000) -> int:
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            frame_rate = source.getframerate()
            frames = source.getnframes()
    except (EOFError, wave.Error) as exc:
        raise InvalidAudioChunkError("audio chunk must be a valid WAV file") from exc

    if channels not in {1, 2}:
        raise InvalidAudioChunkError("WAV chunk must be mono or stereo")
    if sample_width not in {1, 2, 3, 4}:
        raise InvalidAudioChunkError("unsupported WAV sample width")
    if frame_rate < 8_000 or frame_rate > 48_000:
        raise InvalidAudioChunkError("WAV sample rate must be between 8 kHz and 48 kHz")
    if frames <= 0:
        raise InvalidAudioChunkError("WAV chunk must contain audio frames")
    duration_ms = round(frames * 1000 / frame_rate)
    if duration_ms > max_duration_ms:
        raise InvalidAudioChunkError(f"WAV chunk duration must not exceed {max_duration_ms} ms")
    return duration_ms
