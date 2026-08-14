import re
import threading
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.modules.fraud.audio import (
    RelativeTranscriptSegment,
    SpeechRecognitionError,
    SpeechRecognitionUnavailableError,
)

_RICH_TAG = re.compile(r"<\|([^|>]+)\|>")
_LANGUAGE_TAGS = {"zh", "en", "yue", "ja", "ko", "nospeech"}
_EMOTION_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL"}
_CONTROL_TAGS = {"itn", "woitn"}


@dataclass(frozen=True, slots=True)
class SenseVoiceTags:
    language: str | None = None
    emotion: str | None = None
    audio_events: tuple[str, ...] = ()


def extract_rich_tags(raw_text: object) -> SenseVoiceTags:
    tags = [match.group(1).strip() for match in _RICH_TAG.finditer(str(raw_text or ""))]
    language = next((tag.lower() for tag in tags if tag.lower() in _LANGUAGE_TAGS), None)
    emotion = next((tag.upper() for tag in tags if tag.upper() in _EMOTION_TAGS), None)
    audio_events = tuple(
        dict.fromkeys(
            tag.lower()
            for tag in tags
            if tag.lower() not in _LANGUAGE_TAGS
            and tag.upper() not in _EMOTION_TAGS
            and tag.lower() not in _CONTROL_TAGS
        )
    )
    return SenseVoiceTags(
        language=language,
        emotion=emotion,
        audio_events=audio_events,
    )


def _postprocess_text(raw_text: object) -> str:
    text = str(raw_text or "").strip()
    try:
        from funasr.utils.postprocess_utils import (
            rich_transcription_postprocess,
        )
    except ImportError:
        return _RICH_TAG.sub("", text).strip()
    return str(rich_transcription_postprocess(text)).strip()


def extract_timed_segments(result: object) -> list[RelativeTranscriptSegment]:
    items = result if isinstance(result, list) else [result]
    segments: list[RelativeTranscriptSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_tags = extract_rich_tags(item.get("text"))
        sentence_info = item.get("sentence_info") or []
        added_sentence = False
        if isinstance(sentence_info, list):
            for sentence in sentence_info:
                if not isinstance(sentence, dict):
                    continue
                start_ms = sentence.get("start")
                end_ms = sentence.get("end")
                raw_text = sentence.get("text")
                text = _postprocess_text(raw_text)
                sentence_tags = extract_rich_tags(raw_text)
                if start_ms is not None and end_ms is not None and text:
                    segments.append(
                        RelativeTranscriptSegment(
                            start_ms=int(start_ms),
                            end_ms=int(end_ms),
                            text=text,
                            language=sentence_tags.language or item_tags.language,
                            emotion=sentence_tags.emotion or item_tags.emotion,
                            audio_events=sentence_tags.audio_events or item_tags.audio_events,
                        )
                    )
                    added_sentence = True
        if added_sentence or sentence_info:
            continue

        timestamps = item.get("timestamp") or []
        valid_timestamps = [
            pair
            for pair in timestamps
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and pair[0] is not None
            and pair[1] is not None
        ]
        text = _postprocess_text(item.get("text"))
        if valid_timestamps and text:
            segments.append(
                RelativeTranscriptSegment(
                    start_ms=int(valid_timestamps[0][0]),
                    end_ms=int(valid_timestamps[-1][1]),
                    text=text,
                    language=item_tags.language,
                    emotion=item_tags.emotion,
                    audio_events=item_tags.audio_events,
                )
            )
    return sorted(segments, key=lambda segment: segment.start_ms)


def extract_transcript(result: object) -> str:
    items = result if isinstance(result, list) else [result]
    return "".join(
        _postprocess_text(item.get("text")) for item in items if isinstance(item, dict)
    ).strip()


class SenseVoiceRecognizer:
    """Lazy, serialized FunASR adapter for short SenseVoice WAV chunks."""

    def __init__(self, *, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise SpeechRecognitionUnavailableError(
                    "SenseVoice dependencies are not installed"
                ) from exc
            self._model = AutoModel(
                model=self._model_name,
                device=self._device,
                disable_update=True,
            )
            return self._model

    def warmup(self) -> None:
        """Load the model and run one minimal silent inference pass.

        Failures propagate to the caller so startup can record a FAILED state;
        the recognizer still works lazily on first real request.
        """
        model = self._get_model()
        silence = BytesIO()
        with wave.open(silence, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(b"\x00\x00" * 16_000)
        with TemporaryDirectory(prefix="sensevoice-warmup-") as directory:
            audio_path = Path(directory) / "warmup.wav"
            audio_path.write_bytes(silence.getvalue())
            with self._inference_lock:
                model.generate(
                    input=str(audio_path),
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    disable_pbar=True,
                )

    def transcribe_wav(
        self,
        audio: bytes,
        *,
        duration_ms: int,
    ) -> list[RelativeTranscriptSegment]:
        try:
            model = self._get_model()
            with TemporaryDirectory(prefix="sensevoice-") as directory:
                audio_path = Path(directory) / "chunk.wav"
                audio_path.write_bytes(audio)
                with self._inference_lock:
                    result = model.generate(
                        input=str(audio_path),
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=60,
                        disable_pbar=True,
                    )
        except SpeechRecognitionUnavailableError:
            raise
        except Exception as exc:
            raise SpeechRecognitionError("SenseVoice transcription failed") from exc

        timed = extract_timed_segments(result)
        if timed:
            return timed
        text = extract_transcript(result)
        if not text:
            return []
        items = result if isinstance(result, list) else [result]
        tags = extract_rich_tags(
            "".join(str(item.get("text", "")) for item in items if isinstance(item, dict))
        )
        return [
            RelativeTranscriptSegment(
                start_ms=0,
                end_ms=duration_ms,
                text=text,
                language=tags.language,
                emotion=tags.emotion,
                audio_events=tags.audio_events,
            )
        ]
