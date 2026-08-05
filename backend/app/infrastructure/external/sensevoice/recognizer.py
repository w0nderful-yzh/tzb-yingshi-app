import re
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.modules.fraud.audio import (
    RelativeTranscriptSegment,
    SpeechRecognitionError,
    SpeechRecognitionUnavailableError,
)

_RICH_TAG = re.compile(r"<\|[^|>]+\|>")


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
        sentence_info = item.get("sentence_info") or []
        added_sentence = False
        if isinstance(sentence_info, list):
            for sentence in sentence_info:
                if not isinstance(sentence, dict):
                    continue
                start_ms = sentence.get("start")
                end_ms = sentence.get("end")
                text = _postprocess_text(sentence.get("text"))
                if start_ms is not None and end_ms is not None and text:
                    segments.append(
                        RelativeTranscriptSegment(
                            start_ms=int(start_ms),
                            end_ms=int(end_ms),
                            text=text,
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
        return [
            RelativeTranscriptSegment(
                start_ms=0,
                end_ms=duration_ms,
                text=text,
            )
        ]
