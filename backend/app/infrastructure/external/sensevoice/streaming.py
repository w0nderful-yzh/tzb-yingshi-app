import re
import threading
from typing import Any

from app.modules.fraud.audio import (
    SpeechRecognitionError,
    SpeechRecognitionUnavailableError,
    StreamingRecognitionSession,
)


class _ParaformerSession:
    def __init__(self, recognizer: "ParaformerStreamingRecognizer") -> None:
        self._recognizer = recognizer
        self._cache: dict[str, Any] = {}
        self._text = ""

    def transcribe_pcm(self, pcm: bytes, *, is_final: bool) -> str:
        piece = self._recognizer._generate(pcm, cache=self._cache, is_final=is_final)
        if piece:
            self._text += piece
            self._text = self._recognizer.correct_hotwords(self._text)
        return self._text


class ParaformerStreamingRecognizer:
    """Lazy FunASR Paraformer adapter for 600 ms incremental PCM recognition."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        hotwords: str = "",
        hotword_corrections: dict[str, str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._hotwords = " ".join(hotwords.split())
        self._hotword_corrections = hotword_corrections or {}
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def create_session(self) -> StreamingRecognitionSession:
        return _ParaformerSession(self)

    def correct_hotwords(self, text: str) -> str:
        corrected = text
        for wrong, expected in self._hotword_corrections.items():
            corrected = corrected.replace(wrong, expected)
        return re.sub(r"\s+", "", corrected).strip()

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
                    "FunASR streaming dependencies are not installed"
                ) from exc
            self._model = AutoModel(
                model=self._model_name,
                device=self._device,
                disable_update=True,
            )
            return self._model

    def _generate(self, pcm: bytes, *, cache: dict[str, Any], is_final: bool) -> str:
        try:
            import numpy as np
        except ImportError as exc:
            raise SpeechRecognitionUnavailableError(
                "NumPy is required for streaming speech recognition"
            ) from exc
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32_768.0
        if samples.size == 0 and not is_final:
            return ""
        try:
            model = self._get_model()
            with self._inference_lock:
                result = model.generate(
                    input=samples,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=[0, 10, 5],
                    encoder_chunk_look_back=4,
                    decoder_chunk_look_back=1,
                    hotword=self._hotwords,
                    disable_pbar=True,
                )
        except SpeechRecognitionUnavailableError:
            raise
        except Exception as exc:
            raise SpeechRecognitionError("Paraformer streaming transcription failed") from exc
        items = result if isinstance(result, list) else [result]
        return "".join(
            str(item.get("text", "")) for item in items if isinstance(item, dict)
        ).strip()
