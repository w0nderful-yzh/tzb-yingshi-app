import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from app.modules.fraud.audio import (
    SpeechRecognizer,
    StreamingRecognitionSession,
    StreamingSpeechRecognizer,
    validate_wav_chunk,
)
from app.modules.fraud.latency import finish_trace, latency_stage, start_trace
from app.modules.fraud.schemas import (
    FraudAnalyzeData,
    FraudAnalyzeRequest,
    FraudAudioChunkData,
    FraudAudioChunkRequest,
    TranscriptSegment,
)
from app.modules.fraud.service import FraudSessionService


@dataclass(slots=True)
class _StreamingState:
    recognition_session: StreamingRecognitionSession
    duration_ms: int = 0
    last_text: str = ""


class FraudAudioService:
    def __init__(
        self,
        *,
        recognizer: SpeechRecognizer,
        fraud_session_service: FraudSessionService,
        max_chunk_bytes: int,
        streaming_recognizer: StreamingSpeechRecognizer | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._fraud_session_service = fraud_session_service
        self._max_chunk_bytes = max_chunk_bytes
        self._streaming_recognizer = streaming_recognizer
        self._results: dict[tuple[str, str, str], FraudAudioChunkData] = {}
        self._transcript_history: dict[tuple[str, str], list[TranscriptSegment]] = {}
        self._streaming_states: dict[str, _StreamingState] = {}
        self._lock = asyncio.Lock()

    @property
    def streaming_enabled(self) -> bool:
        return self._streaming_recognizer is not None

    async def analyze_streaming_pcm(
        self,
        *,
        session_id: str,
        source_event_id: str,
        device_id: str,
        started_at: datetime,
        elder_alone: bool,
        pcm: bytes,
        is_final: bool,
    ) -> FraudAnalyzeData | None:
        if self._streaming_recognizer is None:
            return None
        trace = start_trace(
            device_id=device_id,
            session_id=session_id,
            source_event_id=source_event_id,
            transcript_status="PARTIAL",
        )
        try:
            async with self._lock:
                state = self._streaming_states.get(source_event_id)
                if state is None:
                    state = _StreamingState(self._streaming_recognizer.create_session())
                    self._streaming_states[source_event_id] = state
                duration_ms = len(pcm) * 1_000 // (16_000 * 2)
                state.duration_ms += duration_ms
                with latency_stage("asr_recognize"):
                    text = await asyncio.to_thread(
                        state.recognition_session.transcribe_pcm,
                        pcm,
                        is_final=is_final,
                    )
                if is_final:
                    self._streaming_states.pop(source_event_id, None)
                normalized = text.strip()
                if not normalized or normalized == state.last_text:
                    return None
                state.last_text = normalized
                return await self._fraud_session_service.analyze(
                    FraudAnalyzeRequest(
                        session_id=session_id,
                        source_event_id=source_event_id,
                        device_id=device_id,
                        occurred_at=started_at,
                        ended_at=started_at + timedelta(milliseconds=state.duration_ms),
                        text=normalized,
                        transcript_status="PARTIAL",
                        elder_alone=elder_alone,
                    )
                )
        finally:
            finish_trace(trace)

    async def analyze_chunk(
        self,
        metadata: FraudAudioChunkRequest,
        audio: bytes,
    ) -> FraudAudioChunkData:
        trace = start_trace(
            device_id=metadata.device_id,
            session_id=metadata.session_id,
            source_event_id=metadata.chunk_id,
            transcript_status="FINAL",
        )
        try:
            key = (metadata.device_id, metadata.session_id, metadata.chunk_id)
            async with self._lock:
                existing = self._results.get(key)
                if existing is not None:
                    return existing.model_copy(update={"status": "duplicate"})
                if len(audio) > self._max_chunk_bytes:
                    raise ValueError("audio chunk exceeds configured size limit")
                duration_ms = validate_wav_chunk(audio)
                with latency_stage("asr_recognize"):
                    relative_segments = await asyncio.to_thread(
                        self._recognizer.transcribe_wav,
                        audio,
                        duration_ms=duration_ms,
                    )

                transcript_segments: list[TranscriptSegment] = []
                latest_risk = None
                for index, segment in enumerate(relative_segments, start=1):
                    start_ms = max(0, min(segment.start_ms, duration_ms))
                    end_ms = max(start_ms, min(segment.end_ms, duration_ms))
                    text = segment.text.strip()
                    if not text:
                        continue
                    source_event_id = (
                        metadata.replaces_source_event_id
                        if index == 1 and metadata.replaces_source_event_id is not None
                        else f"audio:{metadata.chunk_id}:segment:{index:03d}"
                    )
                    occurred_at = metadata.started_at + timedelta(milliseconds=start_ms)
                    ended_at = metadata.started_at + timedelta(milliseconds=end_ms)
                    transcript_segment = TranscriptSegment(
                        source_event_id=source_event_id,
                        occurred_at=occurred_at,
                        ended_at=ended_at,
                        text=text,
                        transcript_status="FINAL",
                        language=segment.language,
                        emotion=segment.emotion,
                        audio_events=list(segment.audio_events),
                    )
                    if self._is_overlapping_duplicate(metadata, transcript_segment):
                        continue
                    fraud_result = await self._fraud_session_service.analyze(
                        FraudAnalyzeRequest(
                            session_id=metadata.session_id,
                            source_event_id=source_event_id,
                            device_id=metadata.device_id,
                            occurred_at=occurred_at,
                            ended_at=ended_at,
                            text=text,
                            transcript_status="FINAL",
                            elder_alone=metadata.elder_alone,
                            language=segment.language,
                            emotion=segment.emotion,
                            audio_events=list(segment.audio_events),
                        )
                    )
                    latest_risk = fraud_result.risk
                    transcript_segments.append(transcript_segment)
                    self._remember_transcript(metadata, transcript_segment)

                if latest_risk is None:
                    latest_risk = await self._fraud_session_service.get_session(
                        device_id=metadata.device_id,
                        session_id=metadata.session_id,
                    )
                chunk_result = FraudAudioChunkData(
                    status="accepted",
                    chunk_id=metadata.chunk_id,
                    duration_ms=duration_ms,
                    transcript_segments=transcript_segments,
                    risk=latest_risk,
                )
                self._results[key] = chunk_result
                return chunk_result
        finally:
            finish_trace(trace)

    def _is_overlapping_duplicate(
        self,
        metadata: FraudAudioChunkRequest,
        candidate: TranscriptSegment,
    ) -> bool:
        key = (metadata.device_id, metadata.session_id)
        history = self._transcript_history.get(key, [])
        candidate_text = re.sub(r"[\W_]+", "", candidate.text)
        if not candidate_text:
            return False
        for existing in history:
            overlap_start = max(existing.occurred_at, candidate.occurred_at)
            overlap_end = min(existing.ended_at, candidate.ended_at)
            if overlap_end <= overlap_start:
                continue
            existing_text = re.sub(r"[\W_]+", "", existing.text)
            similarity = SequenceMatcher(None, existing_text, candidate_text).ratio()
            if similarity >= 0.82:
                return True
        return False

    def _remember_transcript(
        self,
        metadata: FraudAudioChunkRequest,
        segment: TranscriptSegment,
    ) -> None:
        key = (metadata.device_id, metadata.session_id)
        cutoff = segment.occurred_at - timedelta(seconds=120)
        history = self._transcript_history.setdefault(key, [])
        history[:] = [item for item in history if item.ended_at >= cutoff]
        history.append(segment)
