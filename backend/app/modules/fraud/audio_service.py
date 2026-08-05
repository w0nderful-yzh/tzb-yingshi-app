import asyncio
from datetime import timedelta

from app.modules.fraud.audio import SpeechRecognizer, validate_wav_chunk
from app.modules.fraud.schemas import (
    FraudAnalyzeRequest,
    FraudAudioChunkData,
    FraudAudioChunkRequest,
    TranscriptSegment,
)
from app.modules.fraud.service import FraudSessionService


class FraudAudioService:
    def __init__(
        self,
        *,
        recognizer: SpeechRecognizer,
        fraud_session_service: FraudSessionService,
        max_chunk_bytes: int,
    ) -> None:
        self._recognizer = recognizer
        self._fraud_session_service = fraud_session_service
        self._max_chunk_bytes = max_chunk_bytes
        self._results: dict[tuple[str, str, str], FraudAudioChunkData] = {}
        self._lock = asyncio.Lock()

    async def analyze_chunk(
        self,
        metadata: FraudAudioChunkRequest,
        audio: bytes,
    ) -> FraudAudioChunkData:
        key = (metadata.device_id, metadata.session_id, metadata.chunk_id)
        async with self._lock:
            existing = self._results.get(key)
            if existing is not None:
                return existing.model_copy(update={"status": "duplicate"})
            if len(audio) > self._max_chunk_bytes:
                raise ValueError("audio chunk exceeds configured size limit")
            duration_ms = validate_wav_chunk(audio)
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
                source_event_id = f"audio:{metadata.chunk_id}:segment:{index:03d}"
                occurred_at = metadata.started_at + timedelta(milliseconds=start_ms)
                ended_at = metadata.started_at + timedelta(milliseconds=end_ms)
                fraud_result = await self._fraud_session_service.analyze(
                    FraudAnalyzeRequest(
                        session_id=metadata.session_id,
                        source_event_id=source_event_id,
                        device_id=metadata.device_id,
                        occurred_at=occurred_at,
                        ended_at=ended_at,
                        text=text,
                        elder_alone=metadata.elder_alone,
                    )
                )
                latest_risk = fraud_result.risk
                transcript_segments.append(
                    TranscriptSegment(
                        source_event_id=source_event_id,
                        occurred_at=occurred_at,
                        ended_at=ended_at,
                        text=text,
                    )
                )

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
