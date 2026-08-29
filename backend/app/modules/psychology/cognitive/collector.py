"""Non-blocking PCM side-channel and bounded speech collection for cognition."""

import asyncio
import logging
import uuid
import wave
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import webrtcvad

from app.modules.psychology.cognitive.result_store import CognitiveResultStore
from app.modules.psychology.cognitive.schemas import (
    CognitiveAssessmentSnapshot,
    CognitiveInferenceJob,
)

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * FRAME_MS // 1_000
UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10 worker compatibility.

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PcmEnvelope:
    subject_key: str
    device_id: str
    pcm: bytes
    received_at: datetime


@dataclass(slots=True)
class _CollectionSession:
    assessment_id: str
    subject_key: str
    session_id: str
    device_id: str
    started_at: datetime
    last_received_at: datetime
    speech_pcm: bytearray = field(default_factory=bytearray)
    remainder: bytearray = field(default_factory=bytearray)
    voiced_frames: int = 0


class CognitiveAudioCollector:
    """Collects validated App PCM without blocking or consuming the Fraud relay."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        enabled: bool,
        queue_maxsize: int = 8,
        min_speech_seconds: float = 60.0,
        target_speech_seconds: float = 120.0,
        max_session_seconds: float = 30.0 * 60.0,
        cooldown_seconds: float = 15.0 * 60.0,
        job_ttl_seconds: float = 60.0 * 60.0,
        vad_mode: int = 2,
        voice_detector: Callable[[bytes], bool] | None = None,
    ) -> None:
        if min_speech_seconds <= 0 or target_speech_seconds < min_speech_seconds:
            raise ValueError("Cognitive speech target must be greater than or equal to minimum")
        if max_session_seconds < target_speech_seconds:
            raise ValueError("Cognitive session duration must cover the target speech duration")
        if queue_maxsize < 1:
            raise ValueError("Cognitive queue size must be positive")
        vad = webrtcvad.Vad(vad_mode)
        self._voice_detector = voice_detector or (
            lambda frame: bool(vad.is_speech(frame, SAMPLE_RATE))
        )
        self._enabled = enabled
        self._queue: asyncio.Queue[_PcmEnvelope] = asyncio.Queue(maxsize=queue_maxsize)
        self._store = CognitiveResultStore(runtime_root)
        self._min_speech_seconds = min_speech_seconds
        self._min_voiced_frames = round(min_speech_seconds * 1_000 / FRAME_MS)
        self._target_voiced_frames = round(target_speech_seconds * 1_000 / FRAME_MS)
        self._max_session_seconds = max_session_seconds
        self._cooldown_seconds = cooldown_seconds
        self._job_ttl_seconds = job_ttl_seconds
        self._sessions: dict[str, _CollectionSession] = {}
        self._cooldown_until: dict[str, datetime] = {}
        self._task: asyncio.Task[None] | None = None
        self.chunks_received = 0
        self.chunks_dropped = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        if not self._enabled or self._task is not None:
            return
        self._store.prepare()
        self._sweep_stale_jobs(datetime.now(UTC_TZ))
        self._task = asyncio.create_task(self._run(), name="cognitive-audio-collector")
        logger.info("Cognitive Collector started (16 kHz mono PCM16)")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        self._sessions.clear()
        logger.info("Cognitive Collector stopped; in-memory audio discarded")

    def push(
        self,
        *,
        subject_key: str,
        device_id: str,
        pcm: bytes,
        sample_rate: int,
    ) -> bool:
        """Queue a chunk without waiting; returns false when disabled/not running."""

        if not self._enabled or self._task is None:
            return False
        if sample_rate != SAMPLE_RATE:
            raise ValueError("Cognitive PCM sample rate must be 16000 Hz")
        if not pcm or len(pcm) % SAMPLE_WIDTH_BYTES != 0:
            raise ValueError("Cognitive PCM must contain signed 16-bit mono samples")
        envelope = _PcmEnvelope(
            subject_key=subject_key,
            device_id=device_id,
            pcm=pcm,
            received_at=datetime.now(UTC_TZ),
        )
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.chunks_dropped += 1
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(envelope)
        self.chunks_received += 1
        return True

    async def _run(self) -> None:
        while True:
            try:
                envelope = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                now = datetime.now(UTC_TZ)
                self._expire_sessions(now)
                self._sweep_stale_jobs(now)
                continue
            try:
                self._consume(envelope)
            except Exception:
                logger.exception("Cognitive Collector rejected PCM chunk")
            finally:
                self._queue.task_done()

    def _consume(self, envelope: _PcmEnvelope) -> None:
        session = self._sessions.get(envelope.subject_key)
        if session is None:
            cooldown_until = self._cooldown_until.get(envelope.subject_key)
            if cooldown_until is not None and envelope.received_at < cooldown_until:
                return
            session = self._new_session(envelope)
            self._sessions[envelope.subject_key] = session
        elif session.device_id != envelope.device_id:
            logger.warning(
                "Ignoring Cognitive PCM from device %s while session %s uses %s",
                envelope.device_id,
                session.session_id,
                session.device_id,
            )
            return

        session.last_received_at = envelope.received_at
        session.remainder.extend(envelope.pcm)
        while len(session.remainder) >= FRAME_BYTES:
            frame = bytes(session.remainder[:FRAME_BYTES])
            del session.remainder[:FRAME_BYTES]
            if self._voice_detector(frame):
                session.speech_pcm.extend(frame)
                session.voiced_frames += 1
                if session.voiced_frames >= self._target_voiced_frames:
                    self._publish(session, envelope.received_at)
                    return

        if self._elapsed_seconds(session, envelope.received_at) >= self._max_session_seconds:
            self._finish_at_timeout(session, envelope.received_at)

    def _new_session(self, envelope: _PcmEnvelope) -> _CollectionSession:
        session_id = f"cog-session-{uuid.uuid4().hex}"
        assessment_id = f"cog-{uuid.uuid4().hex}"
        session = _CollectionSession(
            assessment_id=assessment_id,
            subject_key=envelope.subject_key,
            session_id=session_id,
            device_id=envelope.device_id,
            started_at=envelope.received_at,
            last_received_at=envelope.received_at,
        )
        self._store.write_snapshot(
            CognitiveAssessmentSnapshot(
                assessment_id=assessment_id,
                subject_key=envelope.subject_key,
                session_id=session_id,
                status="processing",
                window_started_at=envelope.received_at,
                effective_speech_seconds=0.0,
            )
        )
        logger.info(
            "Cognitive collection started assessment=%s subject=%s",
            assessment_id,
            envelope.subject_key,
        )
        return session

    def _publish(self, session: _CollectionSession, ended_at: datetime) -> None:
        effective_seconds = session.voiced_frames * FRAME_MS / 1_000
        wav_bytes = self._build_wav(bytes(session.speech_pcm))
        job = CognitiveInferenceJob(
            assessment_id=session.assessment_id,
            subject_key=session.subject_key,
            session_id=session.session_id,
            device_id=session.device_id,
            window_started_at=session.started_at,
            window_ended_at=ended_at,
            effective_speech_seconds=effective_seconds,
            created_at=ended_at,
            expires_at=ended_at + timedelta(seconds=self._job_ttl_seconds),
        )
        self._store.write_snapshot(
            CognitiveAssessmentSnapshot(
                assessment_id=session.assessment_id,
                subject_key=session.subject_key,
                session_id=session.session_id,
                status="processing",
                window_started_at=session.started_at,
                window_ended_at=ended_at,
                effective_speech_seconds=effective_seconds,
            )
        )
        self._store.publish_job(job, wav_bytes)
        self._complete_session(session, ended_at)
        logger.info(
            "Cognitive inference job ready assessment=%s effective_speech_seconds=%.1f",
            session.assessment_id,
            effective_seconds,
        )

    def _finish_at_timeout(self, session: _CollectionSession, ended_at: datetime) -> None:
        if session.voiced_frames >= self._min_voiced_frames:
            self._publish(session, ended_at)
            return
        effective_seconds = session.voiced_frames * FRAME_MS / 1_000
        self._store.write_snapshot(
            CognitiveAssessmentSnapshot(
                assessment_id=session.assessment_id,
                subject_key=session.subject_key,
                session_id=session.session_id,
                status="failed",
                window_started_at=session.started_at,
                window_ended_at=ended_at,
                effective_speech_seconds=effective_seconds,
                completed_at=ended_at,
                failure_code="insufficient_speech",
                failure_message=(
                    f"Fewer than {self._min_speech_seconds:g} seconds of effective speech "
                    f"within {self._max_session_seconds:g} seconds"
                ),
            )
        )
        self._complete_session(session, ended_at)
        logger.warning(
            "Cognitive collection failed assessment=%s: insufficient speech %.1fs",
            session.assessment_id,
            effective_seconds,
        )

    def _expire_sessions(self, now: datetime) -> None:
        for session in list(self._sessions.values()):
            if self._elapsed_seconds(session, now) >= self._max_session_seconds:
                self._finish_at_timeout(session, now)

    def _complete_session(self, session: _CollectionSession, ended_at: datetime) -> None:
        self._sessions.pop(session.subject_key, None)
        self._cooldown_until[session.subject_key] = ended_at + timedelta(
            seconds=self._cooldown_seconds
        )

    def _sweep_stale_jobs(self, now: datetime) -> None:
        cutoff = now.timestamp() - self._job_ttl_seconds
        for directory in (self._store.inbox_dir, self._store.processing_dir):
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    logger.warning("Removed expired Cognitive temporary file: %s", path.name)

    @staticmethod
    def _elapsed_seconds(session: _CollectionSession, now: datetime) -> float:
        return max(0.0, (now - session.started_at).total_seconds())

    @staticmethod
    def _build_wav(pcm: bytes) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm)
        return output.getvalue()
