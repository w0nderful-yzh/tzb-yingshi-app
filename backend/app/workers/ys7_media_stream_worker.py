import asyncio
import io
import logging
import time
import wave
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from app.infrastructure.external.ys7.api_client import Ys7LiveAddressProvider
from app.infrastructure.external.ys7.media_stream import PcmStreamSource
from app.modules.fraud.audio_service import FraudAudioService
from app.modules.fraud.latency import (
    FraudLatencyTrace,
    finish_trace,
    record_span,
    start_trace,
)
from app.modules.fraud.schemas import FraudAudioChunkRequest
from app.modules.fraud.session_tracker import FraudSessionTracker
from app.modules.fraud.voice_activity import FRAME_MS, VoiceActivitySegmenter, VoiceSegment

logger = logging.getLogger(__name__)
STREAMING_CHUNK_MS = 600
STREAMING_CHUNK_BYTES = 16_000 * 2 * STREAMING_CHUNK_MS // 1_000


@dataclass(frozen=True, slots=True)
class _QueuedPcmChunk:
    chunk_id: str
    session_id: str
    started_at: datetime
    pcm: bytes
    replaces_source_event_id: str | None = None
    enqueued_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(frozen=True, slots=True)
class _QueuedStreamingPcm:
    event_id: str
    session_id: str
    started_at: datetime
    pcm: bytes
    is_final: bool
    enqueued_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(slots=True)
class _OnlineUtterance:
    event_id: str
    session_id: str
    started_at: datetime
    buffer: bytearray


_AnalysisJob = _QueuedPcmChunk | _QueuedStreamingPcm


def pcm16_mono_to_wav(pcm: bytes, *, sample_rate: int = 16_000) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return target.getvalue()


class Ys7MediaStreamWorker:
    def __init__(
        self,
        *,
        address_provider: Ys7LiveAddressProvider,
        stream_source: PcmStreamSource,
        fraud_audio_service: FraudAudioService,
        device_serial: str | None,
        channel_no: int,
        protocol: Literal["hls", "rtmp", "flv"],
        quality: int,
        queue_maxsize: int,
        elder_alone: bool,
        vad_mode: int = 2,
        voice_detector: Callable[[bytes], bool] | None = None,
        session_tracker: FraudSessionTracker | None = None,
        stream_url: str | None = None,
    ) -> None:
        self._address_provider = address_provider
        self._stream_source = stream_source
        self._fraud_audio_service = fraud_audio_service
        self._device_serial = device_serial
        self._channel_no = channel_no
        self._protocol = protocol
        self._quality = quality
        self._elder_alone = elder_alone
        self._vad_mode = vad_mode
        self._voice_detector = voice_detector
        self._session_tracker = session_tracker or FraudSessionTracker()
        self._stream_url = stream_url
        self._queue: asyncio.Queue[_AnalysisJob] = asyncio.Queue(maxsize=queue_maxsize)
        self._stream_task: asyncio.Task[None] | None = None
        self._analysis_task: asyncio.Task[None] | None = None
        self.connected = False
        self.last_error: str | None = None
        self.reconnect_attempts = 0
        self.chunks_processed = 0
        self.chunks_dropped = 0
        self.partials_processed = 0
        self.partials_failed = 0
        self._chunk_sequence = 0
        self._utterance_sequence = 0

    @property
    def running(self) -> bool:
        return self._stream_task is not None and not self._stream_task.done()

    @property
    def session_id(self) -> str | None:
        return self._session_tracker.active_session_id

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self.running:
            return
        self._stream_task = asyncio.create_task(
            self._run_stream(),
            name="ys7-media-stream",
        )
        self._analysis_task = asyncio.create_task(
            self._run_analysis(),
            name="ys7-media-analysis",
        )

    async def stop(self) -> None:
        tasks = [task for task in (self._stream_task, self._analysis_task) if task is not None]
        for task in tasks:
            task.cancel()
        await self._stream_source.close()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._stream_task = None
        self._analysis_task = None
        self.connected = False

    async def _run_stream(self) -> None:
        backoff_seconds = 1
        while True:
            yielded_audio = False
            connection_anchor: datetime | None = None
            segmenter = VoiceActivitySegmenter(
                vad_mode=self._vad_mode,
                voice_detector=self._voice_detector,
            )
            online: _OnlineUtterance | None = None
            try:
                if not self._device_serial:
                    raise RuntimeError("YS7 device serial is not configured")
                live_address = self._stream_url
                if live_address is None:
                    live_address = await self._address_provider.get_live_address(
                        device_serial=self._device_serial,
                        channel_no=self._channel_no,
                        protocol=self._protocol,
                        quality=self._quality,
                    )
                async for pcm in self._stream_source.stream(
                    live_address,
                    frame_ms=FRAME_MS,
                ):
                    yielded_audio = True
                    self.connected = True
                    self.last_error = None
                    self.reconnect_attempts = 0
                    backoff_seconds = 1
                    if connection_anchor is None:
                        connection_anchor = datetime.now(UTC) - timedelta(milliseconds=FRAME_MS)
                    was_active = segmenter.speech_active
                    segments = segmenter.consume(pcm)
                    if self._fraud_audio_service.streaming_enabled:
                        if not was_active and segmenter.speech_active:
                            online = self._new_online_utterance(
                                connection_anchor
                                + timedelta(milliseconds=segmenter.active_start_offset_ms),
                                initial_pcm=segmenter.active_pcm,
                            )
                        elif was_active and online is not None:
                            online.buffer.extend(pcm)

                    if segments:
                        self._finalize_online(online)
                        for segment in segments:
                            self._publish_segment(
                                segment,
                                connection_anchor,
                                session_id=online.session_id if online is not None else None,
                                replaces_source_event_id=(
                                    online.event_id if online is not None else None
                                ),
                            )
                            online = (
                                self._new_online_utterance(
                                    connection_anchor
                                    + timedelta(
                                        milliseconds=segment.start_offset_ms
                                        + len(segment.pcm) * 1_000 // (16_000 * 2)
                                    )
                                )
                                if segment.continues and self._fraud_audio_service.streaming_enabled
                                else None
                            )
                    elif online is not None:
                        self._publish_online_chunks(online, is_final=False)
                if connection_anchor is not None:
                    self._finalize_online(online)
                    for segment in segmenter.flush():
                        self._publish_segment(
                            segment,
                            connection_anchor,
                            session_id=online.session_id if online is not None else None,
                            replaces_source_event_id=(
                                online.event_id if online is not None else None
                            ),
                        )
                raise RuntimeError("YS7 media stream ended")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if connection_anchor is not None:
                    self._finalize_online(online)
                    for segment in segmenter.flush():
                        self._publish_segment(
                            segment,
                            connection_anchor,
                            session_id=online.session_id if online is not None else None,
                            replaces_source_event_id=(
                                online.event_id if online is not None else None
                            ),
                        )
                self.connected = False
                self.reconnect_attempts += 1
                self.last_error = str(exc)
                logger.warning(
                    "YS7 media stream disconnected; retrying",
                    extra={
                        "attempt": self.reconnect_attempts,
                        "yielded_audio": yielded_audio,
                    },
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 60)

    def _new_online_utterance(
        self,
        started_at: datetime,
        *,
        initial_pcm: bytes = b"",
    ) -> _OnlineUtterance:
        self._utterance_sequence += 1
        session_id = self._session_tracker.session_for_segment(
            started_at=started_at,
            ended_at=started_at,
        )
        return _OnlineUtterance(
            event_id=f"stream-partial-{self._utterance_sequence:09d}",
            session_id=session_id,
            started_at=started_at,
            buffer=bytearray(initial_pcm),
        )

    def _publish_online_chunks(
        self,
        utterance: _OnlineUtterance,
        *,
        is_final: bool,
    ) -> None:
        minimum_remaining = 1 if is_final else 0
        while len(utterance.buffer) >= STREAMING_CHUNK_BYTES + minimum_remaining:
            pcm = bytes(utterance.buffer[:STREAMING_CHUNK_BYTES])
            del utterance.buffer[:STREAMING_CHUNK_BYTES]
            self._publish_latest(
                _QueuedStreamingPcm(
                    event_id=utterance.event_id,
                    session_id=utterance.session_id,
                    started_at=utterance.started_at,
                    pcm=pcm,
                    is_final=False,
                )
            )
        if is_final and utterance.buffer:
            self._publish_latest(
                _QueuedStreamingPcm(
                    event_id=utterance.event_id,
                    session_id=utterance.session_id,
                    started_at=utterance.started_at,
                    pcm=bytes(utterance.buffer),
                    is_final=True,
                )
            )
            utterance.buffer.clear()

    def _finalize_online(self, utterance: _OnlineUtterance | None) -> None:
        if utterance is not None:
            self._publish_online_chunks(utterance, is_final=True)

    def _publish_segment(
        self,
        segment: VoiceSegment,
        anchor: datetime,
        *,
        session_id: str | None = None,
        replaces_source_event_id: str | None = None,
    ) -> None:
        self._chunk_sequence += 1
        started_at = anchor + timedelta(milliseconds=segment.start_offset_ms)
        duration_ms = len(segment.pcm) * 1_000 // (16_000 * 2)
        session_id = session_id or self._session_tracker.session_for_segment(
            started_at=started_at,
            ended_at=started_at + timedelta(milliseconds=duration_ms),
        )
        self._publish_latest(
            _QueuedPcmChunk(
                chunk_id=f"stream-{self._chunk_sequence:09d}",
                session_id=session_id,
                started_at=started_at,
                pcm=segment.pcm,
                replaces_source_event_id=replaces_source_event_id,
            )
        )

    def _publish_latest(self, chunk: _AnalysisJob) -> None:
        if self._queue.full():
            with suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self._queue.task_done()
                self.chunks_dropped += 1
        self._queue.put_nowait(chunk)

    def _start_job_trace(self, job: _AnalysisJob) -> FraudLatencyTrace | None:
        if isinstance(job, _QueuedStreamingPcm):
            return start_trace(
                device_id=self._device_serial or "unknown",
                session_id=job.session_id,
                source_event_id=job.event_id,
                transcript_status="PARTIAL",
            )
        return start_trace(
            device_id=self._device_serial or "unknown",
            session_id=job.session_id,
            source_event_id=job.chunk_id,
            transcript_status="FINAL",
        )

    async def _run_analysis(self) -> None:
        while True:
            job = await self._queue.get()
            queue_wait_ms = (time.monotonic_ns() - job.enqueued_ns) / 1_000_000
            trace = self._start_job_trace(job)
            record_span("queue_wait", queue_wait_ms)
            try:
                if self._device_serial is None:
                    continue
                if isinstance(job, _QueuedStreamingPcm):
                    result = await self._fraud_audio_service.analyze_streaming_pcm(
                        session_id=job.session_id,
                        source_event_id=job.event_id,
                        device_id=self._device_serial,
                        started_at=job.started_at,
                        elder_alone=self._elder_alone,
                        pcm=job.pcm,
                        is_final=job.is_final,
                    )
                    if result is not None:
                        self.partials_processed += 1
                    continue
                await self._fraud_audio_service.analyze_chunk(
                    FraudAudioChunkRequest(
                        session_id=job.session_id,
                        chunk_id=job.chunk_id,
                        device_id=self._device_serial,
                        started_at=job.started_at,
                        elder_alone=self._elder_alone,
                        replaces_source_event_id=job.replaces_source_event_id,
                    ),
                    pcm16_mono_to_wav(job.pcm),
                )
                self.chunks_processed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                if isinstance(job, _QueuedStreamingPcm):
                    self.partials_failed += 1
                logger.exception(
                    "Failed to analyze YS7 audio chunk",
                    extra={
                        "chunk_id": (
                            job.event_id if isinstance(job, _QueuedStreamingPcm) else job.chunk_id
                        )
                    },
                )
            finally:
                self._queue.task_done()
                finish_trace(trace)
