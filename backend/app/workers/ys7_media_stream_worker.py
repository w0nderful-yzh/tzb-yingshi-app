import asyncio
import io
import logging
import wave
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from app.infrastructure.external.ys7.api_client import Ys7LiveAddressProvider
from app.infrastructure.external.ys7.media_stream import PcmStreamSource
from app.modules.fraud.audio_service import FraudAudioService
from app.modules.fraud.schemas import FraudAudioChunkRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _QueuedPcmChunk:
    chunk_id: str
    started_at: datetime
    pcm: bytes


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
        chunk_ms: int,
        queue_maxsize: int,
        elder_alone: bool,
    ) -> None:
        self._address_provider = address_provider
        self._stream_source = stream_source
        self._fraud_audio_service = fraud_audio_service
        self._device_serial = device_serial
        self._channel_no = channel_no
        self._protocol = protocol
        self._quality = quality
        self._chunk_ms = chunk_ms
        self._elder_alone = elder_alone
        self._queue: asyncio.Queue[_QueuedPcmChunk] = asyncio.Queue(maxsize=queue_maxsize)
        self._stream_task: asyncio.Task[None] | None = None
        self._analysis_task: asyncio.Task[None] | None = None
        self._session_id: str | None = None
        self.connected = False
        self.last_error: str | None = None
        self.reconnect_attempts = 0
        self.chunks_processed = 0
        self.chunks_dropped = 0
        self._chunk_sequence = 0

    @property
    def running(self) -> bool:
        return self._stream_task is not None and not self._stream_task.done()

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self.running:
            return
        self._session_id = f"ys7-live-{uuid4().hex[:16]}"
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
            yielded_chunk = False
            try:
                if not self._device_serial:
                    raise RuntimeError("YS7 device serial is not configured")
                live_address = await self._address_provider.get_live_address(
                    device_serial=self._device_serial,
                    channel_no=self._channel_no,
                    protocol=self._protocol,
                    quality=self._quality,
                )
                self.connected = True
                connection_anchor: datetime | None = None
                connection_chunk_index = 0
                async for pcm in self._stream_source.stream(
                    live_address,
                    chunk_ms=self._chunk_ms,
                ):
                    yielded_chunk = True
                    self.last_error = None
                    self.reconnect_attempts = 0
                    backoff_seconds = 1
                    if connection_anchor is None:
                        connection_anchor = datetime.now(UTC) - timedelta(
                            milliseconds=self._chunk_ms
                        )
                    started_at = connection_anchor + timedelta(
                        milliseconds=connection_chunk_index * self._chunk_ms
                    )
                    connection_chunk_index += 1
                    self._chunk_sequence += 1
                    self._publish_latest(
                        _QueuedPcmChunk(
                            chunk_id=f"stream-{self._chunk_sequence:09d}",
                            started_at=started_at,
                            pcm=pcm,
                        )
                    )
                raise RuntimeError("YS7 media stream ended")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.reconnect_attempts += 1
                self.last_error = str(exc)
                logger.warning(
                    "YS7 media stream disconnected; retrying",
                    extra={
                        "attempt": self.reconnect_attempts,
                        "yielded_chunk": yielded_chunk,
                    },
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 60)

    def _publish_latest(self, chunk: _QueuedPcmChunk) -> None:
        if self._queue.full():
            with suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self._queue.task_done()
                self.chunks_dropped += 1
        self._queue.put_nowait(chunk)

    async def _run_analysis(self) -> None:
        while True:
            chunk = await self._queue.get()
            try:
                if self._device_serial is None or self._session_id is None:
                    continue
                await self._fraud_audio_service.analyze_chunk(
                    FraudAudioChunkRequest(
                        session_id=self._session_id,
                        chunk_id=chunk.chunk_id,
                        device_id=self._device_serial,
                        started_at=chunk.started_at,
                        elder_alone=self._elder_alone,
                    ),
                    pcm16_mono_to_wav(chunk.pcm),
                )
                self.chunks_processed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed to analyze YS7 audio chunk",
                    extra={"chunk_id": chunk.chunk_id},
                )
            finally:
                self._queue.task_done()
