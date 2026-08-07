import asyncio
from collections.abc import AsyncIterator


class AppPcmRelaySource:
    """Receives decoded PCM from EZOpenSDK and exposes 20 ms frames to the media worker."""

    def __init__(self, *, device_id: str | None, queue_maxsize: int) -> None:
        self._device_id = device_id
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_maxsize)
        self.chunks_received = 0
        self.chunks_dropped = 0

    def push(self, *, device_id: str, pcm: bytes, sample_rate: int) -> None:
        if not self._device_id or device_id != self._device_id:
            raise ValueError("PCM relay device does not match the configured YS7 device")
        if sample_rate != 16_000:
            raise ValueError("PCM relay sample rate must be 16000 Hz")
        if not pcm or len(pcm) % 2 != 0:
            raise ValueError("PCM relay payload must contain signed 16-bit mono samples")
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.chunks_dropped += 1
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(pcm)
        self.chunks_received += 1

    async def stream(self, url: str, *, frame_ms: int) -> AsyncIterator[bytes]:
        del url
        bytes_per_frame = round(16_000 * 2 * frame_ms / 1_000)
        buffered = bytearray()
        while True:
            chunk = await self._queue.get()
            try:
                buffered.extend(chunk)
                while len(buffered) >= bytes_per_frame:
                    yield bytes(buffered[:bytes_per_frame])
                    del buffered[:bytes_per_frame]
            finally:
                self._queue.task_done()

    async def close(self) -> None:
        return
