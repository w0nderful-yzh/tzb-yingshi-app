import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Protocol

import imageio_ffmpeg


class Ys7MediaStreamError(RuntimeError):
    pass


class PcmStreamSource(Protocol):
    def stream(self, url: str, *, chunk_ms: int) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


class FfmpegPcmStreamSource:
    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None

    async def stream(self, url: str, *, chunk_ms: int) -> AsyncIterator[bytes]:
        bytes_per_chunk = round(16_000 * 2 * chunk_ms / 1000)
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        self._process = await asyncio.create_subprocess_exec(
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            url,
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if self._process.stdout is None:
            raise Ys7MediaStreamError("FFmpeg audio output is unavailable")
        try:
            while True:
                try:
                    chunk = await self._process.stdout.readexactly(bytes_per_chunk)
                except asyncio.IncompleteReadError:
                    break
                yield chunk
            exit_code = await self._process.wait()
            if exit_code != 0:
                raise Ys7MediaStreamError(f"FFmpeg media stream exited with code {exit_code}")
        finally:
            await self.close()

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=3.0)
        if process.returncode is None:
            process.kill()
            await process.wait()
