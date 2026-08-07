import asyncio
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Protocol

import imageio_ffmpeg


class Ys7MediaStreamError(RuntimeError):
    pass


class PcmStreamSource(Protocol):
    def stream(self, url: str, *, frame_ms: int) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


_FFMPEG_ERROR_TAIL_BYTES = 8 * 1024
_FFMPEG_ERROR_DETAIL_CHARS = 1_000
_FFMPEG_READ_TIMEOUT_US = 15_000_000
_STREAM_URL_PATTERN = re.compile(
    r"(?i)\b(?:https?|rtmps?|rtsp|ezopen)://\S+",
)


def _resolve_ffmpeg_executable() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


async def _read_bounded_tail(
    reader: asyncio.StreamReader,
    *,
    max_bytes: int = _FFMPEG_ERROR_TAIL_BYTES,
) -> bytes:
    tail = bytearray()
    while chunk := await reader.read(2_048):
        tail.extend(chunk)
        if len(tail) > max_bytes:
            del tail[:-max_bytes]
    return bytes(tail)


def _sanitize_ffmpeg_error(raw_error: bytes) -> str:
    text = raw_error.decode(errors="replace")
    text = _STREAM_URL_PATTERN.sub("<stream-url>", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-4:])[-_FFMPEG_ERROR_DETAIL_CHARS:]


def _format_ffmpeg_failure(exit_code: int, error_detail: str) -> str:
    no_audio_markers = (
        "matches no streams",
        "Failed to set value '0:a:0'",
    )
    if any(marker in error_detail for marker in no_audio_markers):
        return "YS7 live stream does not contain an audio track"
    message = f"FFmpeg media stream exited with code {exit_code}"
    return f"{message}: {error_detail}" if error_detail else message


class FfmpegPcmStreamSource:
    def __init__(self, *, executable: str | None = None) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._executable = executable

    async def stream(self, url: str, *, frame_ms: int) -> AsyncIterator[bytes]:
        bytes_per_frame = round(16_000 * 2 * frame_ms / 1000)
        executable = self._executable or _resolve_ffmpeg_executable()
        self._process = await asyncio.create_subprocess_exec(
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rw_timeout",
            str(_FFMPEG_READ_TIMEOUT_US),
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
            stderr=asyncio.subprocess.PIPE,
        )
        stdout = self._process.stdout
        stderr = self._process.stderr
        if stdout is None or stderr is None:
            await self.close()
            raise Ys7MediaStreamError("FFmpeg audio output is unavailable")
        stderr_task = asyncio.create_task(_read_bounded_tail(stderr))
        try:
            while True:
                try:
                    chunk = await stdout.readexactly(bytes_per_frame)
                except asyncio.IncompleteReadError:
                    break
                yield chunk
            exit_code = await self._process.wait()
            error_detail = _sanitize_ffmpeg_error(await stderr_task)
            if exit_code != 0:
                raise Ys7MediaStreamError(_format_ffmpeg_failure(exit_code, error_detail))
        finally:
            await self.close()
            if not stderr_task.done():
                stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task

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
