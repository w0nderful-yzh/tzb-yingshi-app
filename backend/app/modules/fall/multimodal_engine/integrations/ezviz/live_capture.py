from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import urlsplit


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class EzvizLiveCaptureError(RuntimeError):
    pass


class EzvizLiveCapture:
    """Record a short standard EZVIZ stream clip for the file-based model."""

    def __init__(
        self,
        *,
        python_executable: Path,
        timeout_seconds: int = 60,
        capture_script: Path | None = None,
        process_runner: ProcessRunner = subprocess.run,
    ) -> None:
        self.python_executable = Path(python_executable).resolve()
        self.timeout_seconds = timeout_seconds
        self.capture_script = (
            Path(capture_script).resolve()
            if capture_script is not None
            else Path(__file__).with_name("capture_cli.py").resolve()
        )
        self._process_runner = process_runner

    @property
    def ready(self) -> bool:
        return self.python_executable.is_file() and self.capture_script.is_file()

    def capture(
        self,
        stream_url: str,
        output_path: Path,
        *,
        duration_seconds: int,
    ) -> dict[str, Any]:
        scheme = urlsplit(stream_url).scheme.lower()
        if scheme not in {"http", "https", "rtmp", "rtmps"}:
            raise EzvizLiveCaptureError(
                "live inference requires a standard HLS/RTMP/FLV URL; EZOPEN is browser-only"
            )
        if not self.ready:
            raise EzvizLiveCaptureError("live capture runtime is not ready")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python_executable),
            str(self.capture_script),
            "--output",
            str(output_path),
            "--duration",
            str(duration_seconds),
        ]
        try:
            completed = self._process_runner(
                command,
                input=f"{stream_url}\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(self.timeout_seconds, duration_seconds + 20),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise EzvizLiveCaptureError(
                f"live capture timed out after {max(self.timeout_seconds, duration_seconds + 20)}s"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown capture error")[-1500:]
            detail = detail.replace(stream_url, "[stream-url-redacted]")
            raise EzvizLiveCaptureError(f"live capture failed: {detail}")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise EzvizLiveCaptureError("live capture did not create a video clip")

        try:
            report_line = next(
                line for line in reversed(completed.stdout.splitlines()) if line.strip()
            )
            return json.loads(report_line)
        except (StopIteration, ValueError, json.JSONDecodeError) as exc:
            raise EzvizLiveCaptureError("live capture returned an invalid report") from exc


__all__ = ["EzvizLiveCapture", "EzvizLiveCaptureError"]
