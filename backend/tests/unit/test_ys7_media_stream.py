import pytest

from app.infrastructure.external.ys7 import media_stream


def test_resolve_ffmpeg_executable_prefers_system_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_stream.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        media_stream.imageio_ffmpeg,
        "get_ffmpeg_exe",
        lambda: pytest.fail("bundled FFmpeg should not be selected"),
    )

    assert media_stream._resolve_ffmpeg_executable() == "/usr/bin/ffmpeg"


def test_resolve_ffmpeg_executable_falls_back_to_bundled_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_stream.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        media_stream.imageio_ffmpeg,
        "get_ffmpeg_exe",
        lambda: "/bundled/ffmpeg",
    )

    assert media_stream._resolve_ffmpeg_executable() == "/bundled/ffmpeg"


def test_sanitize_ffmpeg_error_redacts_signed_stream_url_and_limits_output() -> None:
    secret = "very-sensitive-token"
    noisy_prefix = "x" * 2_000
    raw_error = (
        f"{noisy_prefix}\n"
        f"Error opening https://rtmp09open.ys7.com/live.flv?ev={secret}&expire=123\n"
        "Stream map '0:a:0' matches no streams\n"
    ).encode()

    detail = media_stream._sanitize_ffmpeg_error(raw_error)

    assert secret not in detail
    assert "https://rtmp09open.ys7.com" not in detail
    assert "<stream-url>" in detail
    assert "matches no streams" in detail
    assert len(detail) <= 1_000


def test_format_ffmpeg_failure_reports_missing_audio_track() -> None:
    detail = "Failed to set value '0:a:0' for option 'map': Invalid argument"

    assert (
        media_stream._format_ffmpeg_failure(234, detail)
        == "YS7 live stream does not contain an audio track"
    )
