import pytest

from app.infrastructure.external.ys7.pcm_relay import AppPcmRelaySource


@pytest.mark.asyncio
async def test_pcm_relay_splits_uploaded_pcm_into_requested_frames() -> None:
    relay = AppPcmRelaySource(device_id="camera-1", queue_maxsize=2)
    relay.push(device_id="camera-1", pcm=b"\x01\x00" * 640, sample_rate=16_000)
    stream = relay.stream("ignored", frame_ms=20)

    first = await anext(stream)
    second = await anext(stream)
    await stream.aclose()

    assert len(first) == 640
    assert len(second) == 640
    assert relay.chunks_received == 1


def test_pcm_relay_rejects_wrong_device_or_audio_format() -> None:
    relay = AppPcmRelaySource(device_id="camera-1", queue_maxsize=2)

    with pytest.raises(ValueError, match="configured YS7 device"):
        relay.push(device_id="camera-2", pcm=b"\x00\x00", sample_rate=16_000)
    with pytest.raises(ValueError, match="16000 Hz"):
        relay.push(device_id="camera-1", pcm=b"\x00\x00", sample_rate=8_000)
    with pytest.raises(ValueError, match="16-bit mono"):
        relay.push(device_id="camera-1", pcm=b"\x00", sample_rate=16_000)
