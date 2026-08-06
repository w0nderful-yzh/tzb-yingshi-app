from app.modules.fraud.voice_activity import FRAME_MS, SAMPLE_RATE, VoiceActivitySegmenter

FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1_000
SPEECH_FRAME = b"\x01\x00" * FRAME_SAMPLES
SILENCE_FRAME = b"\x00\x00" * FRAME_SAMPLES


def _segmenter() -> VoiceActivitySegmenter:
    return VoiceActivitySegmenter(voice_detector=lambda frame: any(frame))


def _duration_ms(pcm: bytes) -> int:
    return len(pcm) * 1_000 // (SAMPLE_RATE * 2)


def test_vad_ends_segment_on_natural_pause_with_pre_and_post_roll() -> None:
    segmenter = _segmenter()
    segments = []

    for frame in [SILENCE_FRAME] * 15 + [SPEECH_FRAME] * 25 + [SILENCE_FRAME] * 35:
        segments.extend(segmenter.consume(frame))

    assert len(segments) == 1
    assert segments[0].start_offset_ms == 0
    assert _duration_ms(segments[0].pcm) == 1_100


def test_vad_ignores_audio_shorter_than_speech_start_threshold() -> None:
    segmenter = _segmenter()
    segments = []

    for frame in [SPEECH_FRAME] * 9 + [SILENCE_FRAME] * 35:
        segments.extend(segmenter.consume(frame))
    segments.extend(segmenter.flush())

    assert segments == []


def test_vad_forced_split_uses_ten_second_limit_and_one_second_overlap() -> None:
    segmenter = _segmenter()
    segments = []

    for frame in [SPEECH_FRAME] * 600:
        segments.extend(segmenter.consume(frame))
    segments.extend(segmenter.flush())

    assert [segment.start_offset_ms for segment in segments] == [0, 9_000]
    assert [_duration_ms(segment.pcm) for segment in segments] == [10_000, 3_000]
