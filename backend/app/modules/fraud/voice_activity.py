from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import webrtcvad

SAMPLE_RATE = 16_000
FRAME_MS = 20
SPEECH_START_MS = 200
SILENCE_END_MS = 700
PRE_ROLL_MS = 300
POST_ROLL_MS = 300
MAX_SEGMENT_MS = 10_000
FORCED_OVERLAP_MS = 1_000


@dataclass(frozen=True, slots=True)
class VoiceSegment:
    start_offset_ms: int
    pcm: bytes
    continues: bool = False


class VoiceActivitySegmenter:
    """Segments a continuous PCM stream on speech pauses with bounded latency."""

    def __init__(
        self,
        *,
        vad_mode: int = 2,
        voice_detector: Callable[[bytes], bool] | None = None,
    ) -> None:
        self._frame_bytes = SAMPLE_RATE * 2 * FRAME_MS // 1_000
        self._speech_start_frames = SPEECH_START_MS // FRAME_MS
        self._silence_end_frames = SILENCE_END_MS // FRAME_MS
        self._pre_roll_frames = PRE_ROLL_MS // FRAME_MS
        self._post_roll_frames = POST_ROLL_MS // FRAME_MS
        self._max_segment_frames = MAX_SEGMENT_MS // FRAME_MS
        self._overlap_frames = FORCED_OVERLAP_MS // FRAME_MS
        vad = webrtcvad.Vad(vad_mode)
        self._voice_detector = voice_detector or (
            lambda frame: bool(vad.is_speech(frame, SAMPLE_RATE))
        )
        self._pre_roll: deque[tuple[int, bytes]] = deque(
            maxlen=self._pre_roll_frames + self._speech_start_frames
        )
        self._active_frames: list[bytes] = []
        self._segment_start_frame = 0
        self._frame_index = 0
        self._candidate_voice_frames = 0
        self._trailing_silence_frames = 0
        self._has_new_voice = False

    def consume(self, frame: bytes) -> list[VoiceSegment]:
        if len(frame) != self._frame_bytes:
            raise ValueError(f"VAD frame must contain exactly {self._frame_bytes} bytes")

        frame_index = self._frame_index
        self._frame_index += 1
        voiced = self._voice_detector(frame)

        if not self._active_frames:
            self._pre_roll.append((frame_index, frame))
            self._candidate_voice_frames = self._candidate_voice_frames + 1 if voiced else 0
            if self._candidate_voice_frames < self._speech_start_frames:
                return []

            buffered = list(self._pre_roll)
            self._segment_start_frame = buffered[0][0]
            self._active_frames = [item[1] for item in buffered]
            self._pre_roll.clear()
            self._trailing_silence_frames = 0
            self._has_new_voice = True
            return self._cut_at_limit()

        self._active_frames.append(frame)
        if voiced:
            self._trailing_silence_frames = 0
            self._has_new_voice = True
        else:
            self._trailing_silence_frames += 1

        if len(self._active_frames) >= self._max_segment_frames:
            return self._cut_at_limit()
        if self._trailing_silence_frames >= self._silence_end_frames:
            segment = self._finish_active(frame_index)
            return [segment] if segment is not None else []
        return []

    def flush(self) -> list[VoiceSegment]:
        if not self._active_frames:
            return []
        segment = self._build_segment(self._trim_trailing_silence())
        self._reset_active()
        return [segment] if segment is not None else []

    def _cut_at_limit(self) -> list[VoiceSegment]:
        if len(self._active_frames) < self._max_segment_frames:
            return []
        completed = self._active_frames[: self._max_segment_frames]
        segment = self._build_segment(completed, continues=True)
        self._active_frames = completed[-self._overlap_frames :]
        self._segment_start_frame += self._max_segment_frames - self._overlap_frames
        self._trailing_silence_frames = min(
            self._trailing_silence_frames,
            self._overlap_frames,
        )
        self._has_new_voice = False
        return [segment] if segment is not None else []

    def _finish_active(self, frame_index: int) -> VoiceSegment | None:
        segment = self._build_segment(self._trim_trailing_silence())
        tail = self._active_frames[-self._pre_roll_frames :]
        tail_start = frame_index - len(tail) + 1
        self._reset_active()
        self._pre_roll.extend((tail_start + offset, frame) for offset, frame in enumerate(tail))
        return segment

    def _trim_trailing_silence(self) -> list[bytes]:
        trim_count = max(0, self._trailing_silence_frames - self._post_roll_frames)
        if trim_count == 0:
            return list(self._active_frames)
        return self._active_frames[:-trim_count]

    def _build_segment(
        self,
        frames: list[bytes],
        *,
        continues: bool = False,
    ) -> VoiceSegment | None:
        if not frames or not self._has_new_voice:
            return None
        return VoiceSegment(
            start_offset_ms=self._segment_start_frame * FRAME_MS,
            pcm=b"".join(frames),
            continues=continues,
        )

    @property
    def speech_active(self) -> bool:
        return bool(self._active_frames)

    @property
    def active_start_offset_ms(self) -> int:
        return self._segment_start_frame * FRAME_MS

    @property
    def active_pcm(self) -> bytes:
        return b"".join(self._active_frames)

    def _reset_active(self) -> None:
        self._active_frames = []
        self._candidate_voice_frames = 0
        self._trailing_silence_frames = 0
        self._has_new_voice = False
