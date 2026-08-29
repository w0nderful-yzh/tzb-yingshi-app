import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.modules.psychology.cognitive.result_store import CognitiveResultStore
from app.modules.psychology.cognitive.schemas import CognitiveInferenceJob
from app.modules.psychology.cognitive.worker import (
    CognitiveWorker,
    MmseInferenceResult,
)

UTC_TZ = timezone.utc  # noqa: UP017 - exercise Python 3.10-compatible code.


class _FakeRunner:
    def __init__(self, score: float) -> None:
        self._score = score
        self.paths_seen: list[Path] = []

    def infer(self, wav_path: Path) -> MmseInferenceResult:
        assert wav_path.is_file()
        self.paths_seen.append(wav_path)
        return MmseInferenceResult(
            estimated_mmse_score=self._score,
            audio_window_count=11,
        )


def _publish_job(store: CognitiveResultStore, *, assessment_id: str) -> None:
    now = datetime.now(UTC_TZ)
    output = store.inbox_dir / "source.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 16_000)
    wav_bytes = output.read_bytes()
    output.unlink()
    store.publish_job(
        CognitiveInferenceJob(
            assessment_id=assessment_id,
            subject_key="elder-001",
            session_id="session-001",
            device_id="camera-001",
            window_started_at=now - timedelta(seconds=120),
            window_ended_at=now,
            effective_speech_seconds=120.0,
            created_at=now,
            expires_at=now + timedelta(hours=1),
        ),
        wav_bytes,
    )


def test_worker_completes_snapshot_and_deletes_temporary_wav(tmp_path) -> None:
    store = CognitiveResultStore(tmp_path)
    _publish_job(store, assessment_id="cog-valid")
    runner = _FakeRunner(24.75)
    worker = CognitiveWorker(runtime_root=tmp_path, runner=runner)

    assert worker.process_next() is True

    snapshot = store.read_latest("elder-001")
    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.estimated_mmse_score == 24.75
    assert snapshot.audio_window_count == 11
    assert not list(store.inbox_dir.glob("*.wav"))
    assert not list(store.processing_dir.glob("*.wav"))
    assert runner.paths_seen and not runner.paths_seen[0].exists()


def test_worker_rejects_out_of_range_score_without_clamping(tmp_path) -> None:
    store = CognitiveResultStore(tmp_path)
    _publish_job(store, assessment_id="cog-invalid")
    worker = CognitiveWorker(runtime_root=tmp_path, runner=_FakeRunner(31.25))

    assert worker.process_next() is True

    snapshot = store.read_latest("elder-001")
    assert snapshot is not None
    assert snapshot.status == "failed"
    assert snapshot.failure_code == "score_out_of_range"
    assert snapshot.estimated_mmse_score == 31.25
    assert store.read_latest_completed("elder-001") is None
    assert not list(store.processing_dir.iterdir())
