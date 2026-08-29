import wave
from datetime import datetime, timezone

import pytest

from app.core.config import Settings
from app.main import create_app
from app.modules.psychology.cognitive.collector import (
    FRAME_BYTES,
    CognitiveAudioCollector,
)
from app.modules.psychology.cognitive.result_store import CognitiveResultStore
from app.modules.psychology.cognitive.schemas import CognitiveAssessmentSnapshot

UTC_TZ = timezone.utc  # noqa: UP017 - exercise Python 3.10-compatible code.


@pytest.mark.asyncio
async def test_collector_publishes_16khz_mono_pcm16_job_without_asr(tmp_path) -> None:
    collector = CognitiveAudioCollector(
        runtime_root=tmp_path,
        enabled=True,
        queue_maxsize=2,
        min_speech_seconds=0.02,
        target_speech_seconds=0.04,
        max_session_seconds=1.0,
        cooldown_seconds=0.0,
        voice_detector=lambda _frame: True,
    )
    await collector.start()
    try:
        accepted = collector.push(
            subject_key="elder-001",
            device_id="camera-001",
            pcm=b"\x01\x00" * (FRAME_BYTES // 2 * 2),
            sample_rate=16_000,
        )
        await collector._queue.join()  # noqa: SLF001 - deterministic queue drain
    finally:
        await collector.stop()

    assert accepted is True
    store = CognitiveResultStore(tmp_path)
    manifests = list(store.inbox_dir.glob("*.json"))
    wav_files = list(store.inbox_dir.glob("*.wav"))
    assert len(manifests) == 1
    assert len(wav_files) == 1
    job = store.read_job(manifests[0])
    assert job.subject_key == "elder-001"
    assert job.effective_speech_seconds == pytest.approx(0.04)
    with wave.open(str(wav_files[0]), "rb") as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2


def test_processing_snapshot_preserves_latest_completed(tmp_path) -> None:
    store = CognitiveResultStore(tmp_path)
    now = datetime.now(UTC_TZ)
    completed = CognitiveAssessmentSnapshot(
        assessment_id="cog-completed",
        subject_key="elder-001",
        session_id="session-completed",
        status="completed",
        window_started_at=now,
        window_ended_at=now,
        effective_speech_seconds=120.0,
        estimated_mmse_score=24.5,
        audio_window_count=11,
        completed_at=now,
    )
    store.write_snapshot(completed)
    store.write_snapshot(
        CognitiveAssessmentSnapshot(
            assessment_id="cog-processing",
            subject_key="elder-001",
            session_id="session-processing",
            status="processing",
            window_started_at=now,
            effective_speech_seconds=0.0,
        )
    )

    latest = store.read_latest("elder-001")
    assert latest is not None
    assert latest.status == "processing"
    assert store.read_latest_completed("elder-001") == completed


def test_cognitive_collector_is_independent_of_asr_toggles(tmp_path) -> None:
    app = create_app(
        Settings(
            environment="test",
            cognitive_enabled=True,
            cognitive_runtime_dir=tmp_path,
            sensevoice_enabled=False,
            streaming_asr_enabled=False,
            _env_file=None,
        )
    )

    assert app.state.cognitive_collector.enabled is True
    assert app.state.settings.sensevoice_enabled is False
    assert app.state.settings.streaming_asr_enabled is False


def test_cognitive_collection_defaults_match_phase_one_contract() -> None:
    settings = Settings(environment="test", _env_file=None)

    assert settings.cognitive_min_speech_seconds == 60
    assert settings.cognitive_target_speech_seconds == 120
    assert settings.cognitive_max_session_seconds == 30 * 60
