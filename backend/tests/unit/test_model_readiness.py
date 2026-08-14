import pytest

from app.modules.fraud.model_readiness import (
    DISABLED,
    FAILED,
    READY,
    WARMING_UP,
    ModelReadinessTracker,
    warmup_models,
)


@pytest.fixture(autouse=True)
def _reset_classifier_cache():
    from app.modules.fraud import text_classifier

    text_classifier.get_default_classifier.cache_clear()
    yield
    text_classifier.get_default_classifier.cache_clear()


def test_tracker_transitions_to_ready() -> None:
    tracker = ModelReadinessTracker()
    assert tracker.snapshot().models_ready == DISABLED
    tracker.mark_warming("classifier")
    tracker.mark_ready("classifier")
    snapshot = tracker.snapshot()
    assert snapshot.classifier_ready
    assert snapshot.warmup_error is None


def test_tracker_failed_overrides_ready() -> None:
    tracker = ModelReadinessTracker()
    tracker.mark_warming("sensevoice")
    tracker.mark_ready("sensevoice")
    tracker.mark_warming("paraformer")
    tracker.mark_failed("paraformer", "cuda oom")
    snapshot = tracker.snapshot()
    assert snapshot.models_ready == FAILED
    assert snapshot.warmup_error == "cuda oom"


def test_tracker_warming_aggregate() -> None:
    tracker = ModelReadinessTracker()
    tracker.mark_warming("sensevoice")
    tracker.mark_disabled("paraformer")
    assert tracker.snapshot().models_ready == WARMING_UP


def test_all_disabled_is_disabled() -> None:
    tracker = ModelReadinessTracker()
    tracker.mark_disabled("classifier")
    tracker.mark_disabled("sensevoice")
    tracker.mark_disabled("paraformer")
    snapshot = tracker.snapshot()
    assert snapshot.models_ready == DISABLED
    assert not snapshot.classifier_ready


@pytest.mark.asyncio
async def test_warmup_models_marks_ready_and_warms_recognizers() -> None:
    tracker = ModelReadinessTracker()
    warmed: list[str] = []

    class FakeSenseVoice:
        def warmup(self) -> None:
            warmed.append("sensevoice")

    class FakeParaformer:
        def warmup(self) -> None:
            warmed.append("paraformer")

    await warmup_models(
        readiness=tracker,
        classifier_warmup_enabled=True,
        sensevoice_warmup_enabled=True,
        streaming_warmup_enabled=True,
        classifier_loader=lambda: None,
        sensevoice_recognizer=FakeSenseVoice(),
        streaming_recognizer=FakeParaformer(),
    )
    snapshot = tracker.snapshot()
    assert snapshot.classifier_ready
    assert snapshot.models_ready == READY
    assert sorted(warmed) == ["paraformer", "sensevoice"]


@pytest.mark.asyncio
async def test_warmup_failure_is_recorded_not_raised() -> None:
    tracker = ModelReadinessTracker()

    class BrokenRecognizer:
        def warmup(self) -> None:
            raise RuntimeError("model download failed")

    await warmup_models(
        readiness=tracker,
        classifier_warmup_enabled=False,
        sensevoice_warmup_enabled=True,
        streaming_warmup_enabled=False,
        classifier_loader=lambda: None,
        sensevoice_recognizer=BrokenRecognizer(),
        streaming_recognizer=None,
    )
    snapshot = tracker.snapshot()
    assert snapshot.models_ready == FAILED
    assert snapshot.warmup_error == "model download failed"
    assert not snapshot.classifier_ready


@pytest.mark.asyncio
async def test_warmup_models_honors_disabled_flags() -> None:
    tracker = ModelReadinessTracker()
    await warmup_models(
        readiness=tracker,
        classifier_warmup_enabled=False,
        sensevoice_warmup_enabled=False,
        streaming_warmup_enabled=False,
        classifier_loader=lambda: pytest.fail("should not load"),
        sensevoice_recognizer=None,
        streaming_recognizer=None,
    )
    snapshot = tracker.snapshot()
    assert snapshot.models_ready == DISABLED


@pytest.mark.asyncio
async def test_warmup_models_warms_classifier_into_lru_cache() -> None:
    from app.modules.fraud.text_classifier import get_default_classifier

    tracker = ModelReadinessTracker()
    await warmup_models(
        readiness=tracker,
        classifier_warmup_enabled=True,
        sensevoice_warmup_enabled=False,
        streaming_warmup_enabled=False,
        classifier_loader=get_default_classifier,
        sensevoice_recognizer=None,
        streaming_recognizer=None,
    )
    snapshot = tracker.snapshot()
    assert snapshot.classifier_ready
    assert get_default_classifier() is get_default_classifier()
