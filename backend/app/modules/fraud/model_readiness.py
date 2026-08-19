"""Model readiness tracking for the fraud pipeline.

Tracks each warmable model (classifier, SenseVoice, Paraformer) through the
lifecycle DISABLED → WARMING_UP → READY | FAILED. The overall media state is
derived from the per-model states so the media worker and status endpoint can
tell operators whether the pipeline is safe to accept real audio.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

ModelState = str
DISABLED = "DISABLED"
WARMING_UP = "WARMING_UP"
READY = "READY"
FAILED = "FAILED"


@dataclass(slots=True)
class _ModelReadiness:
    state: ModelState = DISABLED
    error: str | None = None


@dataclass(slots=True)
class ModelReadinessSnapshot:
    classifier: ModelState
    classifier_error: str | None
    sensevoice: ModelState
    sensevoice_error: str | None
    paraformer: ModelState
    paraformer_error: str | None

    @property
    def models_ready(self) -> ModelState:
        """Aggregate state of the ASR models that gate real audio ingestion."""
        states = {state for state in (self.sensevoice, self.paraformer) if state != DISABLED}
        if FAILED in states:
            return FAILED
        if WARMING_UP in states:
            return WARMING_UP
        if not states:
            return DISABLED
        return READY

    @property
    def classifier_ready(self) -> bool:
        return self.classifier == READY

    @property
    def warmup_error(self) -> str | None:
        """Most recent desensitized warmup error across all models."""
        errors = [
            error
            for error in (self.classifier_error, self.sensevoice_error, self.paraformer_error)
            if error is not None
        ]
        return errors[-1] if errors else None


class ModelReadinessTracker:
    """Thread-safe in-process state for the currently configured models."""

    def __init__(self) -> None:
        self._models: dict[str, _ModelReadiness] = {
            "classifier": _ModelReadiness(),
            "sensevoice": _ModelReadiness(),
            "paraformer": _ModelReadiness(),
        }

    def mark_disabled(self, name: str) -> None:
        self._models[name].state = DISABLED
        self._models[name].error = None

    def mark_warming(self, name: str) -> None:
        self._models[name].state = WARMING_UP
        self._models[name].error = None

    def mark_ready(self, name: str) -> None:
        self._models[name].state = READY
        self._models[name].error = None

    def mark_failed(self, name: str, error: str) -> None:
        self._models[name].state = FAILED
        self._models[name].error = error

    def snapshot(self) -> ModelReadinessSnapshot:
        return ModelReadinessSnapshot(
            classifier=self._models["classifier"].state,
            classifier_error=self._models["classifier"].error,
            sensevoice=self._models["sensevoice"].state,
            sensevoice_error=self._models["sensevoice"].error,
            paraformer=self._models["paraformer"].state,
            paraformer_error=self._models["paraformer"].error,
        )


async def warmup_models(
    *,
    readiness: ModelReadinessTracker,
    classifier_warmup_enabled: bool,
    sensevoice_warmup_enabled: bool,
    streaming_warmup_enabled: bool,
    classifier_loader: Callable[[], Any],
    sensevoice_recognizer: Any,
    streaming_recognizer: Any,
) -> None:
    """Eagerly warm up configured models without blocking app startup.

    A failed warmup is recorded (FAILED) and logged but never re-raised:
    health checks and non-fraud endpoints must keep working, and each
    recognizer still lazily loads on first real request.
    """

    async def _run(name: str, worker: Callable[[], None]) -> None:
        readiness.mark_warming(name)
        try:
            await asyncio.to_thread(worker)
        except Exception as exc:  # pragma: no cover - depends on model stack
            readiness.mark_failed(name, str(exc))
            logger.warning(
                "fraud model warmup failed; falling back to lazy loading",
                extra={"model": name},
            )
        else:
            readiness.mark_ready(name)
            logger.info("fraud model warmed up", extra={"model": name})

    if classifier_warmup_enabled:
        await _run("classifier", lambda: classifier_loader())
    else:
        readiness.mark_disabled("classifier")

    if sensevoice_warmup_enabled:
        warmup = getattr(sensevoice_recognizer, "warmup", None)
        if callable(warmup):
            await _run("sensevoice", warmup)
        else:
            readiness.mark_disabled("sensevoice")
    else:
        readiness.mark_disabled("sensevoice")

    if streaming_warmup_enabled:
        warmup = getattr(streaming_recognizer, "warmup", None)
        if callable(warmup):
            await _run("paraformer", warmup)
        else:
            readiness.mark_disabled("paraformer")
    else:
        readiness.mark_disabled("paraformer")
