import asyncio
import logging
from contextlib import suppress

from app.modules.fraud.llm import (
    FraudLlmJudge,
    FraudLlmReviewQueue,
    review_to_evidence,
)
from app.modules.fraud.service import FraudSessionService

logger = logging.getLogger(__name__)


class FraudLlmReviewWorker:
    def __init__(
        self,
        *,
        queue: FraudLlmReviewQueue,
        judge: FraudLlmJudge,
        fraud_session_service: FraudSessionService,
        timeout_seconds: float,
    ) -> None:
        self._queue = queue
        self._judge = judge
        self._fraud_session_service = fraud_session_service
        self._timeout_seconds = timeout_seconds
        self._task: asyncio.Task[None] | None = None
        self.reviews_processed = 0
        self.reviews_failed = 0
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def queue_depth(self) -> int:
        return self._queue.depth

    @property
    def model_name(self) -> str:
        return self._judge.model_name

    async def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self._run(), name="fraud-llm-review")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            request = await self._queue.get()
            try:
                review = await asyncio.wait_for(
                    self._judge.review(request),
                    timeout=self._timeout_seconds,
                )
                evidence = review_to_evidence(
                    request,
                    review,
                    model_name=self._judge.model_name,
                )
                await self._fraud_session_service.apply_llm_evidence(
                    device_id=request.device_id,
                    session_id=request.session_id,
                    evidence=evidence,
                )
                self.reviews_processed += 1
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.reviews_failed += 1
                self.last_error = str(exc)
                logger.warning(
                    "Fraud LLM review failed; local state machine remains active",
                    extra={"review_id": request.review_id},
                )
            finally:
                self._queue.task_done()
