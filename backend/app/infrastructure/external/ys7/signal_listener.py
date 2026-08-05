import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from app.infrastructure.event_deduplicator import EventDeduplicator
from app.infrastructure.event_queue import QueuedYs7Signal, Ys7EventQueue
from app.infrastructure.external.ys7.event_parser import Ys7EventParser
from app.infrastructure.raw_signal_store import RawSignalStore


@dataclass(frozen=True, slots=True)
class SignalReceipt:
    status: Literal["accepted", "duplicate"]
    source_event_id: str
    raw_event_ref: str | None


class Ys7SignalListener:
    """Transport-neutral entry point for webhook or future SDK callbacks."""

    def __init__(
        self,
        *,
        parser: Ys7EventParser,
        deduplicator: EventDeduplicator,
        raw_store: RawSignalStore,
        event_queue: Ys7EventQueue,
    ) -> None:
        self._parser = parser
        self._deduplicator = deduplicator
        self._raw_store = raw_store
        self._event_queue = event_queue

    async def receive(self, raw_payload: dict[str, object]) -> SignalReceipt:
        signal = self._parser.parse(raw_payload, received_at=datetime.now(UTC))
        dedup_key = self._deduplicator.key_for(signal, raw_payload)
        if not self._deduplicator.reserve(dedup_key):
            return SignalReceipt(
                status="duplicate",
                source_event_id=signal.source_event_id,
                raw_event_ref=None,
            )
        try:
            raw_event_ref = await asyncio.to_thread(
                self._raw_store.persist,
                dedup_key=dedup_key,
                signal=signal,
                raw_payload=raw_payload,
            )
            self._event_queue.publish_nowait(
                QueuedYs7Signal(
                    signal=signal,
                    dedup_key=dedup_key,
                    raw_event_ref=raw_event_ref,
                )
            )
        except Exception:
            self._deduplicator.release(dedup_key)
            raise
        return SignalReceipt(
            status="accepted",
            source_event_id=signal.source_event_id,
            raw_event_ref=raw_event_ref,
        )
