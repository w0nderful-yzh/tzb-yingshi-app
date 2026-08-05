from app.infrastructure.event_queue import QueuedYs7Signal
from app.infrastructure.external.ys7.models import Ys7Box
from app.modules.fraud.schemas import VisualBoundingBox, VisualEvent


class Ys7EventAdapter:
    def adapt(self, queued: QueuedYs7Signal) -> VisualEvent:
        signal = queued.signal
        return VisualEvent(
            source_event_id=signal.source_event_id,
            message_id=signal.message_id,
            request_id=signal.request_id,
            device_id=signal.device_id,
            occurred_at=signal.occurred_at,
            received_at=signal.received_at,
            source="ys7",
            event_type=signal.event_type.value,
            confidence=signal.confidence,
            people_count=signal.people_count,
            boxes=[self._adapt_box(box) for box in signal.boxes],
            image_url=signal.image_url,
            raw_event_ref=queued.raw_event_ref,
        )

    def _adapt_box(self, box: Ys7Box) -> VisualBoundingBox:
        x1, y1, x2, y2 = box.coordinates
        return VisualBoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=box.label,
            confidence=box.confidence,
        )
