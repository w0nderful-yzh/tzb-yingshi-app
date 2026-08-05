import json
import threading
from hashlib import sha256

from app.infrastructure.external.ys7.models import Ys7Signal


class EventDeduplicator:
    def __init__(self) -> None:
        self._keys: set[str] = set()
        self._lock = threading.Lock()

    def key_for(self, signal: Ys7Signal, raw_payload: dict[str, object]) -> str:
        if signal.message_id:
            return f"message:{signal.message_id}"
        if signal.request_id:
            return f"request:{signal.request_id}"
        if signal.source_event_id:
            return f"event:{signal.source_event_id}"
        canonical = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True)
        return f"hash:{sha256(canonical.encode('utf-8')).hexdigest()}"

    def reserve(self, key: str) -> bool:
        with self._lock:
            if key in self._keys:
                return False
            self._keys.add(key)
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._keys.discard(key)
