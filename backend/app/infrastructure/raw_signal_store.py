import json
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from app.infrastructure.external.ys7.models import Ys7Signal


class RawSignalStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def persist(
        self,
        *,
        dedup_key: str,
        signal: Ys7Signal,
        raw_payload: dict[str, object],
    ) -> str:
        day_directory = self._root / signal.received_at.date().isoformat()
        day_directory.mkdir(parents=True, exist_ok=True)
        filename = f"{sha256(dedup_key.encode('utf-8')).hexdigest()}.json"
        destination = day_directory / filename
        if destination.exists():
            return str(destination.relative_to(self._root))

        payload = {
            "dedup_key": dedup_key,
            "received_at": signal.received_at.isoformat(),
            "parsed": signal.model_dump(mode="json"),
            "raw": raw_payload,
        }
        temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return str(destination.relative_to(self._root))

    def path_for(self, reference: str) -> Path:
        return self._root / reference
