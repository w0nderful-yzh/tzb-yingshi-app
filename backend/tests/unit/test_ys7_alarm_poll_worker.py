from pathlib import Path
from typing import Any

import pytest

from app.infrastructure.event_deduplicator import EventDeduplicator
from app.infrastructure.event_queue import Ys7EventQueue
from app.infrastructure.external.ys7.alarm_mapper import Ys7AlarmMapper
from app.infrastructure.external.ys7.event_parser import Ys7EventParser
from app.infrastructure.external.ys7.signal_listener import Ys7SignalListener
from app.infrastructure.raw_signal_store import RawSignalStore
from app.workers.ys7_alarm_poll_worker import Ys7AlarmPollWorker


class StubAlarmProvider:
    async def list_device_alarms(
        self,
        *,
        device_serial: str,
        start_time_ms: int,
        end_time_ms: int,
        page_size: int,
    ) -> list[dict[str, Any]]:
        assert device_serial == "camera-01"
        assert start_time_ms < end_time_ms
        assert page_size == 50
        return [
            {
                "alarmId": "person-01",
                "alarmType": 10000,
                "alarmStart": end_time_ms,
            },
            {
                "alarmId": "motion-01",
                "alarmType": 10002,
            },
        ]


@pytest.mark.asyncio
async def test_poll_worker_ingests_supported_alarms_and_deduplicates(
    tmp_path: Path,
) -> None:
    queue = Ys7EventQueue(maxsize=10)
    listener = Ys7SignalListener(
        parser=Ys7EventParser(),
        deduplicator=EventDeduplicator(),
        raw_store=RawSignalStore(tmp_path),
        event_queue=queue,
    )
    worker = Ys7AlarmPollWorker(
        alarm_provider=StubAlarmProvider(),
        signal_listener=listener,
        mapper=Ys7AlarmMapper(default_device_serial="camera-01"),
        device_serial="camera-01",
        interval_seconds=15,
        lookback_seconds=120,
        page_size=50,
    )

    await worker.poll_once()
    await worker.poll_once()

    assert worker.polls_completed == 2
    assert worker.alarms_seen == 4
    assert worker.signals_accepted == 1
    assert worker.signals_duplicate == 1
    assert worker.alarms_ignored == 2
    assert worker.last_ignored_alarm_type == "10002"
    queued = await queue.get()
    assert queued.signal.event_type == "person_detected"
    assert len(list(tmp_path.rglob("*.json"))) == 1
