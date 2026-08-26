from unittest.mock import patch

from radar_module.acquisition.ti_reader import ReconnectingTiOfficialOutputAdapter


class FakeDelegate:
    def __init__(self, *, payload=None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def read_decoded(self):
        if self.error is not None:
            raise self.error
        return self.payload

    def stop(self) -> None:
        self.stopped += 1


def test_real_adapter_reconnects_inside_one_logical_worker() -> None:
    first = FakeDelegate(error=RuntimeError("serial disconnected"))
    second = FakeDelegate(payload={"timestamp": "2026-08-26T10:00:00+08:00"})
    delegates = iter((first, second))

    with patch(
        "radar_module.acquisition.ti_reader.TiOfficialOutputAdapter",
        side_effect=lambda **_: next(delegates),
    ):
        adapter = ReconnectingTiOfficialOutputAdapter(
            command=("bridge.exe",),
            reconnect_seconds=0.001,
        )
        adapter.start()
        payload = adapter.read_decoded()
        adapter.stop()

    assert payload == {"timestamp": "2026-08-26T10:00:00+08:00"}
    assert first.started == 1
    assert first.stopped == 1
    assert second.started == 1
    assert second.stopped == 1
