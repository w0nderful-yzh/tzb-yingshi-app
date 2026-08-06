from typing import Any

from app.infrastructure.external.sensevoice.streaming import ParaformerStreamingRecognizer


class FakeParaformerModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append(kwargs)
        text = "安全帐户" if len(self.calls) == 1 else "马上转账"
        return [{"text": text}]


def test_streaming_recognizer_accumulates_chunks_and_corrects_hotwords() -> None:
    model = FakeParaformerModel()
    recognizer = ParaformerStreamingRecognizer(
        model_name="paraformer-zh-streaming",
        device="cpu",
        hotwords="安全账户 转账",
        hotword_corrections={"安全帐户": "安全账户"},
    )
    recognizer._model = model  # type: ignore[attr-defined]
    session = recognizer.create_session()
    pcm = b"\x00\x00" * (16_000 * 600 // 1_000)

    partial = session.transcribe_pcm(pcm, is_final=False)
    final = session.transcribe_pcm(pcm, is_final=True)

    assert partial == "安全账户"
    assert final == "安全账户马上转账"
    assert model.calls[0]["chunk_size"] == [0, 10, 5]
    assert model.calls[0]["hotword"] == "安全账户 转账"
    assert model.calls[1]["is_final"] is True
