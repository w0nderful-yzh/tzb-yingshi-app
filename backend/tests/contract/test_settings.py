import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_ys7_live_quality_is_parsed_from_environment_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_YS7_LIVE_QUALITY", "2")

    settings = Settings(environment="test", _env_file=None)

    assert settings.ys7_live_quality == 2


def test_ys7_live_quality_rejects_unsupported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_YS7_LIVE_QUALITY", "4")

    with pytest.raises(ValidationError, match="less than or equal to 3"):
        Settings(environment="test", _env_file=None)


def test_fraud_llm_api_key_is_redacted() -> None:
    settings = Settings(
        environment="test",
        fraud_llm_api_key="private-llm-key",
        _env_file=None,
    )

    assert "private-llm-key" not in repr(settings)


def test_streaming_asr_hotword_corrections_are_parsed_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "APP_STREAMING_ASR_HOTWORD_CORRECTIONS",
        '{"安全帐户":"安全账户"}',
    )

    settings = Settings(environment="test", _env_file=None)

    assert settings.streaming_asr_hotword_corrections == {"安全帐户": "安全账户"}
