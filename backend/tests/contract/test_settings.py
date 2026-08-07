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


def test_demo_passwords_are_redacted() -> None:
    settings = Settings(
        environment="test",
        demo_elder_password="private-elder-password",
        demo_guardian_password="private-guardian-password",
        _env_file=None,
    )

    assert "private-elder-password" not in repr(settings)
    assert "private-guardian-password" not in repr(settings)


def test_auth_session_ttl_rejects_zero_hours() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Settings(environment="test", auth_session_ttl_hours=0, _env_file=None)


def test_streaming_asr_hotword_corrections_are_parsed_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "APP_STREAMING_ASR_HOTWORD_CORRECTIONS",
        '{"安全帐户":"安全账户"}',
    )

    settings = Settings(environment="test", _env_file=None)

    assert settings.streaming_asr_hotword_corrections == {"安全帐户": "安全账户"}


def test_ys7_alarm_poll_interval_rejects_over_aggressive_polling() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 5"):
        Settings(
            environment="test",
            ys7_alarm_poll_interval_seconds=4,
            _env_file=None,
        )
