from app.modules.auth.passwords import hash_password, verify_password


def test_password_hash_round_trip_and_random_salt() -> None:
    first = hash_password("guardian123")
    second = hash_password("guardian123")

    assert first != second
    assert "guardian123" not in first
    assert verify_password("guardian123", first) is True
    assert verify_password("wrong-password", first) is False


def test_malformed_password_hash_is_rejected() -> None:
    assert verify_password("guardian123", "not-a-password-hash") is False
