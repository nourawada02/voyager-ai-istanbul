"""Hermetic tests for secret redaction/isolation (Checkpoint Phase 4
C.0, Step 8: "secret redaction").

Zero-secret-literal policy (Checkpoint Phase 4 C.2): every credential
*value* below is generated fresh at test-run time via
`_ephemeral_test_secret()` -- never a fixed key-shaped string literal
committed to this file. The field names (`api_key`, `Authorization`,
`Bearer`, `secret`, `password`, `token`) remain as literals because
`redact_secrets()`'s own pattern matching is keyed on exactly those
names -- that is what this file is testing."""

from __future__ import annotations

import secrets

from providers.redaction import SecretString, redact_secrets


def _ephemeral_test_secret() -> str:
    """A fresh, random, per-run test value -- never a fixed key-shaped
    literal committed to the repository (zero-secret-literal policy)."""
    return f"ephemeral-test-token-{secrets.token_hex(16)}"


def test_api_key_query_param_is_redacted():
    secret_value = _ephemeral_test_secret()
    log_line = f"GET https://api.example.com/v1/weather?api_key={secret_value}&location=IST"
    redacted = redact_secrets(log_line)
    assert secret_value not in redacted
    assert "***REDACTED***" in redacted


def test_authorization_bearer_header_is_redacted():
    secret_value = _ephemeral_test_secret()
    log_line = f'headers={{"Authorization": "Bearer {secret_value}"}}'
    redacted = redact_secrets(log_line)
    assert secret_value not in redacted


def test_secret_and_password_and_token_keywords_are_redacted():
    for keyword in ("secret", "password", "token"):
        secret_value = _ephemeral_test_secret()
        line = f"{keyword}={secret_value}"
        redacted = redact_secrets(line)
        assert secret_value not in redacted
        assert "REDACTED" in redacted


def test_non_secret_text_is_left_unchanged():
    line = "GET https://api.example.com/v1/weather?location=Istanbul&units=metric"
    assert redact_secrets(line) == line


def test_secret_string_never_reveals_value_via_str_or_repr():
    secret_value = _ephemeral_test_secret()
    secret = SecretString(secret_value)
    assert secret_value not in str(secret)
    assert secret_value not in repr(secret)
    assert f"{secret}" == "***REDACTED***"


def test_secret_string_reveal_returns_the_real_value_only_when_explicitly_called():
    secret_value = _ephemeral_test_secret()
    secret = SecretString(secret_value)
    assert secret.reveal() == secret_value
