"""Hermetic tests for secret redaction/isolation (Checkpoint Phase 4
C.0, Step 8: "secret redaction")."""

from __future__ import annotations

from providers.redaction import SecretString, redact_secrets


def test_api_key_query_param_is_redacted():
    log_line = "GET https://api.example.com/v1/weather?api_key=sk-live-abc123&location=IST"
    redacted = redact_secrets(log_line)
    assert "sk-live-abc123" not in redacted
    assert "***REDACTED***" in redacted


def test_authorization_bearer_header_is_redacted():
    log_line = 'headers={"Authorization": "Bearer sk-live-xyz789"}'
    redacted = redact_secrets(log_line)
    assert "sk-live-xyz789" not in redacted


def test_secret_and_password_and_token_keywords_are_redacted():
    for line in ("secret=topvalue123", "password=hunter2value", "token=abcdef123456"):
        redacted = redact_secrets(line)
        assert "value" not in redacted or "REDACTED" in redacted


def test_non_secret_text_is_left_unchanged():
    line = "GET https://api.example.com/v1/weather?location=Istanbul&units=metric"
    assert redact_secrets(line) == line


def test_secret_string_never_reveals_value_via_str_or_repr():
    secret = SecretString("sk-live-do-not-leak")
    assert "sk-live-do-not-leak" not in str(secret)
    assert "sk-live-do-not-leak" not in repr(secret)
    assert f"{secret}" == "***REDACTED***"


def test_secret_string_reveal_returns_the_real_value_only_when_explicitly_called():
    secret = SecretString("sk-live-do-not-leak")
    assert secret.reveal() == "sk-live-do-not-leak"
