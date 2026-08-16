"""Hermetic tests for the bounded transient-transport-failure retry
layer shared by GroqProvider and QwenProvider (Checkpoint Phase 3
remediation: transport resilience). No live network calls anywhere in
this file -- every test monkeypatches `urllib.request.urlopen` and, where
relevant, `time.sleep`, before any provider method runs.

Regression coverage for the live failure this repair fixes: a Groq judge
call failed during TLS connection establishment with
`URLError(WinError 10054 ...)` and crashed the whole smoke run with exit
code 1 instead of being recorded as a bounded provider failure."""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request

import pytest

from rag import llm_providers
from rag.llm_providers import GroqProvider, ProviderTransportError, QwenProvider


class _JSONResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _canned_response(content: dict) -> bytes:
    return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode("utf-8")


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError("https://example.invalid", code, "error", headers, None)


@pytest.fixture(autouse=True)
def _provider_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-not-real")
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen-key-not-real")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_BASE_URL", "https://example-workspace.aliyuncs.com/compatible-mode/v1")
    llm_providers._groq_last_call_at[0] = 0.0
    llm_providers._qwen_last_call_at[0] = 0.0


# --- Recovery: one transient failure, then success -------------------


def test_groq_recovers_after_one_transient_connection_reset(monkeypatch):
    """Regression for the live failure: URLError wrapping a
    ConnectionResetError (WinError 10054) on the first attempt, success
    on the second -- exactly one bounded retry."""
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise urllib.error.URLError(
                ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
            )
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    result = GroqProvider().generate("sys", "usr")

    assert call_count["n"] == 2
    assert result


def test_qwen_recovers_after_one_transient_connection_reset(monkeypatch):
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise urllib.error.URLError(ConnectionResetError(10054, "connection reset"))
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    result = QwenProvider().generate("sys", "usr")

    assert call_count["n"] == 2
    assert result


def test_bare_timeout_error_recovers_after_one_retry(monkeypatch):
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TimeoutError("timed out")
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    result = GroqProvider().generate("sys", "usr")
    assert call_count["n"] == 2
    assert result


def test_bare_ssl_error_recovers_after_one_retry(monkeypatch):
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ssl.SSLError("TLS handshake failed")
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    result = QwenProvider().generate("sys", "usr")
    assert call_count["n"] == 2
    assert result


def test_ssl_error_wrapped_in_urlerror_recovers_after_one_retry(monkeypatch):
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise urllib.error.URLError(ssl.SSLError("TLS handshake failed"))
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    result = GroqProvider().generate("sys", "usr")
    assert call_count["n"] == 2
    assert result


# --- Exhaustion: three consecutive transient failures -----------------


def test_groq_three_consecutive_transient_failures_raise_provider_transport_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(ConnectionResetError(10054, "connection reset"))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    with pytest.raises(ProviderTransportError) as exc_info:
        GroqProvider().generate("sys", "usr")

    assert exc_info.value.provider_name == "groq"
    assert exc_info.value.attempts == llm_providers._MAX_TRANSIENT_TRANSPORT_ATTEMPTS
    assert exc_info.value.attempts <= 3


def test_qwen_three_consecutive_transient_failures_raise_provider_transport_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    with pytest.raises(ProviderTransportError) as exc_info:
        QwenProvider().generate("sys", "usr")

    assert exc_info.value.provider_name == "qwen"
    assert exc_info.value.attempts == llm_providers._MAX_TRANSIENT_TRANSPORT_ATTEMPTS
    assert exc_info.value.attempts <= 3


def test_max_transient_transport_attempts_is_at_most_three():
    assert llm_providers._MAX_TRANSIENT_TRANSPORT_ATTEMPTS <= 3


# --- Bounded sleeps -----------------------------------------------------


def test_transient_retry_sleeps_are_bounded_by_the_configured_ceiling(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(ConnectionResetError(10054, "connection reset"))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)

    with pytest.raises(ProviderTransportError):
        GroqProvider().generate("sys", "usr")

    assert sleeps  # at least one backoff sleep occurred
    assert all(s <= llm_providers._GROQ_MAX_RETRY_SLEEP_SECONDS for s in sleeps)


def test_qwen_transient_retry_sleeps_are_bounded_by_its_own_ceiling(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)

    with pytest.raises(ProviderTransportError):
        QwenProvider().generate("sys", "usr")

    assert sleeps
    assert all(s <= llm_providers._QWEN_MAX_RETRY_SLEEP_SECONDS for s in sleeps)


# --- Exception ordering: HTTPError first, non-429 never retried --------


@pytest.mark.parametrize("code", [400, 401, 403])
def test_non_429_http_errors_are_not_retried_as_transient(monkeypatch, code):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(code)

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        GroqProvider().generate("sys", "usr")

    assert exc_info.value.code == code
    assert calls["n"] == 1  # never retried, transient or otherwise


def test_non_transient_dns_failure_is_not_retried():
    """A DNS/host resolution failure is a URLError but not one of the
    listed transient causes -- it must propagate immediately, exactly as
    before this repair, never mistaken for a connection reset/timeout."""
    exc = urllib.error.URLError(socket.gaierror("Name or service not known"))
    assert llm_providers._is_transient_transport_error(exc) is False


def test_existing_429_handling_still_passes(monkeypatch):
    responses = iter([_http_error(429, "2"), _JSONResponse(_canned_response({"ok": True}))])
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=None):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)

    result = GroqProvider().generate("sys", "usr")

    assert 2.0 in sleeps
    assert result


def test_429_after_a_transient_retry_still_resolves_through_existing_policy(monkeypatch):
    """Exception ordering holds even when the two failure modes are
    mixed within one call: HTTPError(429) is always handled by the 429
    branch, a raw URLError by the transient branch, regardless of order."""
    responses = iter([
        urllib.error.URLError(ConnectionResetError(10054, "reset")),
        _http_error(429, "1"),
        _JSONResponse(_canned_response({"ok": True})),
    ])

    def fake_urlopen(req, timeout=None):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    result = GroqProvider().generate("sys", "usr")
    assert result


# --- No secrets in exception text --------------------------------------


def test_groq_transport_error_message_contains_no_secret_header_or_base_url(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-groq-key-value")

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(
            ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
        )

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    with pytest.raises(ProviderTransportError) as exc_info:
        GroqProvider().generate("sys", "usr")

    message = str(exc_info.value)
    assert "super-secret-groq-key-value" not in message
    assert "Bearer" not in message
    assert "Authorization" not in message
    assert "api.groq.com" not in message
    assert "WinError" not in message  # no raw underlying exception text leaked
    assert "10054" not in message


def test_qwen_transport_error_message_contains_no_secret_or_workspace_base_url(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "super-secret-qwen-key-value")
    monkeypatch.setenv("QWEN_BASE_URL", "https://my-workspace-secret-id.aliyuncs.com/compatible-mode/v1")

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda s: None)

    with pytest.raises(ProviderTransportError) as exc_info:
        QwenProvider().generate("sys", "usr")

    message = str(exc_info.value)
    assert "super-secret-qwen-key-value" not in message
    assert "my-workspace-secret-id" not in message
    assert "Bearer" not in message
    assert "Authorization" not in message
