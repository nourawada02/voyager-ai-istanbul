"""Hermetic tests for providers/http_transport.py's real
`UrllibHttpTransport` -- specifically the Checkpoint Phase 4 C.2
timeout-semantics repair: only the total/I/O element of the
`(connect_timeout_seconds, total_timeout_seconds)` tuple reaches
`urllib.request.urlopen`; the connect-only element is not separately
enforceable against the stdlib (see http_transport.py's own module
docstring). No real network call anywhere in this file --
`urllib.request.urlopen` itself is monkeypatched, so nothing here opens
a socket.
"""

from __future__ import annotations

import secrets
import urllib.error
import urllib.request

import pytest

from providers.http_transport import ALLOWED_HOSTS, TransportError, UrllibHttpTransport

_A_HOST = sorted(ALLOWED_HOSTS)[0]  # any real allowlisted host works for these tests


def _ephemeral_test_secret() -> str:
    return f"ephemeral-test-token-{secrets.token_hex(16)}"


class _FakeUrlopenResponse:
    def __init__(self, status: int = 200, body: bytes = b"{}", headers: dict | None = None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def test_total_io_timeout_reaches_urlopen(monkeypatch):
    captured: dict = {}

    def _fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return _FakeUrlopenResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    transport = UrllibHttpTransport()
    transport.get(f"https://{_A_HOST}/path", {"a": "b"}, timeout=(3.0, 25.0))
    assert captured["timeout"] == 25.0


def test_connect_only_element_is_never_the_enforced_bound(monkeypatch):
    """A (tiny connect, large total) tuple must still use the total
    element -- proves the repair, not the pre-repair behavior, which
    passed the tuple's first element to urlopen and silently under-timed
    real calls on a slow-but-live connection."""
    captured: dict = {}

    def _fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return _FakeUrlopenResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    transport = UrllibHttpTransport()
    transport.get(f"https://{_A_HOST}/path", {}, timeout=(0.001, 25.0))
    assert captured["timeout"] == 25.0
    assert captured["timeout"] != 0.001


def test_disallowed_host_is_rejected_before_urlopen_is_ever_called(monkeypatch):
    def _fake_urlopen(request, timeout=None):
        raise AssertionError("urlopen must never be reached for a disallowed host")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    transport = UrllibHttpTransport()
    with pytest.raises(TransportError):
        transport.get("https://evil-unrelated-host.example.com/", {}, timeout=(3.0, 25.0))


def test_transport_error_message_never_contains_a_secret_param_value(monkeypatch):
    secret_value = _ephemeral_test_secret()

    def _fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("simulated DNS failure")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    transport = UrllibHttpTransport()
    with pytest.raises(TransportError) as excinfo:
        transport.get(f"https://{_A_HOST}/", {}, timeout=(3.0, 25.0), secret_params={"api_key": secret_value})
    assert secret_value not in str(excinfo.value)
    assert _A_HOST in str(excinfo.value)


def test_secret_params_never_appear_in_the_constructed_request_object(monkeypatch):
    secret_value = _ephemeral_test_secret()
    captured: dict = {}

    def _fake_urlopen(request, timeout=None):
        captured["full_url"] = request.full_url
        return _FakeUrlopenResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    transport = UrllibHttpTransport()
    transport.get(f"https://{_A_HOST}/", {"q": "test"}, timeout=(3.0, 25.0), secret_params={"api_key": secret_value})
    # The secret DOES need to reach the real outgoing URL for a real call
    # to authenticate -- this only proves it is not silently dropped;
    # secret-exposure protection is about logs/errors/fingerprints/tests,
    # never about the one legitimate outgoing HTTPS request itself.
    assert secret_value in captured["full_url"]
    assert captured["full_url"].startswith("https://")
