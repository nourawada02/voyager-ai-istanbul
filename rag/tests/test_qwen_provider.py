"""Hermetic tests for the Qwen (Alibaba/QwenCloud Model Studio,
OpenAI-compatible) provider -- no live network calls anywhere in this
file. Mirrors test_groq_provider.py's patterns: `urllib.request.urlopen`
is always monkeypatched before any provider method runs, and HTTP errors
are constructed directly rather than raised over a real socket."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from rag import llm_providers
from rag.llm_providers import (
    ProviderRateLimitExceeded,
    QwenConfigurationError,
    QwenProvider,
    StructuredGenerationFailure,
)


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


def _canned_response_raw_content(content_str: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content_str}}]}).encode("utf-8")


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError("https://example.invalid", code, "error", headers, None)


@pytest.fixture(autouse=True)
def _qwen_env(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test-qwen-key-not-real")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    llm_providers._qwen_last_call_at[0] = 0.0


def test_generate_constructs_expected_endpoint_and_body(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _JSONResponse(_canned_response({"answer_status": "insufficient", "claims": []}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)

    result = QwenProvider().generate("sys", "usr")

    req = captured["req"]
    assert req.full_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "qwen3.7-flash"
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert '"answer_status": "insufficient"' in result


def test_json_object_mode_is_requested(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    QwenProvider().generate("sys", "usr")

    body = json.loads(captured["req"].data.decode("utf-8"))
    assert body["response_format"] == {"type": "json_object"}


def test_non_thinking_mode_is_requested(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    QwenProvider().generate("sys", "usr")

    body = json.loads(captured["req"].data.decode("utf-8"))
    assert body["enable_thinking"] is False


def test_authorization_header_uses_bearer_without_leaking_key_elsewhere(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    QwenProvider().generate("sys", "usr")

    req = captured["req"]
    assert req.headers["Authorization"] == "Bearer test-qwen-key-not-real"
    body_text = req.data.decode("utf-8")
    assert "test-qwen-key-not-real" not in body_text
    assert "test-qwen-key-not-real" not in req.full_url


def test_qwen_api_key_and_base_url_are_read_at_call_time_not_stored_on_the_dataclass():
    provider = QwenProvider()
    assert "api_key" not in provider.__dataclass_fields__
    assert "base_url" not in provider.__dataclass_fields__
    assert not any("key" in f.lower() for f in provider.__dataclass_fields__)


def test_dashscope_api_key_fallback_is_honored(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-fallback-key")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    QwenProvider().generate("sys", "usr")

    assert captured["req"].headers["Authorization"] == "Bearer dashscope-fallback-key"


def test_qwen_api_key_takes_precedence_over_dashscope_fallback(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "primary-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fallback-key")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    QwenProvider().generate("sys", "usr")

    assert captured["req"].headers["Authorization"] == "Bearer primary-key"


def test_missing_base_url_fails_closed_before_any_network_call(monkeypatch):
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)

    def fake_urlopen(req, timeout=None):
        raise AssertionError("must never reach the network when QWEN_BASE_URL is missing")

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(QwenConfigurationError, match="QWEN_BASE_URL"):
        QwenProvider().generate("sys", "usr")


def test_non_https_base_url_fails_closed_before_any_network_call(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "http://insecure.example.com/v1")

    def fake_urlopen(req, timeout=None):
        raise AssertionError("must never reach the network with a non-https base url")

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(QwenConfigurationError, match="https"):
        QwenProvider().generate("sys", "usr")


def test_base_url_with_workspace_specific_hostname_is_accepted(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "https://my-workspace-123.dashscope.aliyuncs.com/compatible-mode/v1")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _JSONResponse(_canned_response({"ok": True}))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    QwenProvider().generate("sys", "usr")

    assert captured["req"].full_url.startswith("https://my-workspace-123.dashscope.aliyuncs.com/")


def test_valid_qwen_response_is_parsed(monkeypatch):
    monkeypatch.setattr(
        llm_providers.urllib.request,
        "urlopen",
        lambda req, timeout=None: _JSONResponse(_canned_response({"answer_status": "grounded", "claims": []})),
    )

    def validate(data):
        assert data["answer_status"] in ("grounded", "insufficient")

    result = QwenProvider().generate_json("sys", "usr", validate, max_retries=2)
    assert result["answer_status"] == "grounded"


def test_malformed_json_triggers_only_bounded_structured_generation_repair(monkeypatch):
    call_count = {"n": 0}
    responses = ["not json at all", json.dumps({"answer_status": "insufficient", "claims": []})]

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        return _JSONResponse(_canned_response_raw_content(responses[call_count["n"] - 1]))

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)

    def validate(data):
        assert data["answer_status"] in ("grounded", "insufficient")

    result = QwenProvider().generate_json("sys", "usr", validate, max_retries=2)
    assert result["answer_status"] == "insufficient"
    assert call_count["n"] == 2


def test_malformed_json_never_exceeds_the_bounded_retry_budget(monkeypatch):
    monkeypatch.setattr(
        llm_providers.urllib.request,
        "urlopen",
        lambda req, timeout=None: _JSONResponse(_canned_response_raw_content("still not json")),
    )

    with pytest.raises(StructuredGenerationFailure):
        QwenProvider().generate_json("sys", "usr", lambda data: None, max_retries=2)


def test_401_fails_safely_without_secret_leakage(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "super-secret-qwen-key-401")
    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(401)))

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        QwenProvider().generate("sys", "usr")

    assert exc_info.value.code == 401
    assert "super-secret-qwen-key-401" not in str(exc_info.value)


def test_403_fails_safely_without_secret_leakage(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "super-secret-qwen-key-403")
    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(403)))

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        QwenProvider().generate("sys", "usr")

    assert exc_info.value.code == 403
    assert "super-secret-qwen-key-403" not in str(exc_info.value)


def test_401_and_403_are_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError):
        QwenProvider().generate("sys", "usr")

    assert calls["n"] == 1


def test_429_handling_is_bounded_and_honors_a_valid_short_retry_after(monkeypatch):
    responses = iter([_http_error(429, "3"), _JSONResponse(_canned_response({"ok": True}))])
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=None):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)

    result = QwenProvider().generate("sys", "usr")

    # The honored Retry-After (3s) is among the sleeps; a small
    # additional proactive pacing sleep before the second attempt is
    # expected and bounded -- never an unbounded/indefinite wait.
    assert 3.0 in sleeps
    assert all(s <= llm_providers._QWEN_MAX_RETRY_SLEEP_SECONDS for s in sleeps)
    assert result


def test_429_with_excessive_retry_after_fails_fast_without_sleeping_indefinitely(monkeypatch):
    monkeypatch.setattr(
        llm_providers.urllib.request, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(429, "36000"))
    )
    sleeps: list[float] = []
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)

    with pytest.raises(ProviderRateLimitExceeded) as exc_info:
        QwenProvider().generate("sys", "usr")

    assert exc_info.value.retry_after_seconds == 36000.0
    assert sleeps == []


def test_qwen_and_groq_pacing_state_are_independent(monkeypatch):
    """Regression: Qwen must never share Groq's rate-limit pacing clock
    -- each provider paces only against its own prior call time."""
    llm_providers._groq_last_call_at[0] = 999999.0  # simulate Groq having just called
    llm_providers._qwen_last_call_at[0] = 0.0

    monkeypatch.setattr(
        llm_providers.urllib.request, "urlopen", lambda req, timeout=None: _JSONResponse(_canned_response({"ok": True}))
    )
    sleeps: list[float] = []
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)

    QwenProvider().generate("sys", "usr")

    # Qwen's own pacing state (last call at 0.0, real monotonic clock is
    # far larger) never triggers a proactive sleep, regardless of Groq's.
    assert sleeps == []
