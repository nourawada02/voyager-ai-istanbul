"""Hermetic tests for the Groq hosted provider: same bounded-retry
structured-JSON contract as Ollama, and correct precedence in
detect_available_providers() (hosted providers always beat local Ollama;
Groq specifically was wired in after GROQ_API_KEY became available)."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from rag import llm_providers
from rag.llm_providers import (
    GroqProvider,
    ProviderRateLimitExceeded,
    RequiredProviderUnavailable,
    StructuredGenerationFailure,
    detect_available_providers,
    enforce_required_provider,
)


class _JSONResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b'{"ok": true}'


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError("https://example.invalid", code, "error", headers, None)


def _request() -> urllib.request.Request:
    return urllib.request.Request("https://example.invalid", data=b"{}", method="POST")


def test_generate_json_retries_on_invalid_json_then_succeeds():
    call_count = {"n": 0}
    responses = ["not json at all", '{"answer_status": "grounded", "claims": []}']

    class _FlakyProvider(GroqProvider):
        def generate(self, system: str, user: str) -> str:
            call_count["n"] += 1
            return responses[call_count["n"] - 1]

    provider = _FlakyProvider(model="fake")

    def validate(data):
        assert data["answer_status"] in ("grounded", "insufficient")

    result = provider.generate_json("sys", "usr", validate, max_retries=2)
    assert result["answer_status"] == "grounded"
    assert call_count["n"] == 2


def test_generate_json_raises_structured_generation_failure_after_exhausting_retries():
    class _AlwaysBrokenProvider(GroqProvider):
        def generate(self, system: str, user: str) -> str:
            return "still not json"

    provider = _AlwaysBrokenProvider(model="fake")

    def validate(data):
        pass

    try:
        provider.generate_json("sys", "usr", validate, max_retries=2)
        assert False, "expected StructuredGenerationFailure"
    except StructuredGenerationFailure:
        pass


def test_groq_is_preferred_over_ollama_when_its_key_is_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    detection = detect_available_providers()
    assert detection.hosted_provider == "groq"
    assert detection.local_ollama_model is None  # Ollama is never even probed once a hosted key is found

    provider = detection.selected
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"


def test_no_hosted_key_falls_back_to_ollama_probe(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("rag.llm_providers._ollama_reachable", lambda: None)

    detection = detect_available_providers()
    assert detection.hosted_provider is None
    assert detection.selected is None


def test_groq_api_key_is_read_at_call_time_not_stored_on_the_dataclass():
    """The key must never become a field value that a report/log could
    accidentally serialize -- it's read fresh from the environment inside
    generate(), not captured as dataclass state."""
    provider = GroqProvider()
    assert "api_key" not in provider.__dataclass_fields__
    assert not any("key" in f.lower() for f in provider.__dataclass_fields__)


@pytest.mark.parametrize(
    ("retry_after", "expected_wait"),
    [("5", 5.0), (None, 2.0), ("not-a-delay", 2.0)],
)
def test_rate_limit_uses_only_bounded_retry_delays(monkeypatch, retry_after, expected_wait):
    responses = iter([_http_error(429, retry_after), _JSONResponse()])
    monotonic_values = iter([100.0, 100.0, 200.0, 200.0])
    sleeps: list[float] = []

    def next_response(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", next_response)
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)
    llm_providers._groq_last_call_at[0] = 0.0

    result = llm_providers._post_json_with_rate_limit_handling(_request(), timeout=1.0)

    assert result == {"ok": True}
    assert sleeps == [expected_wait]


def test_excessive_retry_after_fails_fast_without_sleeping(monkeypatch):
    monkeypatch.setattr(
        llm_providers.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(429, "36000")),
    )
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: 100.0)
    sleeps: list[float] = []
    monkeypatch.setattr(llm_providers.time, "sleep", sleeps.append)
    llm_providers._groq_last_call_at[0] = 0.0

    with pytest.raises(ProviderRateLimitExceeded) as exc_info:
        llm_providers._post_json_with_rate_limit_handling(_request(), timeout=1.0)

    assert exc_info.value.retry_after_seconds == 36000.0
    assert exc_info.value.attempts == 1
    assert sleeps == []


def test_final_429_raises_typed_rate_limit_error(monkeypatch):
    monkeypatch.setattr(
        llm_providers.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(429, "1"))
    )
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(llm_providers.time, "sleep", lambda seconds: None)
    llm_providers._groq_last_call_at[0] = 0.0

    with pytest.raises(ProviderRateLimitExceeded) as exc_info:
        llm_providers._post_json_with_rate_limit_handling(_request(), timeout=1.0, max_attempts=1)

    assert exc_info.value.attempts == 1


def test_required_provider_passes_when_groq_selected(monkeypatch):
    monkeypatch.setenv("GEN_EVAL_REQUIRE_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    detection = detect_available_providers()
    enforce_required_provider(detection)  # must not raise


def test_required_provider_fails_before_calls_when_only_ollama_available(monkeypatch):
    monkeypatch.setenv("GEN_EVAL_REQUIRE_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("rag.llm_providers._ollama_reachable", lambda: "some-local-model")

    calls = {"n": 0}

    class _ExplodingProvider(GroqProvider):
        def generate(self, system: str, user: str) -> str:
            calls["n"] += 1
            raise AssertionError("provider must never be called once the required-provider gate has failed")

    detection = detect_available_providers()
    assert detection.provider_name == "ollama"

    with pytest.raises(RequiredProviderUnavailable):
        enforce_required_provider(detection)

    assert calls["n"] == 0


def test_required_provider_error_never_logs_secret(monkeypatch):
    secret = "sk-super-secret-value-should-never-appear"
    monkeypatch.setenv("GEN_EVAL_REQUIRE_PROVIDER", "openai")  # not implemented/selected -> mismatch
    monkeypatch.setenv("GROQ_API_KEY", secret)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    detection = detect_available_providers()
    with pytest.raises(RequiredProviderUnavailable) as exc_info:
        enforce_required_provider(detection)

    assert secret not in str(exc_info.value)


def test_no_required_provider_env_var_is_a_no_op(monkeypatch):
    monkeypatch.delenv("GEN_EVAL_REQUIRE_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("rag.llm_providers._ollama_reachable", lambda: None)

    detection = detect_available_providers()
    enforce_required_provider(detection)  # must not raise even with no provider at all


def test_non_429_http_error_is_not_retried(monkeypatch):
    calls = {"count": 0}

    def fail(*args, **kwargs):
        calls["count"] += 1
        raise _http_error(500)

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fail)
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: 100.0)
    llm_providers._groq_last_call_at[0] = 0.0

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        llm_providers._post_json_with_rate_limit_handling(_request(), timeout=1.0)

    assert exc_info.value.code == 500
    assert calls["count"] == 1


def test_transport_rate_limit_does_not_consume_structured_json_retries():
    calls = {"count": 0}

    class _RateLimitedProvider(GroqProvider):
        def generate(self, system: str, user: str) -> str:
            calls["count"] += 1
            raise ProviderRateLimitExceeded(36000.0, 1)

    with pytest.raises(ProviderRateLimitExceeded):
        _RateLimitedProvider(model="fake").generate_json("sys", "usr", lambda data: None, max_retries=2)

    assert calls["count"] == 1


def test_transport_timeout_does_not_consume_structured_json_retries():
    calls = {"count": 0}

    class _TimedOutProvider(GroqProvider):
        def generate(self, system: str, user: str) -> str:
            calls["count"] += 1
            raise TimeoutError("simulated timeout")

    with pytest.raises(TimeoutError, match="simulated timeout"):
        _TimedOutProvider(model="fake").generate_json("sys", "usr", lambda data: None, max_retries=2)

    assert calls["count"] == 1
