"""Tests for the FX-rate contract, the real Frankfurter adapter, and the
Decimal money-conversion helper (Manual QA remediation Q.1, §B)."""

from __future__ import annotations

import socket
from decimal import Decimal

import pytest

from providers.fx import FakeFxProvider, FxQuery
from providers.fx_frankfurter import FX_URL, FrankfurterFxProvider
from providers.http_transport import FakeHttpTransport, HttpResponse
from providers.money import ConversionUnavailable, FxQuote, convert_minor_units, quote_from_envelope
from providers.policy import RetryPolicy, TimeoutPolicy
from providers.tests.conftest import make_validator


def _json_response(payload: dict, status: int = 200) -> HttpResponse:
    import json

    return HttpResponse(status_code=status, body=json.dumps(payload).encode("utf-8"), headers={})


def _provider(transport: FakeHttpTransport, **kwargs) -> FrankfurterFxProvider:
    kwargs.setdefault("monotonic_clock", _MonotonicStub())
    kwargs.setdefault("clock", lambda: __import__("datetime").datetime(2026, 8, 20, 12, 0, tzinfo=__import__("datetime").timezone.utc))
    return FrankfurterFxProvider(
        transport=transport, sleep_fn=lambda s: None,
        retry_policy=RetryPolicy(max_retries=2, base_backoff_seconds=0.01, max_backoff_seconds=0.02),
        timeout_policy=TimeoutPolicy(connect_timeout_seconds=1.0, total_timeout_seconds=1.0),
        **kwargs,
    )


class _MonotonicStub:
    """A controllable fake monotonic clock -- starts at 0.0, advances only
    when told to, so cache-hit/stale-window tests are deterministic."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --- real adapter: success, caching, degradation --------------------------------------


def test_fx_rate_success_is_schema_valid(envelope_validator, registry, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no real socket")))
    transport = FakeHttpTransport(responses={FX_URL: _json_response({"amount": 1.0, "base": "USD", "date": "2026-08-19", "rates": {"TRY": 47.935}})})
    provider = _provider(transport)
    envelope = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))

    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("FxRateResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]

    assert envelope["status"] == "success"
    assert envelope["provider"] == "frankfurter.app (ECB reference rates)"
    assert envelope["result"]["rate"] == "47.935"
    assert envelope["result"]["effective_date"] == "2026-08-19"
    assert envelope["cache_status"] == "miss"


def test_fx_rate_is_cached_for_a_bounded_period_second_call_is_a_hit():
    transport = FakeHttpTransport(responses={FX_URL: _json_response({"amount": 1.0, "base": "USD", "date": "2026-08-19", "rates": {"TRY": 47.935}})})
    clock = _MonotonicStub()
    provider = _provider(transport, monotonic_clock=clock, cache_ttl_seconds=3600.0)

    e1 = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))
    assert e1["cache_status"] == "miss"
    clock.now = 10.0  # 10s later -- well within the 1h TTL
    e2 = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))
    assert e2["cache_status"] == "hit"
    assert e2["cache_age_seconds"] == 10
    assert len([c for c in transport.call_log if c[0] == FX_URL]) == 1  # only ONE real call for the whole trip


def test_fx_rate_cache_expires_after_ttl_and_refetches():
    transport = FakeHttpTransport(responses={FX_URL: _json_response({"amount": 1.0, "base": "USD", "date": "2026-08-19", "rates": {"TRY": 47.935}})})
    clock = _MonotonicStub()
    provider = _provider(transport, monotonic_clock=clock, cache_ttl_seconds=60.0)

    provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))
    clock.now = 120.0  # past the 60s TTL
    e2 = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))
    assert e2["cache_status"] == "miss"
    assert len([c for c in transport.call_log if c[0] == FX_URL]) == 2


def test_fx_provider_unavailable_never_fabricates_a_rate(envelope_validator):
    transport = FakeHttpTransport(raise_transport_error=True)
    provider = _provider(transport)
    envelope = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))

    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    assert envelope["status"] == "timeout"
    assert envelope["result"]["rate"] is None
    assert envelope["result"]["effective_date"] is None
    assert envelope["data_mode"] == "unavailable"


def test_fx_provider_rate_limited_is_typed_never_silently_success():
    transport = FakeHttpTransport(responses={FX_URL: _json_response({}, status=429)})
    provider = _provider(transport)
    envelope = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))
    assert envelope["status"] == "rate_limited"
    assert envelope["result"]["rate"] is None


def test_fx_provider_falls_back_to_stale_cache_on_transient_failure_never_fabricates():
    transport = FakeHttpTransport(responses={FX_URL: _json_response({"amount": 1.0, "base": "USD", "date": "2026-08-19", "rates": {"TRY": 47.935}})})
    clock = _MonotonicStub()
    provider = _provider(transport, monotonic_clock=clock, cache_ttl_seconds=60.0)
    provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))  # populate cache

    clock.now = 120.0  # cache now stale by TTL
    transport.raise_transport_error = True  # live source now failing
    envelope = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))
    assert envelope["status"] == "success"
    assert envelope["cache_status"] == "stale"
    assert envelope["result"]["rate"] == "47.935"  # the real, previously-fetched rate -- never fabricated


# --- Decimal conversion -----------------------------------------------------------------


def _quote(rate: str = "47.935") -> FxQuote:
    return FxQuote(
        base_currency="USD", quote_currency="TRY", rate=Decimal(rate), effective_date="2026-08-19",
        provider="frankfurter.app (ECB reference rates)", retrieved_at="2026-08-20T12:00:00Z", cache_status="hit",
    )


def test_try_amount_converted_to_try_is_unchanged():
    """Required test: TRY unchanged."""
    assert convert_minor_units(500000, _quote(), "TRY", "TRY") == 500000


def test_usd_to_try_conversion_is_decimal_exact():
    """Required test: USD conversion."""
    # $50.00 * 47.935 = 2396.75 TRY exactly -> 239675 minor units
    assert convert_minor_units(5000, _quote(), "USD", "TRY") == 239675


def test_try_to_usd_conversion_uses_the_inverse_rate():
    # 2396.75 TRY / 47.935 = 50.00 USD exactly
    assert convert_minor_units(239675, _quote(), "TRY", "USD") == 5000


def test_decimal_rounding_uses_round_half_up_never_float_truncation():
    """Required test: Decimal rounding."""
    # $0.01 * 47.935 = 0.4794 TRY -> rounds to 0.48 TRY (48 minor units),
    # never 0.47 (float truncation) and never banker's-rounding to 0.48
    # vs 0.47 ambiguity -- ROUND_HALF_UP is this project's one fixed rule.
    quote = _quote(rate="47.5")  # 1 * 0.005 boundary case: 0.005 -> rounds up
    assert convert_minor_units(1, quote, "USD", "TRY") == round(0.01 * 47.5 * 100)  # sanity cross-check
    quote_half = FxQuote(base_currency="USD", quote_currency="TRY", rate=Decimal("1.005"), effective_date="x", provider="x", retrieved_at="x", cache_status="hit")
    # $1.00 * 1.005 = 1.005 -> ROUND_HALF_UP -> 1.01, never 1.00
    assert convert_minor_units(100, quote_half, "USD", "TRY") == 101


def test_conversion_for_an_unrelated_currency_pair_is_refused_not_triangulated():
    with pytest.raises(ConversionUnavailable):
        convert_minor_units(1000, _quote(), "EUR", "TRY")  # quote only knows USD<->TRY


def test_quote_from_envelope_extracts_full_provenance():
    """Required test: rate provenance."""
    envelope = {
        "status": "success", "provider": "frankfurter.app (ECB reference rates)",
        "retrieved_at": "2026-08-20T12:00:00Z", "cache_status": "hit",
        "result": {"base_currency": "USD", "quote_currency": "TRY", "rate": "47.935", "effective_date": "2026-08-19"},
    }
    quote = quote_from_envelope(envelope)
    assert quote is not None
    assert quote.rate == Decimal("47.935")
    assert quote.effective_date == "2026-08-19"
    assert quote.provider == "frankfurter.app (ECB reference rates)"
    assert quote.retrieved_at == "2026-08-20T12:00:00Z"


def test_quote_from_envelope_returns_none_for_a_failed_envelope():
    """Required test: unavailable/rate-limited FX provider -- the
    downstream caller must be able to detect 'no usable quote' cleanly."""
    envelope = {"status": "timeout", "result": {"base_currency": "USD", "quote_currency": "TRY", "rate": None, "effective_date": None}}
    assert quote_from_envelope(envelope) is None


def test_quote_from_envelope_returns_none_for_rate_limited():
    envelope = {"status": "rate_limited", "result": {"base_currency": "USD", "quote_currency": "TRY", "rate": None, "effective_date": None}}
    assert quote_from_envelope(envelope) is None


# --- no accidental mislabeling -----------------------------------------------------------


def test_fake_fx_provider_never_labels_an_unsuccessful_rate_as_a_real_number():
    """Required test: no accidental TRY/USD label on an amount that was
    never actually converted -- the fake's own unavailable scenario must
    still produce rate=None, never a stale/zero placeholder."""
    provider = FakeFxProvider(scenario="unavailable")
    envelope = provider.fetch_rate(FxQuery(base_currency="USD", quote_currency="TRY"))
    assert envelope["result"]["rate"] is None
    assert quote_from_envelope(envelope) is None
