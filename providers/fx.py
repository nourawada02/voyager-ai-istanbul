"""FX rate: an independent typed tool/provider (Manual QA remediation
Q.1, §B). Mirrors providers/weather.py's own shape exactly -- a shared,
provider-neutral contract (FxQuery/FxProvider/FakeFxProvider) here, the
real adapter in providers/fx_frankfurter.py. Never implemented inside
System A as tightly-coupled code.

The rate itself is always a `decimal.Decimal` in this module and in every
real adapter -- never a Python float -- so no currency conversion built on
top of it can silently lose precision (architecture.md's own "Money is
Decimal or integer minor units -- never binary float" rule, extended here
to the rate that produces a Money amount)."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Optional, Protocol

from providers.fingerprint import deterministic_request_id, fingerprint_request
from providers.policy import completeness_for_status, data_mode_for_status

CAPABILITY = "fx_rate"
SCHEMA_VERSION = "1.0.0"
RESULT_SCHEMA_VERSION = "1.0.0"

_FIXED_TEST_CLOCK = "2026-08-01T12:00:00Z"


@dataclass(frozen=True)
class FxQuery:
    base_currency: str  # ISO 4217, e.g. "USD"
    quote_currency: str  # ISO 4217, e.g. "TRY" -- rate expresses 1 base_currency = rate quote_currency

    def normalized(self) -> dict:
        return {"base_currency": self.base_currency.upper(), "quote_currency": self.quote_currency.upper()}


class FxProvider(Protocol):
    def fetch_rate(self, query: FxQuery) -> dict: ...


def _envelope(
    query: FxQuery, status: str, retrieved_at: str, result: dict, provider_name: str,
    cache_status: str = "bypass", cache_age_seconds: Optional[int] = None,
) -> dict:
    fingerprint = fingerprint_request(CAPABILITY, query.normalized())
    data_mode = data_mode_for_status(status)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "request_id": deterministic_request_id(fingerprint, retrieved_at),
        "provider": provider_name,
        "capability": CAPABILITY,
        "data_mode": data_mode,
        "status": status,
        "query_fingerprint": fingerprint,
        "cache_status": cache_status,
        "retrieved_at": retrieved_at,
        "source_urls": [],
        "quality": {
            "schema_version": "1.0.0",
            "completeness": completeness_for_status(status),
            "freshness": data_mode,
            "assumptions": [] if status == "success" else [f"FX rate call ended with status={status!r}."],
        },
        "result": result,
    }
    if cache_age_seconds is not None:
        envelope["cache_age_seconds"] = cache_age_seconds
    return envelope


@dataclass
class FakeFxProvider:
    """Deterministic fake -- every hermetic test in this project uses
    this, never a real network call. A single fixed rate per currency
    pair, injectable for scripted-failure tests."""

    clock: Callable[[], str] = field(default=lambda: _FIXED_TEST_CLOCK)
    rates: dict[tuple[str, str], Decimal] = field(default_factory=lambda: {("USD", "TRY"): Decimal("40.00")})
    scenario: str = "success"  # "success" | "unavailable" | "timeout" | "rate_limited"

    def fetch_rate(self, query: FxQuery) -> dict:
        retrieved_at = self.clock()
        pair = (query.base_currency.upper(), query.quote_currency.upper())
        if self.scenario != "success":
            result = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "base_currency": pair[0], "quote_currency": pair[1],
                "rate": None, "effective_date": None,
            }
            return _envelope(query, self.scenario, retrieved_at, result, "fake-fx")
        rate = self.rates.get(pair)
        if rate is None:
            result = {"schema_version": RESULT_SCHEMA_VERSION, "base_currency": pair[0], "quote_currency": pair[1], "rate": None, "effective_date": None}
            return _envelope(query, "unsupported", retrieved_at, result, "fake-fx")
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "base_currency": pair[0], "quote_currency": pair[1],
            "rate": str(rate), "effective_date": "2026-08-01",
        }
        return _envelope(query, "success", retrieved_at, result, "fake-fx")
