"""Real Frankfurter.app FX-rate adapter (Manual QA remediation Q.1,
§B). Frankfurter publishes the European Central Bank's own daily
reference rates -- free, keyless, no account/billing, verified reachable
directly (2026-08-20) before this was implemented:

  https://api.frankfurter.dev/v1/latest?base=USD&symbols=TRY

Bounded in-memory TTL cache (providers/policy.py's own CacheEntry/
cache_status_for, reused unchanged -- never a duplicated cache
implementation): "cache rates for a bounded period so one trip does not
repeatedly call the FX source" is satisfied by one adapter instance's
cache surviving for the lifetime of the System A process that owns it,
keyed by (base, quote).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional

from providers.fx import CAPABILITY, FxQuery, RESULT_SCHEMA_VERSION, SCHEMA_VERSION
from providers.fingerprint import deterministic_request_id, fingerprint_request
from providers.http_transport import HttpResponse, HttpTransport, TransportError, UrllibHttpTransport
from providers.policy import (
    CacheEntry,
    RetryPolicy,
    TimeoutPolicy,
    cache_status_for,
    classify_http_status,
    completeness_for_status,
    compute_backoff_seconds,
    data_mode_for_status,
)
from providers.validation import validate_envelope

FX_URL = "https://api.frankfurter.dev/v1/latest"
PROVIDER_NAME = "frankfurter.app (ECB reference rates)"

DEFAULT_CACHE_TTL_SECONDS = 3600.0  # one hour -- rates that fresh never need a second real call for one trip


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FxUnexpectedResponseError(RuntimeError):
    """Raised internally when a 2xx Frankfurter response cannot be
    parsed into the expected shape -- mapped to status='provider_error',
    never allowed to propagate as a raw exception to a caller."""


@dataclass
class FrankfurterFxProvider:
    """Real FxProvider implementation. Every dependency (transport,
    clock, sleep) is injected with a safe real default and replaced with
    a deterministic fake in hermetic tests -- see
    providers/tests/test_fx_frankfurter.py."""

    transport: HttpTransport = field(default_factory=UrllibHttpTransport)
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    monotonic_clock: Callable[[], float] = field(default=time.monotonic)
    sleep_fn: Callable[[float], None] = field(default=time.sleep)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    provider_name: str = PROVIDER_NAME
    _cache: dict[tuple[str, str], CacheEntry] = field(default_factory=dict, repr=False)

    def fetch_rate(self, query: FxQuery) -> dict:
        retrieved_at_dt = self.clock()
        retrieved_at = _iso_z(retrieved_at_dt)
        pair = (query.base_currency.upper(), query.quote_currency.upper())
        now_epoch = self.monotonic_clock()

        entry = self._cache.get(pair)
        cache_status, cache_age = cache_status_for(entry, now_epoch)
        if entry is not None and cache_status == "hit":
            return self._build_envelope(query, "success", retrieved_at, entry.value, cache_status="hit", cache_age_seconds=cache_age)

        response = self._request_with_retries(pair)
        if isinstance(response, str):
            if entry is not None:
                # A real, bounded staleness fallback: serving a
                # slightly-aged real quote (never fabricated) is more
                # honest than failing a whole trip over a transient FX
                # outage -- reported as 'stale', never silently as fresh.
                return self._build_envelope(query, "success", retrieved_at, entry.value, data_mode_override="cached", cache_status="stale", cache_age_seconds=cache_age)
            return self._degraded_envelope(query, response, retrieved_at)

        try:
            result = self._parse(response, pair)
        except FxUnexpectedResponseError:
            if entry is not None:
                return self._build_envelope(query, "success", retrieved_at, entry.value, data_mode_override="cached", cache_status="stale", cache_age_seconds=cache_age)
            return self._degraded_envelope(query, "provider_error", retrieved_at)

        self._cache[pair] = CacheEntry(value=result, stored_at_epoch_seconds=now_epoch, ttl_seconds=self.cache_ttl_seconds)
        return self._build_envelope(query, "success", retrieved_at, result, cache_status="miss")

    def _parse(self, response: HttpResponse, pair: tuple[str, str]) -> dict[str, Any]:
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise FxUnexpectedResponseError("FX response was not valid JSON") from exc
        try:
            raw_rate = body["rates"][pair[1]]
            effective_date = body["date"]
        except (KeyError, TypeError) as exc:
            raise FxUnexpectedResponseError("expected 'rates'/'date' fields absent from response") from exc
        try:
            rate = Decimal(str(raw_rate))
        except InvalidOperation as exc:
            raise FxUnexpectedResponseError(f"rate {raw_rate!r} is not a valid decimal") from exc
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "base_currency": pair[0], "quote_currency": pair[1],
            "rate": str(rate), "effective_date": effective_date,
        }

    def _build_envelope(
        self, query: FxQuery, status: str, retrieved_at: str, result: dict, *,
        cache_status: str, cache_age_seconds: Optional[int] = None, data_mode_override: Optional[str] = None,
    ) -> dict:
        fingerprint = fingerprint_request(CAPABILITY, query.normalized())
        data_mode = data_mode_override or data_mode_for_status(status)
        envelope: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "request_id": deterministic_request_id(fingerprint, retrieved_at),
            "provider": self.provider_name,
            "capability": CAPABILITY,
            "data_mode": data_mode,
            "status": status,
            "query_fingerprint": fingerprint,
            "cache_status": cache_status,
            "retrieved_at": retrieved_at,
            "source_urls": [FX_URL] if status == "success" else [],
            "quality": {
                "schema_version": "1.0.0",
                "completeness": completeness_for_status(status),
                "freshness": data_mode,
                "assumptions": (
                    [f"FX rate served from a {cache_age_seconds}s-old cache entry after a real live fetch failed."]
                    if cache_status == "stale" else
                    ([] if status == "success" else [f"FX rate call ended with status={status!r}."])
                ),
            },
            "result": result,
        }
        if cache_age_seconds is not None:
            envelope["cache_age_seconds"] = cache_age_seconds
        validate_envelope(envelope)
        return envelope

    def _degraded_envelope(self, query: FxQuery, status: str, retrieved_at: str) -> dict:
        pair = (query.base_currency.upper(), query.quote_currency.upper())
        result = {"schema_version": RESULT_SCHEMA_VERSION, "base_currency": pair[0], "quote_currency": pair[1], "rate": None, "effective_date": None}
        return self._build_envelope(query, status, retrieved_at, result, cache_status="bypass")

    def _request_with_retries(self, pair: tuple[str, str]) -> HttpResponse | str:
        params = {"base": pair[0], "symbols": pair[1]}
        timeout = (self.timeout_policy.connect_timeout_seconds, self.timeout_policy.total_timeout_seconds)
        last_status = "provider_error"
        for attempt in range(self.retry_policy.max_retries + 1):
            retry_after: Optional[float] = None
            try:
                response = self.transport.get(FX_URL, params, timeout)
            except TransportError:
                last_status = "timeout"
            else:
                if 200 <= response.status_code < 300:
                    return response
                last_status = classify_http_status(response.status_code)
                if last_status == "rate_limited":
                    header_value = response.header("Retry-After")
                    if header_value is not None:
                        try:
                            retry_after = float(header_value)
                        except ValueError:
                            retry_after = None
                if last_status not in ("timeout", "rate_limited", "provider_error"):
                    return last_status
            if attempt < self.retry_policy.max_retries:
                self.sleep_fn(compute_backoff_seconds(attempt, self.retry_policy, retry_after))
        return last_status
