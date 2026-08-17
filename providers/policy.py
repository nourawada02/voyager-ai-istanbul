"""Provider-neutral failure, retry, cache, and circuit-breaker policy
(Checkpoint Phase 4 C.0, Step 4). Every function here is pure and
deterministic given its inputs (including an explicit `now`) -- no real
clock read, no real sleep, no real network -- so the whole module is
hermetically testable without mocking time or I/O.

Bounds mirror architecture.md §13.3 ("MCP / A2A / provider call retries:
max 2 retries, bounded exponential backoff") as the default, extended
here with the richer operational vocabulary live, retryable providers
need (Retry-After, rate-limit classification, circuit breaker, cache
staleness) that the frozen/historical Phase 1-3 data sources never
required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# The shared status vocabulary (mirrors ProviderResponseEnvelope.schema.json
# 1.1.0's 'status' enum exactly -- kept as a plain tuple, not duplicated
# per provider module).
RESULT_STATUSES = (
    "success", "partial", "unavailable", "unsupported", "invalid_request",
    "timeout", "rate_limited", "provider_error", "stale", "cancelled",
)

# A provider failure must never be represented as an empty successful
# result (Checkpoint Phase 4 C.0 explicit requirement) -- these are the
# statuses that count as "did not succeed" for retry/circuit-breaker
# purposes.
FAILURE_STATUSES = frozenset({"unavailable", "timeout", "rate_limited", "provider_error", "cancelled"})

# Statuses that mean "no data and no computed estimate exist at all" --
# these must use data_mode="unavailable" (ProviderResponseEnvelope 1.1.0's
# correction-pass value), never "estimated". "estimated" is reserved for
# an adapter that genuinely computed a fallback (e.g. a Haversine
# distance) -- none of the statuses below imply that happened.
# invalid_request/unsupported (Checkpoint Phase 4 C.1 fix) are included
# too: a rejected request never even reaches a live data source, so it is
# exactly as "no data" as a timeout or a provider error -- found via
# OpenMeteoWeatherProvider's own hermetic tests, where an invalid_request
# status was incorrectly reported with data_mode='live'.
NO_DATA_STATUSES = frozenset({
    "timeout", "rate_limited", "provider_error", "cancelled", "unavailable",
    "invalid_request", "unsupported",
})


def completeness_for_status(status: str) -> float:
    """Honest completeness for the shared DataQuality.completeness field.
    'stale' data is still complete (just aged) -- 0.0 completeness there
    would falsely suggest no data exists at all. 'partial' sits between
    the two. Every NO_DATA_STATUSES value is genuinely 0.0: nothing was
    retrieved."""
    if status in ("success", "stale"):
        return 1.0
    if status == "partial":
        return 0.5
    return 0.0


def data_mode_for_status(status: str, computed_estimate: bool = False) -> str:
    """The honest data_mode for a given operational status. 'stale' means
    we DO have data, just past its freshness window -- 'cached', never
    'unavailable' or 'estimated'. A NO_DATA_STATUSES status means
    'unavailable' unless the caller explicitly computed a real fallback
    (computed_estimate=True), in which case 'estimated' is honest."""
    if status == "stale":
        return "cached"
    if status in NO_DATA_STATUSES:
        return "estimated" if computed_estimate else "unavailable"
    return "live"


@dataclass(frozen=True)
class TimeoutPolicy:
    """`connect_timeout_seconds` is kept for interface compatibility and
    is meaningful for a transport that can bound connect and read phases
    separately -- `providers.http_transport.UrllibHttpTransport` cannot
    (see its own module docstring: the stdlib `urllib.request` exposes
    only one end-to-end socket timeout), so that transport enforces only
    `total_timeout_seconds` and does not separately enforce
    `connect_timeout_seconds`. A future transport built on a lower-level
    HTTP client could honor both independently without this dataclass
    needing to change shape."""

    connect_timeout_seconds: float = 3.0
    total_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2  # architecture.md §13.3's existing project-wide default
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    respect_retry_after: bool = True


def compute_backoff_seconds(attempt: int, policy: RetryPolicy, retry_after_seconds: Optional[float] = None) -> float:
    """Pure function: attempt is 0-indexed (0 = first retry). Honors a
    provider-supplied Retry-After header when present and
    respect_retry_after is set, always capped at max_backoff_seconds so a
    misbehaving provider can never stall a caller indefinitely. Otherwise
    exponential backoff (base * 2**attempt), also capped."""
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    if policy.respect_retry_after and retry_after_seconds is not None:
        return max(0.0, min(retry_after_seconds, policy.max_backoff_seconds))
    return min(policy.base_backoff_seconds * (2**attempt), policy.max_backoff_seconds)


def classify_http_status(status_code: int) -> str:
    """Maps an HTTP status code to the shared result-status vocabulary.
    Pure and total -- every integer status code maps to something, never
    raises. Not provider-specific: any adapter can reuse this for a
    generic REST backend."""
    if 200 <= status_code < 300:
        return "success"
    if status_code == 400 or status_code == 422:
        return "invalid_request"
    if status_code == 404:
        return "unsupported"
    if status_code == 408:
        return "timeout"
    if status_code == 429:
        return "rate_limited"
    if status_code == 504:
        return "timeout"
    if 500 <= status_code < 600:
        return "provider_error"
    return "provider_error"


@dataclass(frozen=True)
class CacheEntry:
    value: dict
    stored_at_epoch_seconds: float
    ttl_seconds: float

    def age_seconds(self, now_epoch_seconds: float) -> float:
        return max(0.0, now_epoch_seconds - self.stored_at_epoch_seconds)

    def is_stale(self, now_epoch_seconds: float) -> bool:
        return self.age_seconds(now_epoch_seconds) > self.ttl_seconds


def cache_status_for(entry: Optional[CacheEntry], now_epoch_seconds: float) -> tuple[str, Optional[int]]:
    """Returns (cache_status, cache_age_seconds) matching
    ProviderResponseEnvelope.schema.json 1.1.0's cache_status/
    cache_age_seconds fields -- cache_age_seconds is None exactly when
    cache_status is 'miss' or 'bypass' (never present, per the schema's
    own conditional requirement)."""
    if entry is None:
        return "miss", None
    age = int(entry.age_seconds(now_epoch_seconds))
    if entry.is_stale(now_epoch_seconds):
        return "stale", age
    return "hit", age


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Pure, in-memory circuit-breaker state machine -- no real timer, no
    background thread. The caller supplies `now_epoch_seconds` explicitly
    to every method, so this is fully deterministic under test."""

    failure_threshold: int = 3
    open_duration_seconds: float = 30.0
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at_epoch_seconds: Optional[float] = field(default=None, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    def should_allow_call(self, now_epoch_seconds: float) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            assert self._opened_at_epoch_seconds is not None
            if now_epoch_seconds - self._opened_at_epoch_seconds >= self.open_duration_seconds:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow exactly one probe call through

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at_epoch_seconds = None

    def record_failure(self, now_epoch_seconds: float) -> None:
        self._consecutive_failures += 1
        if self._state == CircuitState.HALF_OPEN or self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at_epoch_seconds = now_epoch_seconds
