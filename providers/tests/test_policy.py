"""Hermetic tests for the provider-neutral failure/retry/cache/circuit-
breaker policy (Checkpoint Phase 4 C.0, Step 8: timeout, rate limit,
stale cache, cancellation-adjacent bounds). Every test supplies an
explicit `now` -- no real clock, no real sleep, no real network."""

from __future__ import annotations

import pytest

from providers.policy import (
    CacheEntry,
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    cache_status_for,
    classify_http_status,
    compute_backoff_seconds,
)

# --- backoff -------------------------------------------------------------------------


def test_backoff_doubles_each_attempt_up_to_the_cap():
    policy = RetryPolicy(base_backoff_seconds=1.0, max_backoff_seconds=100.0)
    assert compute_backoff_seconds(0, policy) == 1.0
    assert compute_backoff_seconds(1, policy) == 2.0
    assert compute_backoff_seconds(2, policy) == 4.0


def test_backoff_is_capped_at_max_backoff_seconds():
    policy = RetryPolicy(base_backoff_seconds=1.0, max_backoff_seconds=3.0)
    assert compute_backoff_seconds(10, policy) == 3.0


def test_retry_after_header_is_honored_when_present():
    policy = RetryPolicy(respect_retry_after=True, max_backoff_seconds=60.0)
    assert compute_backoff_seconds(0, policy, retry_after_seconds=12.0) == 12.0


def test_retry_after_is_still_capped_at_max_backoff():
    policy = RetryPolicy(respect_retry_after=True, max_backoff_seconds=5.0)
    assert compute_backoff_seconds(0, policy, retry_after_seconds=999.0) == 5.0


def test_retry_after_ignored_when_policy_disables_it():
    policy = RetryPolicy(respect_retry_after=False, base_backoff_seconds=1.0, max_backoff_seconds=60.0)
    assert compute_backoff_seconds(0, policy, retry_after_seconds=99.0) == 1.0


def test_negative_attempt_rejected():
    with pytest.raises(ValueError):
        compute_backoff_seconds(-1, RetryPolicy())


# --- HTTP status classification --------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        (200, "success"),
        (204, "success"),
        (400, "invalid_request"),
        (422, "invalid_request"),
        (404, "unsupported"),
        (408, "timeout"),
        (429, "rate_limited"),
        (500, "provider_error"),
        (503, "provider_error"),
        (504, "timeout"),
    ],
)
def test_http_status_classification(code, expected):
    assert classify_http_status(code) == expected


def test_http_status_classification_is_total_never_raises_on_unknown_code():
    assert classify_http_status(999) == "provider_error"


# --- cache staleness -------------------------------------------------------------------


def test_cache_status_is_miss_when_no_entry():
    status, age = cache_status_for(None, now_epoch_seconds=1000.0)
    assert status == "miss"
    assert age is None


def test_cache_status_is_hit_within_ttl():
    entry = CacheEntry(value={}, stored_at_epoch_seconds=1000.0, ttl_seconds=60.0)
    status, age = cache_status_for(entry, now_epoch_seconds=1030.0)
    assert status == "hit"
    assert age == 30


def test_cache_status_is_stale_past_ttl_never_labeled_live():
    entry = CacheEntry(value={}, stored_at_epoch_seconds=1000.0, ttl_seconds=60.0)
    status, age = cache_status_for(entry, now_epoch_seconds=1200.0)
    assert status == "stale"
    assert age == 200


def test_cache_age_seconds_never_negative():
    entry = CacheEntry(value={}, stored_at_epoch_seconds=1000.0, ttl_seconds=60.0)
    assert entry.age_seconds(now_epoch_seconds=500.0) == 0.0  # clock skew: never negative age


# --- circuit breaker ---------------------------------------------------------------------


def test_circuit_starts_closed_and_allows_calls():
    breaker = CircuitBreaker(failure_threshold=3)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.should_allow_call(now_epoch_seconds=0.0) is True


def test_circuit_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        breaker.record_failure(now_epoch_seconds=0.0)
    assert breaker.state == CircuitState.OPEN
    assert breaker.should_allow_call(now_epoch_seconds=0.0) is False


def test_circuit_half_opens_after_open_duration_elapses():
    breaker = CircuitBreaker(failure_threshold=2, open_duration_seconds=30.0)
    breaker.record_failure(now_epoch_seconds=0.0)
    breaker.record_failure(now_epoch_seconds=0.0)
    assert breaker.state == CircuitState.OPEN
    assert breaker.should_allow_call(now_epoch_seconds=31.0) is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_circuit_reopens_on_failure_during_half_open_probe():
    breaker = CircuitBreaker(failure_threshold=2, open_duration_seconds=30.0)
    breaker.record_failure(now_epoch_seconds=0.0)
    breaker.record_failure(now_epoch_seconds=0.0)
    breaker.should_allow_call(now_epoch_seconds=31.0)  # transitions to HALF_OPEN
    breaker.record_failure(now_epoch_seconds=31.0)
    assert breaker.state == CircuitState.OPEN


def test_circuit_closes_on_success_during_half_open_probe():
    breaker = CircuitBreaker(failure_threshold=2, open_duration_seconds=30.0)
    breaker.record_failure(now_epoch_seconds=0.0)
    breaker.record_failure(now_epoch_seconds=0.0)
    breaker.should_allow_call(now_epoch_seconds=31.0)
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.should_allow_call(now_epoch_seconds=31.0) is True


def test_circuit_still_open_before_duration_elapses():
    breaker = CircuitBreaker(failure_threshold=1, open_duration_seconds=30.0)
    breaker.record_failure(now_epoch_seconds=100.0)
    assert breaker.should_allow_call(now_epoch_seconds=120.0) is False
