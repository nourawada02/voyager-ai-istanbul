"""Real SerpApi (Google Search) web-evidence adapter (Checkpoint Phase 4
C.2; docs/adr/0011-phase4-checkpointc2-serpapi-web-evidence-adapter.md).
Replaces an earlier, uncommitted Tavily-based implementation abandoned in
this checkpoint after both Tavily's website and keyless API proved
unreachable from this network (see ADR 0011 for the diagnosis) -- no
Tavily code or Tavily-specific test literal is present anywhere in this
module or its tests.

Exactly one fixed endpoint is ever contacted:

  https://serpapi.com/search   (engine=google)

Authentication: SerpApi's `google` engine requires `api_key` as a request
*query parameter* -- there is no keyless mode, unlike Tavily. The key is
read once from `SERPAPI_API_KEY` (or an explicitly injected `api_key`),
wrapped immediately in `providers.redaction.SecretString`, and revealed
exactly once, at the point `_request_with_retries` builds the transport
call's `secret_params` argument -- a channel `providers.http_transport`
keeps structurally separate from the public `params` dict so the key is
merged into the outgoing URL only inside `UrllibHttpTransport.get()`
itself, and is never recorded in a call log, a header log, an exception
message, a request fingerprint, a fixture, or a provenance record (see
providers/http_transport.py's own module docstring for the shared
mechanism). Only the request's public configuration (engine, language,
geographic context) is ever recorded, via `quality.assumptions`.

`providers/http_transport.py` is the only module in this project
permitted to build the final request URL, and it is transmitted only over
HTTPS -- exercised end-to-end by `providers/tests/test_web_evidence_serpapi.py`.

This module also exposes `SEARCH_URL`, `_request_params`, and the shared
`sanitize_url`/`classify_source_type` helpers below in a shape a future
SerpApi Google Flights adapter (Checkpoint C.3) can reuse directly (same
fixed host, same secret-parameter channel, same retry/backoff policy) --
Checkpoint C.2 implements only the web-search path; flight search is
explicitly out of scope for this checkpoint.

SerpApi results are treated as web evidence, never verified facts and
never frozen RAG citations (see providers/web_evidence.py's own module
docstring and contracts/WebEvidenceResult.schema.json). This adapter
never fetches or scrapes a returned URL -- URLs are evidence references
only. No LLM-generated answer or raw HTML is requested or surfaced.
"""

from __future__ import annotations

import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from providers.fingerprint import deterministic_request_id
from providers.http_transport import HttpResponse, HttpTransport, TransportError, UrllibHttpTransport
from providers.policy import (
    RetryPolicy,
    TimeoutPolicy,
    classify_http_status,
    completeness_for_status,
    compute_backoff_seconds,
    data_mode_for_status,
)
from providers.redaction import SecretString
from providers.validation import validate_envelope
from providers.web_evidence import CAPABILITY, RESULT_SCHEMA_VERSION, SCHEMA_VERSION, WebEvidenceQuery, normalize_query

SEARCH_URL = "https://serpapi.com/search"

PROVIDER_NAME = "serpapi"

DEFAULT_MAX_RESULTS = 5
HARD_MAX_RESULTS = 8  # never request or keep more than this, regardless of caller input
MAX_TITLE_LENGTH = 200
MAX_SNIPPET_LENGTH = 500  # stays comfortably under WebEvidenceResult's own 600-char schema cap

# This project's own scope is exclusively Istanbul trip planning
# (CLAUDE.md), so a fixed Istanbul/Turkey geographic context is always
# the right default -- there is no per-query override in this checkpoint,
# since every query this adapter will ever receive is Istanbul-related.
DEFAULT_LOCATION = "Istanbul, Turkey"
DEFAULT_GL = "tr"

# Mirrors the project's own established EN/TR/AR multilingual scope
# (Phase 3 RAG, docs/adr/0006-phase3-generation-acceptance.md). An
# unrecognized language_hint falls back to "en" rather than being sent to
# SerpApi unvalidated -- "language derived only from an allowlisted
# language hint" (Checkpoint Phase 4 C.2 requirement).
SUPPORTED_LANGUAGES = frozenset({"en", "tr", "ar"})

# Keyword fragments SerpApi's own documented error strings use for
# quota/rate-limit conditions (e.g. "You have run out of searches for
# this month.") -- classified as 'rate_limited', never 'provider_error',
# exactly like Tavily's 432/433 quota codes were in the prior
# implementation of this checkpoint. Any other top-level error string
# (e.g. "Invalid API key.") is classified as 'provider_error'.
_QUOTA_ERROR_KEYWORDS = ("run out of searches", "out of searches", "rate limit", "exceeded")


def _classify_error_message(message: str) -> str:
    lowered = message.lower()
    if any(keyword in lowered for keyword in _QUOTA_ERROR_KEYWORDS):
        return "rate_limited"
    return "provider_error"


# Deterministic source-type classification (same documented limitation as
# the prior Tavily implementation of this checkpoint): explicit, small
# host allowlists -- never an LLM classification, and never inferred from
# a result's own title/text ("official" is never assigned merely because
# a title says so). Intentionally short and will under-classify many
# genuinely official/news/reference sources as 'secondary' -- a caller
# must never treat 'secondary' as "definitely not official", only as
# "this adapter did not recognize it."
_GOV_MIL_SUFFIX_RE = re.compile(r"\.(gov|mil)(\.[a-z]{2})?$")
_NEWS_HOSTS = frozenset({
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "theguardian.com", "aljazeera.com", "cnn.com", "hurriyetdailynews.com",
})
_REFERENCE_HOSTS = frozenset({"wikipedia.org", "wikimedia.org", "britannica.com"})

_STRICT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STRICT_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _bare_host(hostname: str) -> str:
    host = hostname.lower()
    return host[4:] if host.startswith("www.") else host


def classify_source_type(hostname: Optional[str]) -> str:
    """Pure, deterministic, host-based. See module-level comment above
    this constant block for the documented limitation."""
    if not hostname:
        return "unknown"
    host = _bare_host(hostname)
    if _GOV_MIL_SUFFIX_RE.search(host):
        return "official"
    if host in _NEWS_HOSTS or any(host.endswith("." + d) for d in _NEWS_HOSTS):
        return "news"
    if host in _REFERENCE_HOSTS or any(host.endswith("." + d) for d in _REFERENCE_HOSTS):
        return "reference"
    return "secondary"


def sanitize_url(raw_url: str) -> Optional[str]:
    """Returns a canonical http(s) URL with credentials and fragment
    stripped, or None if the URL is invalid/unsafe. Pure string
    validation -- never resolves DNS, never opens a connection, never
    follows the URL."""
    try:
        parsed = urllib.parse.urlsplit(raw_url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None  # credentials embedded in the URL itself -- reject outright
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _truncate(text: str, max_length: int) -> tuple[str, bool]:
    if len(text) <= max_length:
        return text, False
    return text[: max_length - 1].rstrip() + "…", True


def _parse_supplied_date(raw_date: Any) -> Optional[str]:
    """Only a strict ISO date/datetime is ever accepted -- SerpApi's
    organic-result `date` field is frequently a relative or ambiguous
    string ("3 days ago", "Jan 1, 2024") that cannot be converted to an
    absolute timestamp without guessing, which WebEvidenceResult's own
    published_at contract forbids ("never guessed or backfilled").
    Anything not already an unambiguous ISO date/datetime is treated as
    genuinely unknown -- published_at stays null, never fabricated."""
    if not raw_date:
        return None
    text = str(raw_date).strip()
    if _STRICT_DATE_RE.match(text):
        return f"{text}T00:00:00Z"
    if _STRICT_DATETIME_RE.match(text):
        return text if (text.endswith("Z") or "+" in text[19:]) else f"{text}Z"
    return None


def _default_clock() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# SerpApi-specific timeout/retry configuration (Checkpoint Phase 4 C.2
# timeout-semantics repair). Sanitized live-gate diagnostics showed a
# cold connection to serpapi.com from this project's network can take
# several seconds longer to complete than the shared provider default
# (providers.policy.TimeoutPolicy's 10s) allows for -- see
# providers/http_transport.py's own docstring for why only the total/I/O
# bound (not a separate connect bound) is actually enforceable against
# urllib. 25s total/I/O timeout with at most one retry (two attempts
# total) and a short, capped backoff keeps the worst case at
# 25 + 5 + 25 = 55s, comfortably under this project's accepted
# 60-second workflow deadline.
_IO_TIMEOUT_SECONDS = 25.0
_MAX_RETRIES = 1
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 5.0


def _default_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_retries=_MAX_RETRIES, base_backoff_seconds=_BASE_BACKOFF_SECONDS, max_backoff_seconds=_MAX_BACKOFF_SECONDS
    )


def _default_timeout_policy() -> TimeoutPolicy:
    return TimeoutPolicy(connect_timeout_seconds=_IO_TIMEOUT_SECONDS, total_timeout_seconds=_IO_TIMEOUT_SECONDS)


@dataclass
class SerpApiWebEvidenceProvider:
    """Real WebEvidenceProvider implementation. Every dependency
    (transport, clock, sleep, cancellation check, api_key) is injected
    with a safe real default, and every one is replaced with a
    deterministic fake in hermetic tests -- see
    providers/tests/test_web_evidence_serpapi.py. `retry_policy` and
    `timeout_policy` default to this module's own SerpApi-tuned values
    above, not providers.policy's shared generic defaults."""

    transport: HttpTransport = field(default_factory=UrllibHttpTransport)
    clock: Callable[[], str] = field(default=_default_clock)
    sleep_fn: Callable[[float], None] = field(default=time.sleep)
    cancellation_check: Callable[[], bool] = field(default=lambda: False)
    retry_policy: RetryPolicy = field(default_factory=_default_retry_policy)
    timeout_policy: TimeoutPolicy = field(default_factory=_default_timeout_policy)
    api_key: Optional[SecretString] = field(default=None)
    provider_name: str = PROVIDER_NAME

    def __post_init__(self) -> None:
        if self.api_key is None:
            raw_key = os.environ.get("SERPAPI_API_KEY")
            if raw_key:
                self.api_key = SecretString(raw_key)

    # --- public interface (WebEvidenceProvider protocol) ----------------------------

    def search(self, query: WebEvidenceQuery) -> dict:
        retrieved_at = self.clock()

        if self.cancellation_check():
            return self._degraded_envelope(query, "cancelled", retrieved_at, [])

        validation_status = self._validate_request(query)
        if validation_status is not None:
            return self._degraded_envelope(query, validation_status, retrieved_at, [])

        response_or_status = self._request_with_retries(query)
        if isinstance(response_or_status, str):
            return self._degraded_envelope(query, response_or_status, retrieved_at, [])

        parsed = self._parse_or_status(response_or_status, query, retrieved_at)
        if isinstance(parsed, str):
            return self._degraded_envelope(query, parsed, retrieved_at, [])
        items, warnings = parsed

        envelope = self._build_envelope(query, status="success", data_mode="live", retrieved_at=retrieved_at, items=items, warnings=warnings)
        validate_envelope(envelope)
        return envelope

    # --- request validation --------------------------------------------------------

    def _validate_request(self, query: WebEvidenceQuery) -> Optional[str]:
        if self.api_key is None:
            return "unavailable"  # a deployment/config gap, not something the caller's query caused
        if not query.query or not query.query.strip():
            return "invalid_request"
        if query.max_results is not None and query.max_results < 1:
            return "invalid_request"
        return None

    # --- HTTP -----------------------------------------------------------------------

    def _effective_language(self, query: WebEvidenceQuery) -> str:
        return query.language_hint if query.language_hint in SUPPORTED_LANGUAGES else "en"

    def _request_params(self, query: WebEvidenceQuery) -> dict[str, Any]:
        bounded_num = min(max(1, query.max_results or DEFAULT_MAX_RESULTS), HARD_MAX_RESULTS)
        return {
            "engine": "google",
            "q": query.query,
            "output": "json",
            "safe": "active",
            "num": bounded_num,
            "hl": self._effective_language(query),
            "gl": DEFAULT_GL,
            "location": DEFAULT_LOCATION,
            # Explicitly "false" (SerpApi's own default): allows SerpApi to
            # serve a matching cached result when one exists, conserving
            # the free-tier search quota, rather than forcing a fresh
            # fetch on every call.
            "no_cache": "false",
        }

    def _request_with_retries(self, query: WebEvidenceQuery) -> HttpResponse | str:
        timeout = (self.timeout_policy.connect_timeout_seconds, self.timeout_policy.total_timeout_seconds)
        params = self._request_params(query)
        # Built fresh right here, immediately before the one transport
        # call that needs it -- never stored on self, never merged into
        # `params` above, so it can never leak into a log, a fingerprint,
        # or anything derived from `params`.
        secret_params = {"api_key": self.api_key.reveal()}
        last_status = "provider_error"
        for attempt in range(self.retry_policy.max_retries + 1):
            retry_after: Optional[float] = None
            try:
                response = self.transport.get(SEARCH_URL, params, timeout, secret_params=secret_params)
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
                    return last_status  # invalid_request-shaped provider response: retrying will not help

            if attempt < self.retry_policy.max_retries:
                self.sleep_fn(compute_backoff_seconds(attempt, self.retry_policy, retry_after))
        return last_status

    # --- response parsing ------------------------------------------------------------

    def _parse_or_status(
        self, response: HttpResponse, query: WebEvidenceQuery, retrieved_at: str
    ) -> tuple[list[dict], list[str]] | str:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            return "provider_error"
        if not isinstance(body, dict):
            return "provider_error"

        top_level_error = body.get("error")
        if top_level_error:
            return _classify_error_message(str(top_level_error))

        metadata = body.get("search_metadata")
        if isinstance(metadata, dict) and metadata.get("status") == "Error":
            return "provider_error"

        raw_results = body.get("organic_results")
        if raw_results is None or not isinstance(raw_results, list):
            return "provider_error"  # unexpected shape -- not even an empty list present

        return self._normalize_results(raw_results, self._effective_language(query), retrieved_at)

    def _normalize_results(self, raw_results: list, language: str, retrieved_at: str) -> tuple[list[dict], list[str]]:
        items: list[dict] = []
        warnings: list[str] = []
        seen_urls: set[str] = set()

        for raw in raw_results:
            if not isinstance(raw, dict):
                warnings.append("skipped a malformed (non-object) result entry")
                continue
            raw_link = raw.get("link")
            raw_title = raw.get("title")
            raw_snippet = raw.get("snippet")
            if not raw_link or not raw_title or not raw_snippet:
                warnings.append("skipped a result missing link/title/snippet")
                continue

            canonical_url = sanitize_url(str(raw_link))
            if canonical_url is None:
                warnings.append("skipped a result with an invalid or unsafe URL")
                continue
            if canonical_url in seen_urls:
                continue  # deterministic de-duplication: keep only the first (highest-ranked) occurrence
            seen_urls.add(canonical_url)

            title, title_truncated = _truncate(str(raw_title), MAX_TITLE_LENGTH)
            snippet, snippet_truncated = _truncate(str(raw_snippet), MAX_SNIPPET_LENGTH)
            if title_truncated:
                warnings.append(f"truncated an over-length title for {canonical_url}")
            if snippet_truncated:
                warnings.append(f"truncated an over-length snippet for {canonical_url}")

            hostname = urllib.parse.urlsplit(canonical_url).hostname

            items.append({
                "title": title,
                "canonical_url": canonical_url,
                "publisher": _bare_host(hostname) if hostname else "unknown",
                "published_at": _parse_supplied_date(raw.get("date")),
                "retrieved_at": retrieved_at,
                "snippet": snippet,
                "language": language,  # the request's own hl -- an honest echo, never a per-result detected value
                "rank": len(items) + 1,
                "freshness": "live",
                "source_type": classify_source_type(hostname),
                "provenance": {
                    "schema_version": "1.0.0",
                    "provider": self.provider_name,
                    "data_mode": "live",
                    "retrieved_at": retrieved_at,
                    "source_urls": [canonical_url],
                },
            })
            if len(items) >= HARD_MAX_RESULTS:
                break

        return items, warnings

    # --- envelope construction ------------------------------------------------------

    def _build_envelope(
        self, query: WebEvidenceQuery, status: str, data_mode: str, retrieved_at: str,
        items: list[dict], warnings: list[str],
    ) -> dict:
        fingerprint = query.fingerprint()
        hl = self._effective_language(query)
        assumptions = [
            f"SerpApi request executed via Google engine (hl={hl!r}, gl={DEFAULT_GL!r}, location={DEFAULT_LOCATION!r})."
        ]
        assumptions.extend(warnings)
        if status != "success":
            assumptions.append(f"SerpApi call ended with status={status!r}.")
        # A whitespace-only/empty query normalizes to "" -- WebEvidenceResult
        # requires both query fields non-empty even on a degraded envelope,
        # so an honest placeholder is used rather than ever emitting an
        # empty string.
        original_query = query.query if query.query.strip() else "(empty query)"
        normalized_query = normalize_query(query.query) or "(empty query)"
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": deterministic_request_id(fingerprint, retrieved_at),
            "provider": self.provider_name,
            "capability": CAPABILITY,
            "data_mode": data_mode,
            "status": status,
            "query_fingerprint": fingerprint,
            "cache_status": "bypass",  # this adapter implements no cache layer of its own -- reported honestly
            "retrieved_at": retrieved_at,
            "source_urls": [item["canonical_url"] for item in items],
            "quality": {
                "schema_version": "1.0.0",
                "completeness": completeness_for_status(status),
                "freshness": data_mode,
                "assumptions": assumptions,
            },
            "result": {
                "schema_version": RESULT_SCHEMA_VERSION,
                "original_query": original_query,
                "normalized_query": normalized_query,
                "items": items,
            },
        }

    def _degraded_envelope(self, query: WebEvidenceQuery, status: str, retrieved_at: str, items: list[dict]) -> dict:
        data_mode = data_mode_for_status(status)
        envelope = self._build_envelope(query, status=status, data_mode=data_mode, retrieved_at=retrieved_at, items=items, warnings=[])
        validate_envelope(envelope)
        return envelope
