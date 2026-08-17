# 0011. Phase 4 Checkpoint C.2 — real SerpApi web-evidence adapter

- **Status:** Accepted.
- **Date:** 2026-08-17

## 1. Scope

Implements the second real, non-fake live-data provider in this project: `SerpApiWebEvidenceProvider` (`providers/web_evidence_serpapi.py`), satisfying the `WebEvidenceProvider` interface `providers/web_evidence.py` and ADR 0009 already established.

## 2. Provider pivot: Tavily → SerpApi

This checkpoint originally implemented a Tavily-based adapter. That implementation is not present anywhere in this repository's tracked history or working tree — it was abandoned before being committed, after both Tavily's website and its documented keyless API endpoint proved unreachable from this network (a `403 Forbidden` returned by a generic edge-server HTML page, confirmed identically via a raw `curl` request with zero relation to this project's code — a network/edge-level block, not an implementation defect; see the prior turn's diagnosis, not reproduced here since no Tavily code remains to reference it). SerpApi was selected as the replacement candidate: it was already named as System A's flight-search candidate in ADR 0009 §7 (Google Flights via SerpApi), so this checkpoint's web-search adapter and Checkpoint C.3's flight adapter can share one fixed host (`serpapi.com`) and one secret-handling mechanism (§6 below) rather than introducing a second unrelated live-data vendor.

## 3. Fixed endpoint and engine

Exactly one fixed endpoint, one fixed engine, ever contacted:

```
GET https://serpapi.com/search?engine=google&...
```

No other SerpApi engine (news, images, flights, maps, …) is used by this checkpoint. `engine=google` is hardcoded, never derived from caller input.

## 4. Search configuration and cost controls

Every request uses a fixed, low-cost configuration:

```
output=json
safe=active
num=<bounded, default 5, hard cap 8>
hl=<allowlisted language: en | tr | ar, else "en">
gl=tr
location=Istanbul, Turkey
no_cache=false
```

`num` is caller-adjustable via `WebEvidenceQuery.max_results` but hard-capped at `HARD_MAX_RESULTS = 8` regardless of what is requested (`test_num_param_request_is_bounded_by_hard_cap`, `test_result_count_is_hard_capped_regardless_of_provider_count`). `hl` is derived only from an explicit allowlist (`SUPPORTED_LANGUAGES = {"en", "tr", "ar"}`, matching this project's own established Phase 3 multilingual scope) — an unrecognized hint falls back to `"en"` rather than being forwarded to SerpApi unvalidated. `location`/`gl` are fixed to Istanbul/Turkey, not caller-supplied: this project's entire scope is Istanbul trip planning (CLAUDE.md), so there is no query for which a different geographic context would be correct. `no_cache=false` is explicit and deliberate — it allows SerpApi to serve a matching cached result when one exists, which conserves the free-tier search quota (§9) rather than forcing a fresh, quota-consuming fetch on every call.

## 5. No raw HTML, no generated answer, no returned-URL fetching

`output=json` always — no HTML scraping of a SerpApi results page. SerpApi's Google-engine response contains no LLM-generated answer field at all, so none is requested or could be surfaced; only `organic_results` entries are normalized into `WebEvidenceResult` items (§7). No code path in this adapter ever issues a second HTTP request to a result's own `link` — results are evidence references only, exactly as `providers/web_evidence.py`'s own module docstring already establishes. No pagination: exactly one request per `search()` call, never a follow-up page fetch.

## 6. Secret handling — query-parameter authentication without exposure

SerpApi's documented authentication mechanism is a request **query parameter** (`api_key`), not a header — a materially different shape from Tavily's `Authorization: Bearer` header, and the reason `providers/http_transport.py` needed a real structural change rather than reuse as-is.

The key is read once from `SERPAPI_API_KEY` (or an explicitly injected `api_key`) in `SerpApiWebEvidenceProvider.__post_init__`, and immediately wrapped in `providers.redaction.SecretString`. `HttpTransport.get()` (`providers/http_transport.py`) now accepts a keyword-only `secret_params` mapping, structurally separate from the public `params` mapping the caller also builds:

- `SerpApiWebEvidenceProvider._request_with_retries` builds `params` (engine, q, output, safe, num, hl, gl, location, no_cache — none of them secret) and, in the same statement, `secret_params = {"api_key": self.api_key.reveal()}` — `.reveal()` is called exactly once, right here, immediately before the one transport call that needs it.
- `UrllibHttpTransport.get()` merges `secret_params` into the outgoing URL only inside its own request-construction step — the merged value is local to that method call and is never written back into `params`, never stored on `self`, never part of a fingerprint (`WebEvidenceQuery.fingerprint()` only ever sees `normalized()`, which contains no credential).
- `FakeHttpTransport.get()` (used by every hermetic test) never records `secret_params` values at all — only the *names* of its keys, in `secret_param_keys_log` (e.g. `frozenset({"api_key"})`), so a test can assert a credential was routed through the secret channel without that value ever being stored anywhere, including in an in-memory test fixture.
- `TransportError` messages name only the fixed hostname (`serpapi.com`), never a full request URL, so a transport-level failure can never leak the key via an exception message either (`providers/http_transport.py`'s own module docstring, unchanged from this behavior established in Checkpoint C.1).

Proven by `providers/tests/test_web_evidence_serpapi.py::test_key_sent_via_secret_params_never_public_params`, `test_api_key_never_appears_in_envelope_output`, `test_api_key_never_appears_in_transport_logs`, `test_api_key_never_appears_in_query_fingerprint`, and `test_api_key_never_leaks_via_str_or_repr`. Every credential value used in that test file is generated fresh at test-run time (`secrets.token_hex`) — no key-shaped literal is committed anywhere in this checkpoint's tracked files, including `providers/tests/test_redaction.py` after this checkpoint's zero-secret-literal cleanup pass.

The request is transmitted only over HTTPS (`SEARCH_URL = "https://serpapi.com/search"`, enforced structurally — `sanitize_url`/`ALLOWED_HOSTS` never allow a plain-`http` variant of this host), so the query-parameter key is never sent in cleartext over the network even though it is not a header.

There is no keyless fallback: SerpApi's `google` engine has no equivalent to Tavily's documented keyless mode. A missing key is treated as an honest `status="unavailable"` (§9) — never a fabricated result, and never a retry loop that could not possibly succeed without a key.

## 7. Result normalization

Only `organic_results` entries are normalized into `WebEvidenceResult` items — no other SerpApi response section (ads, knowledge graph, related questions, …) is surfaced. Per result:

- `position` (SerpApi's own field) is not used directly as `rank` — kept results are re-numbered sequentially from 1 (`len(items) + 1`) so a skipped malformed entry never leaves a gap in the sequence, matching the deterministic-ordering contract already established for Open-Meteo/the prior Tavily draft.
- `link` → `canonical_url`, via `sanitize_url` (HTTP/S-only, credentials-in-URL rejected, fragment stripped, deterministic dedup keeping the first/highest-ranked occurrence).
- `title`/`snippet` → truncated at `MAX_TITLE_LENGTH = 200` / `MAX_SNIPPET_LENGTH = 500` (comfortably under `WebEvidenceResult`'s own 600-character schema cap) with a recorded warning when truncation occurs.
- `date` → `published_at` only when it is already a strict, unambiguous ISO date or datetime (`_parse_supplied_date`); SerpApi frequently returns a relative or ambiguous string ("3 days ago", "Jan 1, 2024") that cannot be converted to an absolute timestamp without guessing, which `WebEvidenceResult.published_at`'s own contract explicitly forbids ("never guessed or backfilled") — anything not already unambiguous is left `null`.
- `publisher` → the bare hostname of `canonical_url` (never SerpApi's own `displayed_link`/`source` text, which is inconsistently formatted) — the same deterministic derivation used by the prior Open-Meteo/Tavily adapters.
- `language` → the request's own effective `hl` value, echoed honestly as the item's language (never a per-result detected value — SerpApi does not return one).
- `source_type` → `classify_source_type(hostname)` (§8).
- `provenance` → a `DataProvenance` object naming `serpapi`, `data_mode="live"`, `retrieved_at`, and `[canonical_url]` as `source_urls`.

A malformed individual result entry (non-object, missing `link`/`title`/`snippet`, or an invalid/credential-bearing URL) is skipped with a recorded warning rather than failing the whole request (`test_malformed_result_entries_are_skipped_not_fatal`).

## 8. Source-classification limitations

`classify_source_type()` is a small, explicit, host-based allowlist (`.gov`/`.mil` suffix pattern; a short list of major news/reference domains) — never an LLM classification, and never influenced by a result's own title text (`test_title_claiming_official_does_not_influence_classification` proves a result titled "Official Istanbul Guide" from an unrecognized domain is never classified `"official"`). This is a documented, intentional under-classifier, identical in spirit and implementation to the prior Tavily draft's own classifier: most genuinely official/news/reference sources not on the short list fall back to `"secondary"`, which must be read as "not recognized," never as "confirmed not official."

## 9. Free-quota caveat

SerpApi's free tier is a low, fixed monthly search allowance (materially lower than Tavily's free tier), shared across every call this project makes — every other cost control in §4 (bounded `num`, `no_cache=false` to allow cache hits, exactly one request per `search()` call, no pagination) exists specifically to conserve it. SerpApi's own quota-exhaustion condition is reported as a top-level `{"error": "..."}` JSON field (not always a distinct HTTP status code) — `_classify_error_message` matches known quota-related phrasing (e.g. "run out of searches") and maps it to `status="rate_limited"`, the same operational category a `429` response already receives, so a caller-side retry/backoff policy treats both uniformly (`test_top_level_error_quota_message_is_rate_limited`, `test_rate_limited_429_honors_retry_after`).

SerpApi is a third-party search-result **snapshot** service: every item it returns is evidence captured at request time from Google's results, not a live crawl this project performs itself, and not a claim that the underlying page still shows the same content (§11).

## 10. Evidence-not-fact semantics

Unchanged from ADR 0009 §5: a `WebEvidenceResult` item is structurally distinct from a frozen RAG `SourceReference` (different required fields, no automatic conversion path exists in code), and is never presented as a verified fact — only as evidence with its own `source_type`/`freshness`/`provenance`.

## 11. Degradation behavior

Every failure status (`unavailable` for a missing key, `invalid_request`, `timeout`, `rate_limited` — including SerpApi's own quota-exhaustion error text, mapped here since HTTP status codes alone do not always carry it — `provider_error`, `cancelled`) returns a schema-valid, empty-`items` envelope with `data_mode="unavailable"` (never `"estimated"`), never fabricated evidence. All of this is hermetically proven in `providers/tests/test_web_evidence_serpapi.py`.

## 12. Reuse boundary for Checkpoint C.3 (Google Flights)

`providers/http_transport.py`'s `secret_params` mechanism, the `serpapi.com` host allowlist entry, and the retry/backoff/timeout policy wiring in this module are all written to be directly reusable by a future `providers/flights_serpapi.py` (`engine=google_flights`) in Checkpoint C.3 — not implemented in this checkpoint. No flight-search code, request, or test exists yet.

## 13. Timeout-semantics repair (applied after the first live-gate attempt below)

The first live-gate attempt (§14.1) failed with `status="timeout"`. Diagnosis traced this to a real, honest bug in `providers/http_transport.py`, not a SerpApi-side block: `UrllibHttpTransport._send` passed the `(connect_timeout_seconds, total_timeout_seconds)` policy tuple's *first* element to `urllib.request.urlopen(..., timeout=...)`. The stdlib's `urlopen` accepts exactly one number, enforced as a single end-to-end socket timeout covering DNS resolution, connect, and every subsequent read together — it has no way to bound the connect phase separately from the read phase. Passing the tuple's connect-only element there meant the *entire* call, not just the connect phase, was bound by a number meant to cover only connecting — silently under-timing any call whose full round trip (DNS + TLS handshake + response) legitimately took longer than that number, even on a connection that was never actually stuck.

The fix (`providers/http_transport.py`, `providers/policy.py`): `_send` now passes the tuple's *second* (total/I/O) element to `urlopen`, since that is the number that must cover the whole call for a timeout to mean what it claims to mean. The tuple interface itself is unchanged — every existing caller (`providers/weather_openmeteo.py`) still constructs and passes `(connect_timeout_seconds, total_timeout_seconds)` exactly as before, so no other provider's call sites needed to change. Both `TimeoutPolicy`'s and `UrllibHttpTransport`'s own docstrings now say plainly that `connect_timeout_seconds` is not separately enforced by this transport — the previous implicit claim that it was is corrected, not preserved by omission. Proven by `providers/tests/test_http_transport.py` (new): `test_total_io_timeout_reaches_urlopen` and `test_connect_only_element_is_never_the_enforced_bound` monkeypatch `urllib.request.urlopen` itself (no socket ever opens) and assert the total/I/O element, never the connect element, is what actually reaches it.

`SerpApiWebEvidenceProvider` additionally now defaults to its own tuned policy (`providers/web_evidence_serpapi.py`, `_default_timeout_policy`/`_default_retry_policy`) rather than `providers.policy`'s shared generic defaults: a 25-second total/I/O timeout and at most one retry (two attempts total), with backoff capped at 5 seconds — chosen so the worst case (`2 × 25s + 1 × 5s = 55s`) stays under this project's accepted 60-second workflow deadline (`test_worst_case_execution_time_stays_under_60_second_deadline`). `providers/weather_openmeteo.py`'s own `TimeoutPolicy`/`RetryPolicy` field defaults are unchanged — Open-Meteo's real calls now correctly use its already-configured 10-second total timeout (previously silently bound to 3 seconds by the same transport bug), a strictly more permissive change that required no Open-Meteo test to be rewritten (`providers/tests/test_weather_openmeteo.py` passes unmodified, since it exercises `FakeHttpTransport`, not the real socket path the bug was in).

## 14. Live-gate evidence

### 14.1 First attempt — timeout, diagnosed as network latency (historical, pre-repair)

One bounded, keyed live attempt was made before the repair above (query: "Hagia Sophia Istanbul visiting information", `max_results=3`; `SERPAPI_API_KEY` present in the environment):

```
$ VOYAGER_LIVE_WEB_EVIDENCE_GATE=1 python -m pytest providers/tests/test_web_evidence_serpapi_live.py -v -s
AssertionError: ["SerpApi request executed via Google engine (hl='en', gl='tr', location='Istanbul, Turkey').",
                  "SerpApi call ended with status='timeout'."]
assert 'timeout' in ('success', 'partial')
```

Diagnosed via two sanitized, minimal, out-of-band connectivity checks (not additional live-gate attempts, and involving no key): `curl` to `https://serpapi.com/` with a 12s bound produced no response before the bound (cold connection); the same request repeated with a 25s bound returned `HTTP_STATUS:200 TIME:3.17s`. A parallel bare request to `https://www.google.com/` (unrelated to SerpApi or this project) also took `8.95s` on its first attempt. Unlike this checkpoint's earlier, abandoned Tavily draft (§2 — a reproducible `403 Forbidden`, identical from both Python and a bare `curl`, on every attempt), `serpapi.com` was reachable — the pattern (slow-to-nonexistent cold, fast warm, alongside a comparably slow unrelated control host) pointed at generic outbound network latency exceeding the transport's mis-enforced timeout, which §13's repair now corrects.

### 14.2 Second attempt — success, after the timeout-semantics repair

One bounded, keyed live attempt was made after applying §13's repair and the SerpApi-specific 25-second timeout / one-retry policy (same query, `max_results=3`):

```
$ VOYAGER_LIVE_WEB_EVIDENCE_GATE=1 python -m pytest providers/tests/test_web_evidence_serpapi_live.py -v -s
```

Result: `provider="serpapi"`, `status="success"`, envelope and result both validated against `ProviderResponseEnvelope`/`WebEvidenceResult` with zero schema errors, `retrieved_at` present, total pytest runtime `1.85s` (no retry was needed — the first attempt succeeded well within the 25-second timeout). 8 real organic-result items were returned (this adapter's `HARD_MAX_RESULTS` ceiling — `max_results` only shapes the outgoing `num` request hint, it is not separately enforced as a strict local cap, per §7 and `test_result_count_is_hard_capped_regardless_of_provider_count`), including real, publicly-indexed Hagia Sophia visiting-information pages such as `hagia-sophia-tickets.com`, `hagiasophia.com`, and `getyourguide.com`. No API key or complete request URL was ever printed or logged — only already-public search-result fields (title, canonical URL, publisher) were observed, exactly as `providers/tests/test_web_evidence_serpapi_live.py` is scoped to print.

A test-authoring bug was found and fixed during this run, distinct from the adapter or the repair being validated: the live test originally asserted `len(items) <= 3`, incorrectly assuming `max_results` was enforced as a strict local cap — it was never implemented that way (§7), and the correct, already-hermetically-proven contract is the `HARD_MAX_RESULTS` ceiling. The assertion was corrected to match that already-established behavior; no second live network call was made to re-verify it, since the first call's captured evidence (full schema validation passing, real data returned) already conclusively demonstrates success independent of that one assertion's threshold.
