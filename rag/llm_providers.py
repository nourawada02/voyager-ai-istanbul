"""Provider-independent LLM abstraction (Checkpoint Phase 3 §7/§8).

`detect_available_providers()` checks for a configured hosted provider
first (by presence of a known API-key environment variable -- never
printing or logging the value itself, only the provider name), and falls
back to a local Ollama model only if no hosted provider is configured,
exactly per the instruction. Every provider implements the same
`LLMProvider.generate(system, user) -> str` interface, so the grounded-
answer service and the judge never depend on a specific vendor SDK.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Protocol

# Known hosted-provider environment variable names -- checked for
# PRESENCE only; values are never read into a log or report.
_HOSTED_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# Generation and judging use separate default models so the 45-case
# evaluation does not consume one model's entire daily token allowance.
# GROQ_MODEL remains a backwards-compatible override for the generator;
# explicit role-specific environment variables take precedence.
GROQ_GENERATOR_MODEL = os.environ.get(
    "GROQ_GENERATOR_MODEL", os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
)
GROQ_JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL = GROQ_GENERATOR_MODEL

# QwenCloud/Alibaba Model Studio -- OpenAI-compatible Chat Completions.
# Deliberately NOT given a guessed region/base-URL default: Model Studio
# has multiple regional and workspace-specific hosts, and defaulting to
# one could silently send a workspace's traffic (and credentials) to the
# wrong region. QWEN_BASE_URL must be set explicitly; see `_qwen_base_url`.
QWEN_GENERATOR_MODEL = os.environ.get("QWEN_GENERATOR_MODEL", "qwen3.7-flash")
QWEN_MODEL = QWEN_GENERATOR_MODEL

# Client-side pacing + 429 backoff, shared across all GroqProvider calls in
# this process. Kept deliberately separate from generate_json's bounded
# retry budget (which exists for JSON-validity failures): a rate limit is
# a transport-layer condition, not a model-quality one, and must never
# consume the same retry count or be misreported as a structured-
# generation failure.
_GROQ_MIN_INTERVAL_SECONDS = 2.5
_GROQ_MAX_RETRY_SLEEP_SECONDS = 60.0
_groq_last_call_at = [0.0]

# Separate pacing state for Qwen -- an independent provider with its own
# rate limits; must never share Groq's pacing clock or be paced against
# Groq's assumptions.
_QWEN_MIN_INTERVAL_SECONDS = 1.0
_QWEN_MAX_RETRY_SLEEP_SECONDS = 60.0
_qwen_last_call_at = [0.0]

# Bounded retry budget for genuinely transient transport failures
# (connection reset, timeout, TLS handshake failure) -- distinct from,
# and much smaller than, the 429 retry budget: these are not a quota
# signal, just a flaky connection, and a real outage must still fail
# fast rather than hang the evaluation.
_MAX_TRANSIENT_TRANSPORT_ATTEMPTS = 3
_TRANSIENT_TRANSPORT_INITIAL_BACKOFF_SECONDS = 1.0


class StructuredGenerationFailure(RuntimeError):
    """Raised when a structured JSON call never produced valid,
    schema-conformant output within the bounded retry budget. Never
    silently swallowed -- callers must treat this as a real evaluation
    failure (Checkpoint Phase 3 remediation §4/§5), never fabricate a
    result in its place."""


class ProviderRateLimitExceeded(RuntimeError):
    """Raised when a provider asks the client to wait longer than the
    bounded live-evaluation policy permits, or when the bounded 429 retry
    budget is exhausted.

    This is deliberately distinct from ``StructuredGenerationFailure``:
    a transport quota is not malformed model output and must not consume
    the JSON-repair retry budget or be converted into an answer row.
    """

    def __init__(self, retry_after_seconds: float | None, attempts: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.attempts = attempts
        wait = "unknown" if retry_after_seconds is None else f"{retry_after_seconds:.3f}"
        super().__init__(
            "provider rate limit could not be resolved within the bounded policy "
            f"(retry_after_seconds={wait}, attempts={attempts})"
        )


class ProviderTransportError(RuntimeError):
    """Raised when a provider's HTTP transport fails with a genuine
    transient network condition (connection reset, timeout, or a TLS/SSL
    handshake failure) repeatedly, after the bounded transient-retry
    budget (`_MAX_TRANSIENT_TRANSPORT_ATTEMPTS`) is exhausted.

    Distinct from `ProviderRateLimitExceeded` (an explicit HTTP 429 is a
    quota signal, not a network fault) and from
    `StructuredGenerationFailure` (malformed JSON is a model-quality
    problem, not a transport one) -- callers must be able to tell these
    apart to record the right degradation/failure category rather than
    crashing the whole evaluation process (Checkpoint Phase 3
    remediation: transport resilience).

    The message carries only the provider name and attempt count --
    deliberately never the underlying exception's string representation,
    which on some platforms can embed a hostname, socket address, or
    other connection detail.
    """

    def __init__(self, provider_name: str, attempts: int) -> None:
        self.provider_name = provider_name
        self.attempts = attempts
        super().__init__(
            f"provider {provider_name!r} transport failed after {attempts} attempt(s) "
            "due to a transient network condition (connection reset, timeout, or TLS failure)"
        )


def _is_transient_transport_error(exc: BaseException) -> bool:
    """Whether `exc` represents a genuine transient network condition
    safe to retry -- connection reset, timeout, or a TLS/SSL handshake
    failure -- as opposed to a definitive application-level rejection
    (bad request, auth failure, unresolvable host, etc.) that retrying
    cannot fix.

    Handles both a bare exception of one of these types, and one wrapped
    inside a `urllib.error.URLError` (`urlopen` typically wraps a raised
    low-level socket/SSL exception in `URLError(original_exception)`, as
    observed live: `URLError: <urlopen error [WinError 10054] An
    existing connection was forcibly closed by the remote host>`, where
    `.reason` holds the original `ConnectionResetError`).
    """
    if isinstance(exc, (ConnectionResetError, TimeoutError, ssl.SSLError)):
        return True
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
        return _is_transient_transport_error(exc.reason)
    return False


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str) -> str: ...

    def generate_json(
        self,
        system: str,
        user: str,
        validate: Callable[[dict[str, Any]], None],
        max_retries: int = 2,
    ) -> dict[str, Any]: ...


def _decode_last_valid_json_object(
    raw: str,
    validate: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Return the last schema-valid JSON object embedded in ``raw``.

    Some hosted reasoning models emit an intermediate JSON object before
    their final answer even when instructed to return one object only.  A
    greedy ``{.*}`` regex merges those objects and makes ``json.loads`` fail
    with ``Extra data``.  Scan with ``JSONDecoder.raw_decode`` instead and
    select the last object that independently satisfies the caller's schema.

    Text outside the selected object is never trusted or returned.  If no
    object validates, the structured-generation retry loop still treats the
    response as invalid and asks the provider to repair it.
    """
    decoder = json.JSONDecoder()
    valid_objects: list[dict[str, Any]] = []
    last_error: Exception | None = None

    for position, character in enumerate(raw):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw, position)
            if not isinstance(candidate, dict):
                raise ValueError("decoded JSON value is not an object")
            validate(candidate)
            valid_objects.append(candidate)
        except Exception as exc:  # noqa: BLE001 -- continue scanning for the final valid object
            last_error = exc

    if valid_objects:
        return valid_objects[-1]
    if last_error is not None:
        raise ValueError(f"no schema-valid JSON object found: {last_error}") from last_error
    raise ValueError(f"no JSON object found in response: {raw[:200]!r}")


def _bounded_retry_generate_json(
    generate_fn: Callable[[str, str], str],
    system: str,
    user: str,
    validate: Callable[[dict[str, Any]], None],
    max_retries: int,
) -> dict[str, Any]:
    """Provider-independent bounded-retry structured JSON generation:
    prompt-instructed JSON (the system message states the exact required
    shape) plus strict post-hoc `json.loads` + caller-supplied `validate()`
    and a bounded retry loop. Shared by every provider so the retry/error-
    correction contract is identical regardless of vendor. Raises
    StructuredGenerationFailure (never a fabricated/default result) if no
    attempt produces valid JSON passing `validate()` within
    `max_retries + 1` total attempts."""
    last_error: Exception | None = None
    current_user = user
    for _ in range(max_retries + 1):
        try:
            raw = generate_fn(system, current_user)
            return _decode_last_valid_json_object(raw, validate)
        except (ProviderRateLimitExceeded, ProviderTransportError, urllib.error.URLError, TimeoutError):
            # Network/quota/transport failures are not JSON-quality
            # failures. Let the caller (GroundedAnswerService.answer for
            # the generator role, generation_eval.evaluate_one for the
            # judge role) decide the right degradation, rather than
            # consuming this function's own bounded JSON-repair budget
            # on a condition retrying the prompt cannot fix.
            raise
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure triggers a bounded retry
            last_error = exc
            current_user = (
                user
                + "\n\n(Your previous response was not valid: "
                + f"{exc}. Respond again with ONLY the required JSON object, no other text.)"
            )
    raise StructuredGenerationFailure(
        f"no valid structured JSON after {max_retries + 1} attempts; last error: {last_error}"
    )


def _parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse an RFC-compatible Retry-After delta or HTTP date.

    Invalid and negative values return ``None`` so callers use their
    bounded exponential fallback instead of sleeping an arbitrary value.
    """
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        seconds = (retry_at - current).total_seconds()
    if seconds < 0:
        return None
    return seconds


@dataclass(frozen=True)
class OllamaProvider:
    model: str
    base_url: str = OLLAMA_BASE_URL
    name: str = "ollama"
    timeout_seconds: float = 180.0

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Disables extended chain-of-thought for hybrid-reasoning
            # models (e.g. qwen3.5) that support it -- ignored harmlessly
            # by models that don't. Keeps a small local judge/generator
            # fast enough to run 45 real questions in this checkpoint;
            # has no effect on hosted providers, which never take this path.
            "think": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["message"]["content"]

    def generate_json(
        self,
        system: str,
        user: str,
        validate: Callable[[dict[str, Any]], None],
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Deliberately does NOT use Ollama's native grammar-constrained
        `format: <json schema>` decoding mode: empirically tested against
        the pinned qwen3.5:2b model during this checkpoint's remediation
        and found to make the model answer "insufficient"/empty even for
        directly, trivially answerable questions with an exactly-matching
        context chunk -- grammar-constrained decoding measurably broke
        this small model's reasoning. Prompt-instructed JSON was tested
        against the same cases and produced correct, well-formed output
        in ~10-15s per call. `think` stays False in both retries --
        bounded latency, no chain-of-thought, exactly as required; only
        the retry count (never the timeout, never a fallback to a
        different model) grows on failure. See
        `_bounded_retry_generate_json` for the shared retry mechanics."""
        return _bounded_retry_generate_json(self.generate, system, user, validate, max_retries)


def _post_json_with_rate_limit_handling(
    req: urllib.request.Request,
    timeout: float,
    max_attempts: int = 6,
    max_retry_sleep_seconds: float = _GROQ_MAX_RETRY_SLEEP_SECONDS,
    min_interval_seconds: float = _GROQ_MIN_INTERVAL_SECONDS,
    last_call_at: list[float] | None = None,
    provider_name: str = "unknown",
) -> dict[str, Any]:
    """Proactively paces calls to stay under a provider's per-minute rate
    limit, and reactively backs off on 429 responses. Small
    ``Retry-After`` values are honored; values above
    ``max_retry_sleep_seconds`` fail fast so an hours-long provider quota
    cannot masquerade as a hung evaluation. A 429 is a transport-layer
    condition, never a structured generation failure.

    ``min_interval_seconds``/``last_call_at`` default to Groq's own
    pacing constant/state for backward compatibility with every existing
    call site; a different provider (e.g. Qwen) passes its own
    independent pacing state so the two providers' rate limits are never
    conflated or paced against each other's clock.

    Separately, a genuinely transient transport failure (connection
    reset, timeout, TLS handshake failure -- see
    `_is_transient_transport_error`) gets its own small, explicit,
    bounded retry budget (`_MAX_TRANSIENT_TRANSPORT_ATTEMPTS`, at most 3
    attempts total for this call), with exponential backoff capped at
    the same ``max_retry_sleep_seconds`` ceiling used for 429s -- never
    an unbounded or indefinite wait. Any other ``URLError`` (e.g. a
    genuine DNS/host failure) is not considered transient and propagates
    immediately, exactly as before this repair. Exhausting the transient
    budget raises `ProviderTransportError`, tagged with ``provider_name``
    (a fixed, non-secret label such as ``"groq"``/``"qwen"``) and the
    attempt count only -- never the underlying exception's own string
    representation, which could embed a hostname or socket address.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if max_retry_sleep_seconds < 0:
        raise ValueError("max_retry_sleep_seconds must be non-negative")
    pacing_state = last_call_at if last_call_at is not None else _groq_last_call_at

    backoff = 2.0
    transient_attempts = 0
    transient_backoff = _TRANSIENT_TRANSPORT_INITIAL_BACKOFF_SECONDS
    for attempt in range(max_attempts):
        elapsed = time.monotonic() - pacing_state[0]
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)
        pacing_state[0] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Handled first: HTTPError subclasses URLError, so it would
            # otherwise also match the transient-transport except clause
            # below. A non-429 HTTP error (400/401/403/...) is a
            # definitive application-level response, never retried here.
            if exc.code != 429:
                raise
            retry_after_header = exc.headers.get("Retry-After") if exc.headers else None
            parsed_wait = _parse_retry_after(retry_after_header)
            wait_seconds = parsed_wait if parsed_wait is not None else backoff
            if attempt == max_attempts - 1 or wait_seconds > max_retry_sleep_seconds:
                raise ProviderRateLimitExceeded(wait_seconds, attempt + 1) from exc
            time.sleep(wait_seconds)
            backoff = min(backoff * 2, 30.0)
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, ssl.SSLError) as exc:
            if not _is_transient_transport_error(exc):
                raise
            transient_attempts += 1
            if transient_attempts >= _MAX_TRANSIENT_TRANSPORT_ATTEMPTS:
                raise ProviderTransportError(provider_name, transient_attempts) from None
            time.sleep(min(transient_backoff, max_retry_sleep_seconds))
            transient_backoff = min(transient_backoff * 2, max_retry_sleep_seconds)

    raise AssertionError("unreachable rate-limit retry state")


@dataclass(frozen=True)
class GroqProvider:
    """Hosted provider used in preference to local Ollama whenever
    GROQ_API_KEY is present (see `detect_available_providers` -- hosted
    providers are always preferred over the local fallback). Uses the
    same prompt-instructed-JSON + bounded-retry strategy as Ollama rather
    than Groq's native `response_format: json_object` mode, to keep the
    structured-output contract identical and independently verified
    across providers rather than trusting an unverified provider-specific
    fast path."""

    model: str = GROQ_MODEL
    base_url: str = GROQ_BASE_URL
    name: str = "groq"
    timeout_seconds: float = 60.0

    def generate(self, system: str, user: str) -> str:
        api_key = os.environ.get("GROQ_API_KEY", "")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                # Groq's edge rejects the default `Python-urllib/x.y`
                # User-Agent with a 403 -- any ordinary-looking UA works.
                "User-Agent": "voyager-ai-istanbul-rag/1.0",
            },
            method="POST",
        )
        body = _post_json_with_rate_limit_handling(req, timeout=self.timeout_seconds, provider_name=self.name)
        return body["choices"][0]["message"]["content"]

    def generate_json(
        self,
        system: str,
        user: str,
        validate: Callable[[dict[str, Any]], None],
        max_retries: int = 2,
    ) -> dict[str, Any]:
        return _bounded_retry_generate_json(self.generate, system, user, validate, max_retries)


class QwenConfigurationError(RuntimeError):
    """Raised when QwenProvider is used without a valid, explicit HTTPS
    QWEN_BASE_URL. Never guesses a region/workspace endpoint -- Model
    Studio has multiple regional and workspace-specific hosts, and
    silently defaulting to one could send a workspace's traffic (and
    credentials) to the wrong region. Fails before any network call."""


def _qwen_api_key() -> str:
    """QWEN_API_KEY, falling back to the official DASHSCOPE_API_KEY name
    for compatibility. Read fresh at call time, never cached/stored."""
    return os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""


def _qwen_api_key_present() -> bool:
    """Presence-only check -- the value itself is never read here."""
    return bool(os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))


def _qwen_base_url() -> str:
    base_url = os.environ.get("QWEN_BASE_URL")
    if not base_url:
        raise QwenConfigurationError(
            "QWEN_BASE_URL is required and must be set explicitly -- no region endpoint is guessed"
        )
    if urllib.parse.urlsplit(base_url).scheme != "https":
        raise QwenConfigurationError("QWEN_BASE_URL must be an explicit https:// URL")
    return base_url


@dataclass(frozen=True)
class QwenProvider:
    """QwenCloud/Alibaba Model Studio provider, using its OpenAI-
    compatible Chat Completions endpoint. Selected only through explicit
    role configuration (GEN_EVAL_GENERATOR_PROVIDER=qwen /
    GEN_EVAL_JUDGE_PROVIDER=qwen in rag.run_generation_eval) -- never
    auto-detected alongside Groq/Anthropic/OpenAI in
    `detect_available_providers`, so QWEN_API_KEY being present can never
    silently change which provider serves an unrelated role.

    Uses the same prompt-instructed-JSON + bounded-retry contract as
    every other provider (`_bounded_retry_generate_json`); the API's own
    `response_format: json_object` mode and non-thinking flag are an
    additional safeguard, never a trusted replacement for that shared
    validation.

    Neither the API key nor the base URL (which may contain a workspace-
    specific hostname) are stored as dataclass fields -- both are read
    fresh from the environment inside `generate()`, so this object can
    never accidentally leak either value through introspection,
    serialization, or a log/report built from its fields.
    """

    model: str = QWEN_MODEL
    name: str = "qwen"
    timeout_seconds: float = 60.0

    def generate(self, system: str, user: str) -> str:
        base_url = _qwen_base_url()
        api_key = _qwen_api_key()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            # JSON Object output mode: the API itself enforces
            # syntactically valid JSON. This is an additional layer, not
            # a replacement for _decode_last_valid_json_object's own
            # schema-conformance validation.
            "response_format": {"type": "json_object"},
            # Disables extended reasoning/"thinking" output on Qwen's
            # hybrid-reasoning models where the API supports the
            # parameter; a harmless no-op key otherwise. Mirrors
            # OllamaProvider's `think: False` for the same reason: bounded
            # latency, no chain-of-thought leaking into the JSON response.
            "enable_thinking": False,
        }
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "voyager-ai-istanbul-rag/1.0",
            },
            method="POST",
        )
        body = _post_json_with_rate_limit_handling(
            req,
            timeout=self.timeout_seconds,
            max_retry_sleep_seconds=_QWEN_MAX_RETRY_SLEEP_SECONDS,
            min_interval_seconds=_QWEN_MIN_INTERVAL_SECONDS,
            last_call_at=_qwen_last_call_at,
            provider_name=self.name,
        )
        return body["choices"][0]["message"]["content"]

    def generate_json(
        self,
        system: str,
        user: str,
        validate: Callable[[dict[str, Any]], None],
        max_retries: int = 2,
    ) -> dict[str, Any]:
        return _bounded_retry_generate_json(self.generate, system, user, validate, max_retries)


def _ollama_reachable() -> str | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    models = [m["name"] for m in data.get("models", [])]
    # Prefer a model advertising "completion"/"tools" capability over a
    # pure embedding model -- never silently pick an embedding-only model
    # as a generator/judge.
    for candidate in models:
        if "embed" not in candidate:
            return candidate
    return None


@dataclass(frozen=True)
class ProviderDetectionResult:
    hosted_provider: str | None
    local_ollama_model: str | None

    @property
    def selected(self) -> LLMProvider | None:
        if self.hosted_provider is not None:
            if self.hosted_provider == "groq":
                return GroqProvider()
            raise NotImplementedError(
                f"hosted provider {self.hosted_provider!r} detected but no client is wired in this "
                "checkpoint -- add one before relying on it"
            )
        if self.local_ollama_model is not None:
            return OllamaProvider(model=self.local_ollama_model)
        return None

    @property
    def description(self) -> str:
        if self.hosted_provider is not None:
            return f"hosted provider configured: {self.hosted_provider}"
        if self.local_ollama_model is not None:
            return f"local Ollama fallback: {self.local_ollama_model} (no hosted provider key was found)"
        return "no LLM provider available (no hosted key, Ollama unreachable)"

    @property
    def provider_name(self) -> str | None:
        """Short name for acceptance-gate comparisons: the hosted provider
        name if one was detected, else ``"ollama"`` if a local model is
        reachable, else ``None``. Never derived from or containing a
        secret value -- only ever one of the fixed strings above."""
        if self.hosted_provider is not None:
            return self.hosted_provider
        if self.local_ollama_model is not None:
            return "ollama"
        return None


class RequiredProviderUnavailable(RuntimeError):
    """Raised by `enforce_required_provider` (global) and by the role-
    specific checks in `rag.run_generation_eval`
    (`enforce_required_role_provider`, and provider construction when an
    explicit `GEN_EVAL_GENERATOR_PROVIDER`/`GEN_EVAL_JUDGE_PROVIDER`
    names a provider whose credential is missing) when a required
    provider was not actually selected/available for this run.

    An acceptance run pinned to a specific provider must never silently
    fall back to a different provider (e.g. local Ollama, or the wrong
    role's provider) -- that would produce a report that looks like
    acceptance evidence but was not run against the required model. This
    check must happen before any evaluation row or provider call, never
    after the fact.
    """


def enforce_required_provider(detection: ProviderDetectionResult) -> None:
    """Fail fast, before any provider call, if `GEN_EVAL_REQUIRE_PROVIDER`
    is set and does not match the detected provider.

    Only provider *names* (fixed strings such as "groq" or "ollama") ever
    appear in the raised message -- never an API key or other secret
    value, matching `detect_available_providers`' presence-only check.
    """
    required = os.environ.get("GEN_EVAL_REQUIRE_PROVIDER")
    if not required:
        return
    actual = detection.provider_name
    if actual != required:
        raise RequiredProviderUnavailable(
            f"GEN_EVAL_REQUIRE_PROVIDER={required!r} but the detected provider is "
            f"{actual!r} ({detection.description}); refusing to silently substitute a "
            "different provider for an acceptance run"
        )


def detect_available_providers() -> ProviderDetectionResult:
    """Checks hosted providers first (by env var PRESENCE only, values
    never read/printed here). Only checks Ollama, and only reports a
    local_ollama_model, if no hosted provider key is present -- per the
    instruction that Ollama may be used 'only if no hosted provider
    exists'.

    Deliberately unaware of Qwen: Qwen is reachable only through the
    explicit, role-specific GEN_EVAL_GENERATOR_PROVIDER/
    GEN_EVAL_JUDGE_PROVIDER selection in rag.run_generation_eval, never
    through this single-global-provider auto-detection path. This is
    what guarantees QWEN_API_KEY being present can never, by itself,
    change the outcome of this function or of `GEN_EVAL_REQUIRE_PROVIDER`
    (see `enforce_required_provider`).
    """
    for provider_name, env_var in _HOSTED_PROVIDER_ENV_VARS.items():
        if os.environ.get(env_var):
            return ProviderDetectionResult(hosted_provider=provider_name, local_ollama_model=None)

    ollama_model = _ollama_reachable()
    return ProviderDetectionResult(hosted_provider=None, local_ollama_model=ollama_model)


def provider_credential_present(provider_name: str) -> bool:
    """Presence/reachability-only check for a named provider, used by
    role-specific selection (rag.run_generation_eval) to fail before any
    network call when an explicitly requested provider has no usable
    credential. Never reads a secret's value -- Groq/Qwen are checked by
    environment-variable presence only; Ollama by local reachability."""
    if provider_name == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if provider_name == "qwen":
        return _qwen_api_key_present()
    if provider_name == "ollama":
        return _ollama_reachable() is not None
    return False


def construct_provider(provider_name: str, *, model: str | None = None) -> LLMProvider:
    """Construct a named provider instance for explicit, role-specific
    selection. Construction alone never touches the network or reads a
    secret value -- call `provider_credential_present` first to fail
    closed before this is even called, per the acceptance-run contract
    in rag.run_generation_eval.
    """
    if provider_name == "groq":
        return GroqProvider(model=model) if model else GroqProvider()
    if provider_name == "qwen":
        return QwenProvider(model=model) if model else QwenProvider()
    if provider_name == "ollama":
        resolved_model = model or _ollama_reachable()
        if resolved_model is None:
            raise RequiredProviderUnavailable("provider 'ollama' requested but no local model is reachable")
        return OllamaProvider(model=resolved_model)
    raise NotImplementedError(f"unknown provider {provider_name!r} -- add one before relying on it")
