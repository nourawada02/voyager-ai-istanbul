"""Injectable HTTP transport (Checkpoint Phase 4 C.1, extended C.2). Keeps
real network I/O behind a small `HttpTransport` Protocol so every provider
adapter -- the Open-Meteo weather adapter, the SerpApi web-evidence
adapter, and any future real adapter -- can be exercised hermetically with
`FakeHttpTransport`, which never opens a socket. `UrllibHttpTransport` is
the only real implementation, built entirely on the standard library
(`urllib.request`) so `providers/` continues to need zero third-party HTTP
dependency.

`get()` accepts an optional, keyword-only `secret_params` mapping kept
structurally separate from `params`: a credential (e.g. SerpApi's
query-parameter `api_key`) passed this way is merged into the request URL
only inside the final request construction in `UrllibHttpTransport.get()`,
and is never recorded in `FakeHttpTransport.call_log` or any other logged
structure -- only the *names* of secret parameter keys are ever recorded
(`FakeHttpTransport.secret_param_keys_log`), never their values.

Timeout semantics (honest statement, corrected Checkpoint Phase 4 C.2):
the `timeout` argument is a `(connect_timeout_seconds, total_timeout_seconds)`
tuple, kept as a pair because `providers.policy.TimeoutPolicy` and every
existing provider already construct and pass it that way. `urllib.request`,
however, has no built-in way to bound the connect phase separately from
the read phase -- `urlopen(..., timeout=...)` accepts exactly one number,
applied as a single socket timeout covering DNS resolution, connect, and
every subsequent read together. `UrllibHttpTransport._send` used to pass
the tuple's first (connect-only) element to `urlopen`, which silently
under-timed the whole call on a slow-but-live connection (a real,
observed failure mode against SerpApi: a connection that completed in
~3.2s on a warm attempt could not even finish DNS+connect within a 3.0s
bound on a cold one). `_send` now passes the tuple's second (total/I/O)
element instead, since that is the number that actually has to cover the
entire request for a timeout to mean what it claims to mean. The first
element is accepted for interface compatibility with every existing
provider but is not separately enforced by this transport.
"""

from __future__ import annotations

import json as _json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol


class TransportError(RuntimeError):
    """Raised for a genuine transport-level failure (connection refused,
    DNS failure, timeout) -- never for a well-formed HTTP error response,
    which is returned as a normal HttpResponse with its real status code
    instead. The message names only the fixed hostname, never a full
    request URL -- a secret query parameter must never appear in an
    exception message."""


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str]

    def json(self) -> Any:
        return _json.loads(self.body.decode("utf-8"))

    def header(self, name: str) -> Optional[str]:
        lower = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lower:
                return value
        return None


class HttpTransport(Protocol):
    def get(
        self,
        url: str,
        params: Mapping[str, Any],
        timeout: tuple[float, float],
        *,
        secret_params: Optional[Mapping[str, Any]] = None,
    ) -> HttpResponse: ...


# Fixed, explicit allowlist of hosts this transport will ever contact --
# user-controlled input (e.g. a free-text location or search query) is
# never used to select a host or build an arbitrary URL. A
# caller-controlled `url` argument here must always be one of the fixed
# endpoint constants each real adapter module declares, enforced by the
# caller (providers.weather_openmeteo, providers.web_evidence_serpapi),
# not by this generic transport, which is deliberately reusable for
# future providers too.
ALLOWED_HOSTS = frozenset({"geocoding-api.open-meteo.com", "api.open-meteo.com", "serpapi.com"})

_DEFAULT_HEADERS = {"User-Agent": "voyager-ai-istanbul/1.0"}


class UrllibHttpTransport:
    """Real transport. Refuses to contact any host outside ALLOWED_HOSTS
    -- a defense-in-depth check independent of whatever URL a caller
    constructs."""

    def get(
        self,
        url: str,
        params: Mapping[str, Any],
        timeout: tuple[float, float],
        *,
        secret_params: Optional[Mapping[str, Any]] = None,
    ) -> HttpResponse:
        parsed = urllib.parse.urlsplit(url)
        self._check_allowed(parsed.hostname)

        merged: dict[str, Any] = {k: v for k, v in params.items() if v is not None}
        if secret_params:
            # Merged into the outgoing URL only here, at the very last
            # step before the request is built -- never stored back into
            # `params`, never logged, never part of a fingerprint.
            merged.update({k: v for k, v in secret_params.items() if v is not None})
        query = urllib.parse.urlencode(merged)
        full_url = f"{url}?{query}" if query else url
        request = urllib.request.Request(full_url, headers=dict(_DEFAULT_HEADERS))
        return self._send(request, parsed.hostname, timeout)

    @staticmethod
    def _check_allowed(hostname: Optional[str]) -> None:
        if hostname not in ALLOWED_HOSTS:
            raise TransportError(f"refusing to contact disallowed host {hostname!r}")

    @staticmethod
    def _send(request: urllib.request.Request, hostname: Optional[str], timeout: tuple[float, float]) -> HttpResponse:
        # urllib.request.urlopen's own `timeout` parameter is a single
        # end-to-end socket timeout -- it does not distinguish a connect
        # phase from a read phase, so the tuple's first element cannot be
        # separately enforced here (see this module's own docstring).
        # Only the total/I/O bound is passed to urlopen, since that is
        # the value that must actually cover the whole call.
        _connect_timeout, io_timeout = timeout
        try:
            with urllib.request.urlopen(request, timeout=io_timeout) as response:
                body = response.read()
                headers = dict(response.headers.items())
                return HttpResponse(status_code=response.status, body=body, headers=headers)
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp else b""
            headers = dict(exc.headers.items()) if exc.headers else {}
            return HttpResponse(status_code=exc.code, body=body, headers=headers)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(f"transport failure contacting {hostname!r}") from exc


@dataclass
class FakeHttpTransport:
    """Deterministic fake transport for hermetic tests -- never opens a
    socket. `responses` maps a URL prefix (matched via startswith) to
    either a fixed HttpResponse or a callable(params) -> HttpResponse,
    so a test can return different bodies for different calls, or
    simulate a Retry-After header. `raise_transport_error`, if set, makes
    every call raise TransportError instead (simulating a genuine
    connection failure).

    `secret_params` passed to `get()` is deliberately never recorded in
    `call_log` -- only the set of its key *names* is recorded in
    `secret_param_keys_log`, so a test can assert a credential was routed
    through the secret channel without ever storing its value anywhere,
    including in a hermetic test's own in-memory fixture state."""

    responses: dict[str, HttpResponse | Callable[[Mapping[str, Any]], HttpResponse]] = field(default_factory=dict)
    raise_transport_error: bool = False
    call_log: list[tuple[str, dict]] = field(default_factory=list)
    secret_param_keys_log: list[frozenset[str]] = field(default_factory=list)

    def get(
        self,
        url: str,
        params: Mapping[str, Any],
        timeout: tuple[float, float],
        *,
        secret_params: Optional[Mapping[str, Any]] = None,
    ) -> HttpResponse:
        self.call_log.append((url, dict(params)))
        self.secret_param_keys_log.append(frozenset(secret_params.keys()) if secret_params else frozenset())
        if self.raise_transport_error:
            raise TransportError("simulated transport failure")
        for prefix, response in self.responses.items():
            if url.startswith(prefix):
                return response(params) if callable(response) else response
        raise AssertionError(f"FakeHttpTransport has no configured response for url={url!r}")
