"""Injectable HTTP transport (Checkpoint Phase 4 C.1). Keeps real network
I/O behind a small `HttpTransport` Protocol so every provider adapter --
this checkpoint's Open-Meteo weather adapter, and any future real
adapter -- can be exercised hermetically with `FakeHttpTransport`, which
never opens a socket. `UrllibHttpTransport` is the only real
implementation, built entirely on the standard library (`urllib.request`)
so `providers/` continues to need zero third-party HTTP dependency.
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
    instead."""


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
    def get(self, url: str, params: Mapping[str, Any], timeout: tuple[float, float]) -> HttpResponse: ...


# Fixed, explicit allowlist of hosts this transport will ever contact --
# user-controlled input (e.g. a free-text location string) is never used
# to select a host or build an arbitrary URL. A caller-controlled `url`
# argument here must always be one of these two constants, enforced by
# the caller (providers.weather_openmeteo), not by this generic
# transport, which is deliberately reusable for future providers too.
ALLOWED_HOSTS = frozenset({"geocoding-api.open-meteo.com", "api.open-meteo.com"})


class UrllibHttpTransport:
    """Real transport. Refuses to contact any host outside ALLOWED_HOSTS
    -- a defense-in-depth check independent of whatever URL a caller
    constructs."""

    def get(self, url: str, params: Mapping[str, Any], timeout: tuple[float, float]) -> HttpResponse:
        parsed = urllib.parse.urlsplit(url)
        if parsed.hostname not in ALLOWED_HOSTS:
            raise TransportError(f"refusing to contact disallowed host {parsed.hostname!r}")

        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        full_url = f"{url}?{query}" if query else url
        request = urllib.request.Request(full_url, headers={"User-Agent": "voyager-ai-istanbul/1.0"})
        connect_timeout, _read_timeout = timeout
        try:
            with urllib.request.urlopen(request, timeout=connect_timeout) as response:
                body = response.read()
                headers = dict(response.headers.items())
                return HttpResponse(status_code=response.status, body=body, headers=headers)
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp else b""
            headers = dict(exc.headers.items()) if exc.headers else {}
            return HttpResponse(status_code=exc.code, body=body, headers=headers)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(f"transport failure contacting {parsed.hostname!r}") from exc


@dataclass
class FakeHttpTransport:
    """Deterministic fake transport for hermetic tests -- never opens a
    socket. `responses` maps a URL prefix (matched via startswith) to
    either a fixed HttpResponse or a callable(params) -> HttpResponse,
    so a test can return different bodies for geocoding vs. forecast
    calls, or simulate a Retry-After header. `raise_transport_error`, if
    set, makes every call raise TransportError instead (simulating a
    genuine connection failure)."""

    responses: dict[str, HttpResponse | Callable[[Mapping[str, Any]], HttpResponse]] = field(default_factory=dict)
    raise_transport_error: bool = False
    call_log: list[tuple[str, dict]] = field(default_factory=list)

    def get(self, url: str, params: Mapping[str, Any], timeout: tuple[float, float]) -> HttpResponse:
        self.call_log.append((url, dict(params)))
        if self.raise_transport_error:
            raise TransportError("simulated transport failure")
        for prefix, response in self.responses.items():
            if url.startswith(prefix):
                return response(params) if callable(response) else response
        raise AssertionError(f"FakeHttpTransport has no configured response for url={url!r}")
