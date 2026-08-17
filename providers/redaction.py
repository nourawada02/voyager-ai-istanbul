"""Secret isolation and redacted logging (Checkpoint Phase 4 C.0, Step 4;
architecture.md: "No hardcoded secrets anywhere; .env.example committed
with placeholders, real .env never committed"). Extends that existing
project-wide rule to live-data provider adapters specifically, which are
the first components in this project to hold real outbound API keys.
"""

from __future__ import annotations

import re

_SECRET_KEY_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key)\"?\s*[:=]\s*\"?[^\s&\"']+"),
    re.compile(r"(?i)(authorization)\"?\s*:\s*\"?bearer\s+[^\s\"']+"),
    re.compile(r"(?i)(secret)\"?\s*[:=]\s*\"?[^\s&\"']+"),
    re.compile(r"(?i)(token)\"?\s*[:=]\s*\"?[^\s&\"']+"),
    re.compile(r"(?i)(password)\"?\s*[:=]\s*\"?[^\s&\"']+"),
)

_REDACTED = "***REDACTED***"


def redact_secrets(text: str) -> str:
    """Best-effort redaction of common secret-bearing patterns (query
    params, headers, key=value pairs) from a string before it is ever
    logged. Not a substitute for never putting secrets in a log line at
    all -- SecretString below is the structural guarantee for that."""
    redacted = text
    for pattern in _SECRET_KEY_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1)}={_REDACTED}", redacted)
    return redacted


class SecretString:
    """Wraps a credential value so it can be passed around and used to
    make a real call, but can never be accidentally logged, printed, or
    serialized in full: __str__/__repr__ always return a fixed redacted
    marker, and it deliberately does not implement __eq__ against a raw
    string to discourage secret comparison via logs/asserts in tests."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The only way to get the real value back -- callers must name
        this explicitly, so a `str(secret)` or an f-string never leaks it
        by accident."""
        return self._value

    def __str__(self) -> str:  # noqa: D105
        return _REDACTED

    def __repr__(self) -> str:  # noqa: D105
        return f"SecretString({_REDACTED})"
