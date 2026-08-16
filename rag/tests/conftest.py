"""Shared fixtures for rag/tests. Uses the real qdrant-client library in
embedded in-memory mode (a real vector engine, just no network server) --
never a mock of Qdrant itself. Embedding vectors are deterministic
hash-derived fakes for hermetic tests that don't need the real 470MB
pinned E5 model; tests that verify the real model/prefixes/token counts
load it for real (see test_embeddings.py)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from rag import qdrant_store

FAKE_DIM = 16


@dataclass
class StubStructuredProvider:
    """Test double for the `generate_json`-based LLMProvider protocol.
    `responses` is a queue of dicts to return in order (one per call,
    last one repeats if the queue is exhausted); `raise_failure=True`
    makes every call raise StructuredGenerationFailure, simulating the
    'no valid JSON after bounded retries' path."""

    responses: list[dict[str, Any]] = field(default_factory=list)
    raise_failure: bool = False
    name: str = "stub"
    model: str = "stub-model"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.raise_failure:
            return "not valid json at all"
        data = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return json.dumps(data)

    def generate_json(
        self, system: str, user: str, validate: Callable[[dict[str, Any]], None], max_retries: int = 2
    ) -> dict[str, Any]:
        from rag.llm_providers import StructuredGenerationFailure

        if self.raise_failure:
            self.calls.append((system, user))
            raise StructuredGenerationFailure("stub configured to always fail")
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            self.calls.append((system, user))
            data = self.responses[min(attempt, len(self.responses) - 1)]
            try:
                validate(data)
                return data
            except Exception as exc:  # noqa: BLE001 -- mirrors the production bounded validation loop
                last_error = exc
        raise StructuredGenerationFailure(f"stub exhausted validation retries: {last_error}")


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic, dependency-free fake embedding: every text maps to
    the same vector every time, and different texts very likely map to
    different vectors (good enough for hermetic upsert/search mechanics
    tests, never used for the real experiment)."""
    vectors = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vectors.append([b / 255.0 for b in h[:FAKE_DIM]])
    return vectors


def fake_embed_query(text: str) -> list[float]:
    return fake_embed([text])[0]


@pytest.fixture
def qdrant_client_hermetic():
    return qdrant_store.local_client(path=None)
