"""Pinned dense embedding model wrapper (Checkpoint Phase 3,
architecture.md §9.2).

Exactly one model, one immutable revision, ever: `intfloat/multilingual-e5-small`
pinned by commit hash below -- never a silent fallback to a different
model or an unpinned "latest" revision. `passage:` is applied to every
indexed chunk; `query:` is applied to every search query -- required by
E5's own training convention, applied uniformly, never omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
# Exact immutable revision (full git commit hash on the HF model repo,
# not a mutable ref like "main") -- confirmed via
# https://huggingface.co/api/models/intfloat/multilingual-e5-small at
# pin time (2026-08-15); real committed sha, not a placeholder.
EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIM = 384

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@dataclass(frozen=True)
class EmbeddingModelFingerprint:
    model_name: str
    revision: str
    dim: int

    def as_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "revision": self.revision, "dim": self.dim}


def fingerprint() -> EmbeddingModelFingerprint:
    return EmbeddingModelFingerprint(EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION, EMBEDDING_DIM)


class HFTokenizerAdapter:
    """Adapts a real HuggingFace fast tokenizer to the `chunking.Tokenizer`
    protocol (encode_with_offsets), for exact, model-consistent token
    counts during ingestion."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tokenizer = hf_tokenizer

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoding = self._tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return list(encoding["input_ids"]), [tuple(o) for o in encoding["offset_mapping"]]


@lru_cache(maxsize=1)
def load_sentence_transformer():
    """Loads the pinned model exactly once per process. Raises loudly
    (never silently substitutes a different model/revision) if the
    installed sentence-transformers/transformers stack cannot load the
    pinned revision."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION)
    return model


@lru_cache(maxsize=1)
def load_tokenizer() -> HFTokenizerAdapter:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION)
    return HFTokenizerAdapter(tok)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embeds already-chunked passage text. Applies the `passage:` prefix
    to every text -- never omitted, never applied twice."""
    model = load_sentence_transformer()
    prefixed = [PASSAGE_PREFIX + t for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embeds one search query. Applies the `query:` prefix -- never the
    `passage:` prefix, and never left unprefixed."""
    model = load_sentence_transformer()
    vector = model.encode([QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()
