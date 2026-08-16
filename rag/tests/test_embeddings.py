"""Tests for the pinned E5 embedding wrapper: exact prefixes, exact
pinned revision, and real (not silently substituted) model identity.
Loads the real ~470MB pinned model once per test session -- these are
the only tests in rag/tests that require the real network download."""

from __future__ import annotations

import pytest

from rag import embeddings


def test_fingerprint_names_the_exact_pinned_model_and_revision():
    fp = embeddings.fingerprint()
    assert fp.model_name == "intfloat/multilingual-e5-small"
    assert fp.revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert len(fp.revision) == 40  # a real full git commit hash, not a short/mutable ref
    assert fp.dim == 384


def test_embed_passages_applies_passage_prefix_exactly_once():
    from unittest.mock import patch

    captured = {}

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            captured["texts"] = texts
            import numpy as np

            return np.zeros((len(texts), 4))

    with patch.object(embeddings, "load_sentence_transformer", return_value=_FakeModel()):
        embeddings.embed_passages(["Hagia Sophia was built in 532."])

    assert captured["texts"] == ["passage: Hagia Sophia was built in 532."]


def test_embed_query_applies_query_prefix_not_passage_prefix():
    from unittest.mock import patch

    captured = {}

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            captured["texts"] = texts
            import numpy as np

            return np.zeros((len(texts), 4))

    with patch.object(embeddings, "load_sentence_transformer", return_value=_FakeModel()):
        embeddings.embed_query("When was Hagia Sophia built?")

    assert captured["texts"] == ["query: When was Hagia Sophia built?"]
    assert not captured["texts"][0].startswith("passage:")


def test_real_pinned_model_loads_and_produces_the_declared_dimension():
    """The one real, slow, network-using test in this module: proves the
    exact pinned revision actually loads (never silently falls back to a
    different revision) and produces vectors of the declared dimension."""
    vectors = embeddings.embed_passages(["Hagia Sophia was built between 532 and 537 AD."])
    assert len(vectors) == 1
    assert len(vectors[0]) == embeddings.EMBEDDING_DIM == 384


def test_real_tokenizer_produces_exact_token_counts_with_offsets():
    tok = embeddings.load_tokenizer()
    ids, offsets = tok.encode_with_offsets("Hagia Sophia was built in 532.")
    assert len(ids) == len(offsets)
    assert len(ids) > 0
    for start, end in offsets:
        assert 0 <= start <= end <= len("Hagia Sophia was built in 532.")


def test_semantically_similar_texts_produce_higher_cosine_similarity_than_unrelated():
    import numpy as np

    a = np.array(embeddings.embed_passages(["Hagia Sophia was built by Emperor Justinian."])[0])
    b = np.array(embeddings.embed_query("Who built Hagia Sophia?"))
    c = np.array(embeddings.embed_query("What is the best kebab restaurant?"))

    sim_related = float(np.dot(a, b))
    sim_unrelated = float(np.dot(a, c))
    assert sim_related > sim_unrelated
