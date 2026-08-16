"""Hermetic tests for the deterministic BM25 sparse index."""

from __future__ import annotations

from rag.bm25 import build_index, tokenize


def test_tokenize_is_lowercase_and_unicode_aware():
    assert tokenize("Hagia Sophia was built in 532.") == ["hagia", "sophia", "was", "built", "in", "532"]
    tokens = tokenize("Boğaziçi Köprüsü")
    assert tokens == ["boğaziçi", "köprüsü"]


def test_search_ranks_more_relevant_document_higher():
    ids = ["a", "b", "c"]
    texts = [
        "Hagia Sophia was built by Justinian in 532 AD.",
        "The Grand Bazaar opened in the fifteenth century.",
        "Topkapi Palace housed Ottoman sultans for centuries.",
    ]
    index = build_index(ids, texts)
    results = index.search("Hagia Sophia Justinian", top_k=3)
    assert results[0][0] == "a"


def test_search_is_deterministic():
    ids = ["a", "b"]
    texts = ["repeated word word word", "repeated word word word"]
    index = build_index(ids, texts)
    r1 = index.search("word", top_k=2)
    r2 = index.search("word", top_k=2)
    assert r1 == r2


def test_search_tie_break_is_ascending_chunk_id():
    # Three documents so the shared query terms have a genuinely positive
    # IDF (a term present in every document can score 0 under BM25's IDF
    # formula) -- two of the three tie exactly, one is a clear non-match.
    ids = ["z", "a", "unrelated"]
    texts = [
        "castle fortress tower stone wall",
        "castle fortress tower stone wall",
        "completely different topic about food and cuisine",
    ]
    index = build_index(ids, texts)
    results = index.search("castle fortress tower", top_k=3)
    tied = [r[0] for r in results if r[0] in ("a", "z")]
    assert tied == ["a", "z"]


def test_empty_query_returns_empty_results():
    index = build_index(["a"], ["some text"])
    assert index.search("???", top_k=5) == []


def test_mismatched_lengths_raise():
    import pytest

    with pytest.raises(ValueError):
        build_index(["a", "b"], ["only one text"])
