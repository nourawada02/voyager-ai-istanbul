"""Hermetic tests for deterministic ID/fingerprint helpers."""

from __future__ import annotations

from rag.ids import chunk_id, config_fingerprint, corpus_fingerprint, document_id, ingestion_fingerprint, sha256_hex


def test_document_id_is_the_source_id():
    assert document_id("wiki_en_istanbul") == "wiki_en_istanbul"


def test_chunk_id_is_deterministic():
    a = chunk_id("wiki_en_istanbul", "A", 0)
    b = chunk_id("wiki_en_istanbul", "A", 0)
    assert a == b
    assert a.startswith("chunk_")


def test_chunk_id_differs_by_config_and_index():
    ids = {
        chunk_id("src1", "A", 0),
        chunk_id("src1", "B", 0),
        chunk_id("src1", "A", 1),
        chunk_id("src2", "A", 0),
    }
    assert len(ids) == 4


def test_config_fingerprint_is_order_independent_via_sorted_keys():
    fp1 = config_fingerprint({"a": 1, "b": 2})
    fp2 = config_fingerprint({"b": 2, "a": 1})
    assert fp1 == fp2


def test_config_fingerprint_changes_with_any_value_change():
    fp1 = config_fingerprint({"chunk_tokens": 350, "overlap_tokens": 50})
    fp2 = config_fingerprint({"chunk_tokens": 351, "overlap_tokens": 50})
    assert fp1 != fp2


def test_corpus_fingerprint_is_pure_function_of_checksums():
    checksums = {"doc1": "aaa", "doc2": "bbb"}
    fp1 = corpus_fingerprint(checksums)
    fp2 = corpus_fingerprint(dict(reversed(list(checksums.items()))))
    assert fp1 == fp2


def test_corpus_fingerprint_changes_if_any_checksum_changes():
    fp1 = corpus_fingerprint({"doc1": "aaa"})
    fp2 = corpus_fingerprint({"doc1": "aab"})
    assert fp1 != fp2


def test_ingestion_fingerprint_combines_both_fingerprints():
    a = ingestion_fingerprint("corpusA", "configA")
    b = ingestion_fingerprint("corpusB", "configA")
    c = ingestion_fingerprint("corpusA", "configB")
    assert len({a, b, c}) == 3


def test_sha256_hex_is_stable_and_correct_length():
    h = sha256_hex("hello")
    assert len(h) == 64
    assert h == sha256_hex("hello")
