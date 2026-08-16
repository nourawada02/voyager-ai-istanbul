"""Hermetic tests for heading-aware recursive token chunking. Uses
SimpleWhitespaceTokenizer -- no model download, no network."""

from __future__ import annotations

from rag.chunking import SimpleWhitespaceTokenizer, chunk_document, split_into_sections

TOK = SimpleWhitespaceTokenizer()


def test_split_into_sections_preserves_headings():
    text = "# Intro\n\nHello world.\n\n# History\n\nSome history text here."
    sections = split_into_sections(text, "Doc Title")
    assert [s.heading_path for s in sections] == [("Doc Title", "Intro"), ("Doc Title", "History")]
    assert "Hello world." in sections[0].text
    assert "Some history text" in sections[1].text


def test_no_heading_falls_back_to_document_title():
    sections = split_into_sections("Just plain text, no headings.", "Plain Doc")
    assert len(sections) == 1
    assert sections[0].heading_path == ("Plain Doc",)


def test_short_section_becomes_exactly_one_chunk():
    text = "# Only Section\n\n" + " ".join(f"word{i}" for i in range(10))
    chunks = chunk_document(text, "Title", TOK, chunk_tokens=350, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].token_count == 10
    assert chunks[0].heading_path == ("Title", "Only Section")


def test_long_section_splits_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    text = "# Long\n\n" + " ".join(words)
    chunks = chunk_document(text, "Title", TOK, chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # stride = 100 - 20 = 80; verify consecutive chunks overlap by exactly
    # the token count implied by the stride (spot-check via char offsets
    # increasing monotonically and overlapping).
    for i in range(len(chunks) - 1):
        assert chunks[i].char_end > chunks[i + 1].char_start  # real overlap in character space
    assert all(c.token_count <= 100 for c in chunks)


def test_chunk_index_is_sequential_within_document():
    text = "# A\n\n" + " ".join(f"a{i}" for i in range(200)) + "\n\n# B\n\n" + " ".join(f"b{i}" for i in range(200))
    chunks = chunk_document(text, "Title", TOK, chunk_tokens=50, overlap_tokens=10)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk_tokens():
    import pytest

    with pytest.raises(ValueError):
        chunk_document("# A\n\ntext", "Title", TOK, chunk_tokens=50, overlap_tokens=50)


def test_chunking_is_deterministic_across_repeated_calls():
    text = "# A\n\n" + " ".join(f"tok{i}" for i in range(500))
    c1 = chunk_document(text, "Title", TOK, chunk_tokens=100, overlap_tokens=20)
    c2 = chunk_document(text, "Title", TOK, chunk_tokens=100, overlap_tokens=20)
    assert [(c.char_start, c.char_end, c.text) for c in c1] == [(c.char_start, c.char_end, c.text) for c in c2]


def test_all_four_predeclared_configs_produce_valid_chunks():
    from rag.ingest import CHUNK_CONFIGS

    text = "# A\n\n" + " ".join(f"tok{i}" for i in range(1500))
    for name, cfg in CHUNK_CONFIGS.items():
        chunks = chunk_document(text, "Title", TOK, cfg["chunk_tokens"], cfg["overlap_tokens"])
        assert len(chunks) >= 1, name
        assert all(c.token_count <= cfg["chunk_tokens"] for c in chunks), name
