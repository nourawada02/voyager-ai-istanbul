"""Tests for language handling across the corpus/chunk/ground-truth
pipeline -- every document, chunk, and ground-truth question carries an
explicit, valid language tag, and the three required languages are all
represented."""

from __future__ import annotations

from rag.chunking import SimpleWhitespaceTokenizer, chunk_document
from rag.ground_truth import QUESTIONS
from rag.ingest import load_documents

_VALID_LANGUAGES = {"en", "tr", "ar"}


def test_every_loaded_document_has_a_valid_language_tag():
    docs = load_documents()
    for doc in docs:
        assert doc.language in _VALID_LANGUAGES, doc.source_id


def test_all_three_languages_are_represented_in_the_corpus():
    docs = load_documents()
    languages = {d.language for d in docs}
    assert languages == _VALID_LANGUAGES


def test_every_ground_truth_question_has_a_valid_language_tag():
    for q in QUESTIONS:
        assert q.language in _VALID_LANGUAGES, q.question_id


def test_arabic_text_chunks_without_corruption():
    tok = SimpleWhitespaceTokenizer()
    text = "# القسم الأول\n\nهذا نص تجريبي باللغة العربية للتحقق من التقطيع الصحيح."
    chunks = chunk_document(text, "عنوان", tok, chunk_tokens=50, overlap_tokens=5)
    assert len(chunks) == 1
    assert "العربية" in chunks[0].text


def test_turkish_special_characters_survive_chunking():
    tok = SimpleWhitespaceTokenizer()
    text = "# Bölüm\n\nÜsküdar, Beşiktaş ve Kadıköy İstanbul'un ilçeleridir."
    chunks = chunk_document(text, "Başlık", tok, chunk_tokens=50, overlap_tokens=5)
    assert "Üsküdar" in chunks[0].text
    assert "Kadıköy" in chunks[0].text
