"""RAG-FIRST SYSTEM B R.1 CORRECTION: cross-repo regression test.

Guards against the exact bug found via live testing during this
correction pass: 20 corpus documents (an entire batch of Arabic/Turkish/
official-source additions) were appended to rag/corpus_sources.py but
never added to services/istanbul-expert-b/phase4/knowledge/
citation_registry.py's own frozen, hand-copied provenance table. Every
chunk from those documents then silently had `url=None`/`title=None`
downstream (phase4.knowledge.qdrant_client.RetrievedChunk resolves
title/url via that registry, never from Qdrant payload), which made
phase4.rag_candidates._validate_and_build reject every one of them --
a real, high-impact defect that would never surface from either
package's own test suite in isolation, since rag/tests/ never imports
System B and phase4/tests/ never imports the real root corpus.

System B's own container never imports the root `rag` package at
runtime (this is a root-owned, dev/test-only cross-check, exactly the
same convention phase4.poi_catalog._try_import_rag_poi_catalog already
uses for the reverse direction)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ISTANBUL_EXPERT_B_ROOT = Path(__file__).resolve().parents[2] / "services" / "istanbul-expert-b"


def _citation_registry_by_source_id():
    if not _ISTANBUL_EXPERT_B_ROOT.exists():
        pytest.skip("services/istanbul-expert-b submodule not present in this checkout")
    if str(_ISTANBUL_EXPERT_B_ROOT) not in sys.path:
        sys.path.insert(0, str(_ISTANBUL_EXPERT_B_ROOT))
    from phase4.knowledge.citation_registry import by_source_id

    return by_source_id


def test_every_corpus_document_resolves_in_system_b_citation_registry():
    from rag.corpus_sources import DOCUMENTS

    by_source_id = _citation_registry_by_source_id()
    missing = [d.source_id for d in DOCUMENTS if by_source_id(d.source_id) is None]
    assert not missing, f"documents missing from phase4.knowledge.citation_registry: {missing}"


def test_registry_entries_match_corpus_title_url_language():
    from rag.corpus_sources import DOCUMENTS

    by_source_id = _citation_registry_by_source_id()
    mismatches = []
    for doc in DOCUMENTS:
        meta = by_source_id(doc.source_id)
        if meta is None:
            continue  # already reported by the completeness test above
        if meta.title != doc.title or meta.url != doc.url or meta.language != doc.language:
            mismatches.append(doc.source_id)
    assert not mismatches, f"registry entries out of sync with corpus_sources.py: {mismatches}"
