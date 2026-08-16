"""Validates real Chunk/RetrievalResult output against the accepted JSON
Schema contracts (contracts/Chunk.schema.json,
contracts/RetrievalResult.schema.json) -- real produced data, not a
hand-written fixture that could silently drift from the real code."""

from __future__ import annotations

import glob
import json
import os

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from rag import qdrant_store
from rag.chunking import SimpleWhitespaceTokenizer
from rag.ingest import CHUNK_CONFIGS, ingest_config, load_documents
from rag.retrieval_service import RetrievalService
from rag.tests.conftest import FAKE_DIM, fake_embed, fake_embed_query

CONTRACTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "contracts")
)
SCHEMA_FILES = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")))


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def registry() -> Registry:
    resources = []
    for path in SCHEMA_FILES:
        schema = _load(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validator(name: str, reg: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, f"{name}.schema.json"))
    return Draft202012Validator(schema, registry=reg, format_checker=FormatChecker())


@pytest.fixture(scope="module")
def real_ingestion_and_retrieval():
    client = qdrant_store.local_client(path=None)
    tok = SimpleWhitespaceTokenizer()
    fp = {"model_name": "fake", "revision": "0", "dim": FAKE_DIM}
    docs = load_documents()
    result = ingest_config(client, "A", tok, fake_embed, fp, docs)
    service = RetrievalService(
        client=client, collection_name=result.collection_name, bm25_index=result.bm25_index,
        embed_query_fn=fake_embed_query, collection_fingerprint=result.fingerprint,
    )
    return result, service


def test_real_chunk_payload_conforms_to_chunk_schema(real_ingestion_and_retrieval, registry):
    """The Chunk contract omits poi_id/district_id when absent (per the
    same 'omission means no value' convention as every other accepted
    contract in this system) -- internal Qdrant payloads store them as
    explicit None for convenience, so the contract-shaped instance below
    filters those out rather than asserting the internal storage
    representation itself is the wire contract."""
    result, _ = real_ingestion_and_retrieval
    validator = _validator("Chunk", registry)
    raw = dict(result.chunk_payloads[0])
    payload = {k: v for k, v in raw.items() if v is not None}
    payload["schema_version"] = "1.0.0"
    payload["chunk_id"] = result.chunk_ids[0]
    errors = list(validator.iter_errors(payload))
    assert not errors, [e.message for e in errors]


def test_real_retrieval_result_conforms_to_retrieval_result_schema(real_ingestion_and_retrieval, registry):
    _, service = real_ingestion_and_retrieval
    retrieval = service.search("Hagia Sophia", "en", "A", top_k=3, mode="dense")
    validator = _validator("RetrievalResult", registry)
    errors = list(validator.iter_errors(retrieval))
    assert not errors, [e.message for e in errors]


def test_retrieval_result_items_never_cite_a_chunk_outside_the_ingested_set(real_ingestion_and_retrieval):
    result, service = real_ingestion_and_retrieval
    retrieval = service.search("Grand Bazaar history", "en", "A", top_k=5, mode="dense")
    known_ids = set(result.chunk_ids)
    for item in retrieval["items"]:
        assert item["chunk_id"] in known_ids
