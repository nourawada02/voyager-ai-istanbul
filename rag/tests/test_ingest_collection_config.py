"""Tests for the official 69-document RAG corpus promotion checkpoint:
configurable collection naming (rag/ingest_cli.py) and the corpus-aware
ingestion fingerprint (rag/ingest.py::compute_fingerprint).

Hermetic where possible. The small number of real-Qdrant-server cases
reuse rag/tests/test_qdrant_real_server.py's own pattern -- a unique
throwaway collection per test, NEVER the legacy
`istanbul_rag_B`/`istanbul_rag_B_r1_shadow` collections and never the
literal `istanbul_rag_B_v2` name either (that exact name is verified
end-to-end separately, against a disposable Qdrant instance -- see this
checkpoint's own verification report, not this repository's shared demo
Qdrant volume)."""

from __future__ import annotations

import argparse
import json
import uuid

import pytest

from rag import embeddings, ingest, ingest_cli, qdrant_store
from rag.chunking import SimpleWhitespaceTokenizer, chunk_document
from rag.ids import corpus_fingerprint, sha256_hex
from rag.tests.conftest import FAKE_DIM, fake_embed
from rag.tests.test_qdrant_real_server import QDRANT_TEST_URL, _get_real_server_client

_LEGACY_COLLECTIONS = {"istanbul_rag_B", "istanbul_rag_B_r1_shadow"}


def _doc(source_id: str, language: str = "en", text: str = "hello world, a short demo fact") -> ingest.LoadedDocument:
    return ingest.LoadedDocument(
        source_id=source_id, title=source_id, language=language, content_type="history",
        poi_id=None, district_id=None, checksum=sha256_hex(text), text=text,
    )


# --- collection-name configuration (hermetic, argparse-level) --------------------------


def test_default_collection_name_is_istanbul_rag_b_v2(monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)
    args = ingest_cli.build_parser().parse_args(["status"])
    assert args.collection_name == "istanbul_rag_B_v2" == ingest_cli.DEFAULT_COLLECTION_NAME


def test_collection_name_env_override_works(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_collection")
    args = ingest_cli.build_parser().parse_args(["status"])
    assert args.collection_name == "test_collection"


def test_explicit_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "from_env")
    args = ingest_cli.build_parser().parse_args(["--collection-name", "from_flag", "status"])
    assert args.collection_name == "from_flag"


def test_status_and_bootstrap_use_the_same_configured_collection_name(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_collection")
    status_args = ingest_cli.build_parser().parse_args(["status"])
    bootstrap_args = ingest_cli.build_parser().parse_args(["bootstrap"])
    assert status_args.collection_name == bootstrap_args.collection_name == "test_collection"


def test_default_collection_name_is_never_a_legacy_collection(monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)
    args = ingest_cli.build_parser().parse_args(["status"])
    assert args.collection_name not in _LEGACY_COLLECTIONS


def test_system_b_and_rag_ingest_share_the_same_default_collection_name(monkeypatch):
    """The two independent repos/services never hardcode divergent
    defaults -- both resolve to the literal 'istanbul_rag_B_v2' string
    (services/istanbul-expert-b/phase4/config.py::DEFAULT_QDRANT_COLLECTION_NAME
    is checked for the identical literal in that submodule's own test
    suite, since this root repo never imports across the submodule
    boundary at runtime or in tests)."""
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)
    args = ingest_cli.build_parser().parse_args(["status"])
    assert args.collection_name == "istanbul_rag_B_v2"


# --- corpus-aware fingerprint (hermetic) ------------------------------------------------


def test_corpus_fingerprint_differs_between_a_small_and_a_large_corpus():
    small = ingest._corpus_identity([_doc("a"), _doc("b")])
    large = ingest._corpus_identity([_doc("a"), _doc("b"), _doc("c"), _doc("d")])
    assert corpus_fingerprint(small) != corpus_fingerprint(large)


def test_20_document_and_69_document_corpora_produce_different_fingerprints(monkeypatch):
    """Simulates the exact historical defect this checkpoint fixes: the
    original 20-document corpus and the expanded 69-document corpus must
    no longer collide on one shared stored fingerprint."""
    all_docs = ingest.load_documents()
    assert len(all_docs) == 69  # the real, current, committed corpus
    twenty = all_docs[:20]
    embedding_fp = {"model_name": "fake", "revision": "0", "dim": 16}

    monkeypatch.setattr(ingest, "load_documents", lambda: twenty)
    fp_20 = ingest.compute_fingerprint("B", embedding_fp)
    monkeypatch.setattr(ingest, "load_documents", lambda: all_docs)
    fp_69 = ingest.compute_fingerprint("B", embedding_fp)
    assert fp_20 != fp_69


def test_any_corpus_content_change_alters_the_fingerprint():
    a = ingest._corpus_identity([_doc("a", text="original text")])
    b = ingest._corpus_identity([_doc("a", text="a single changed word here")])
    assert corpus_fingerprint(a) != corpus_fingerprint(b)


def test_language_only_change_alters_the_fingerprint_even_with_identical_text():
    text = "identical content, different declared language"
    a = ingest._corpus_identity([_doc("a", language="en", text=text)])
    b = ingest._corpus_identity([_doc("a", language="tr", text=text)])
    assert corpus_fingerprint(a) != corpus_fingerprint(b)


def test_chunking_config_change_alters_the_fingerprint():
    embedding_fp = {"model_name": "fake", "revision": "0", "dim": 16}
    assert ingest.compute_fingerprint("A", embedding_fp) != ingest.compute_fingerprint("B", embedding_fp)


def test_embedding_identity_change_alters_the_fingerprint():
    fp_v1 = ingest.compute_fingerprint("B", {"model_name": "fake", "revision": "0", "dim": 16})
    fp_v2 = ingest.compute_fingerprint("B", {"model_name": "fake", "revision": "1", "dim": 16})
    assert fp_v1 != fp_v2


def test_incremental_partial_ingest_does_not_change_the_collections_committed_fingerprint():
    """rag/ingest_cli.py's `ingest --source` path adds a genuinely partial
    document subset -- proves the corpus-aware fingerprint still reflects
    the canonical full corpus, not whatever partial `documents=` a single
    ingest_config() call happens to embed (a real bug this checkpoint's
    own design work caught: an early version of this fix computed the
    fingerprint from `documents` directly and broke incremental add)."""
    embedding_fp = {"model_name": "fake", "revision": "0", "dim": 16}
    canonical_fp = ingest.compute_fingerprint("A", embedding_fp)

    client = qdrant_store.local_client(path=None)
    tok = SimpleWhitespaceTokenizer()
    partial_docs = [_doc("only_one_demo_doc")]
    result = ingest.ingest_config(client, "A", tok, fake_embed, embedding_fp, documents=partial_docs)
    assert result.fingerprint == canonical_fp


# --- official corpus counts (real corpus, real chunking config) ------------------------


def test_corpus_document_count_and_language_distribution_is_exactly_39_14_16():
    docs = ingest.load_documents()
    assert len(docs) == 69
    counts = {"en": 0, "tr": 0, "ar": 0}
    for d in docs:
        counts[d.language] += 1
    assert counts == {"en": 39, "tr": 14, "ar": 16}


def test_corpus_chunk_count_under_config_b_is_exactly_287():
    """287 document chunks + 1 fingerprint marker = 288 total Qdrant
    points -- the exact official istanbul_rag_B_v2 target this checkpoint's
    disposable-Qdrant verification proves live."""
    tok = embeddings.load_tokenizer()  # the real, pinned E5 tokenizer -- same one bootstrap uses
    docs = ingest.load_documents()
    cfg = ingest.CHUNK_CONFIGS["B"]
    total_chunks = sum(
        len(chunk_document(d.text, d.title, tok, cfg["chunk_tokens"], cfg["overlap_tokens"])) for d in docs
    )
    assert total_chunks == 287


# --- real-Qdrant-server: safety/idempotency/fail-closed (throwaway collections only) ----


@pytest.fixture
def throwaway_chunk_config(monkeypatch):
    name = f"PROMOQ_{uuid.uuid4().hex[:8]}"
    monkeypatch.setitem(ingest.CHUNK_CONFIGS, name, {"chunk_tokens": 200, "overlap_tokens": 20, "top_k": 3})
    return name


@pytest.fixture
def real_client_for_cleanup():
    client = _get_real_server_client()
    try:
        yield client
    finally:
        client.close()


def _args(chunk_config: str, collection_name: str, **overrides) -> argparse.Namespace:
    base = {
        "qdrant_url": QDRANT_TEST_URL, "chunk_config": chunk_config, "collection_name": collection_name,
        "rebuild_on_fingerprint_mismatch": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_bootstrap_is_idempotent_for_the_configured_collection(throwaway_chunk_config, real_client_for_cleanup):
    collection_name = f"PROMO_TEST_{uuid.uuid4().hex[:8]}"
    assert collection_name not in _LEGACY_COLLECTIONS and collection_name != "istanbul_rag_B_v2"
    try:
        args = _args(throwaway_chunk_config, collection_name)
        assert ingest_cli.cmd_bootstrap(args) == 0
        count1 = real_client_for_cleanup.count(collection_name).count
        assert count1 > 1
        assert ingest_cli.cmd_bootstrap(args) == 0
        count2 = real_client_for_cleanup.count(collection_name).count
        assert count2 == count1
    finally:
        try:
            real_client_for_cleanup.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass


def test_fingerprint_mismatch_fails_closed_then_explicit_rebuild_targets_only_that_collection(
    monkeypatch, throwaway_chunk_config, real_client_for_cleanup
):
    collection_name = f"PROMO_TEST_{uuid.uuid4().hex[:8]}"
    assert collection_name not in _LEGACY_COLLECTIONS and collection_name != "istanbul_rag_B_v2"
    try:
        assert ingest_cli.cmd_bootstrap(_args(throwaway_chunk_config, collection_name)) == 0
        first_count = real_client_for_cleanup.count(collection_name).count
        first_fp = real_client_for_cleanup.retrieve(
            collection_name=collection_name, ids=[qdrant_store.FINGERPRINT_POINT_ID], with_payload=True
        )[0].payload["__fingerprint__"]

        # Simulate a corpus change: bootstrap now "expects" a different
        # corpus than what this collection was actually built with.
        different_docs = [_doc("a_completely_different_demo_corpus")]
        monkeypatch.setattr(ingest, "load_documents", lambda: different_docs)

        # Fails closed: refuses, exit 2, collection completely untouched.
        refuse_exit = ingest_cli.cmd_bootstrap(_args(throwaway_chunk_config, collection_name))
        assert refuse_exit == 2
        assert real_client_for_cleanup.count(collection_name).count == first_count
        still_fp = real_client_for_cleanup.retrieve(
            collection_name=collection_name, ids=[qdrant_store.FINGERPRINT_POINT_ID], with_payload=True
        )[0].payload["__fingerprint__"]
        assert still_fp == first_fp

        # Explicit, clearly named rebuild: succeeds, targets ONLY this
        # exact configured collection.
        rebuild_exit = ingest_cli.cmd_bootstrap(_args(throwaway_chunk_config, collection_name, rebuild_on_fingerprint_mismatch=True))
        assert rebuild_exit == 0
        rebuilt_fp = real_client_for_cleanup.retrieve(
            collection_name=collection_name, ids=[qdrant_store.FINGERPRINT_POINT_ID], with_payload=True
        )[0].payload["__fingerprint__"]
        assert rebuilt_fp != first_fp  # genuinely rebuilt under the new corpus identity

        # Legacy collections were never named, read, or written by any of
        # the above -- the rebuild call only ever referenced
        # `collection_name`, the throwaway name this test itself created.
    finally:
        try:
            real_client_for_cleanup.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass


def test_rebuild_flag_absent_by_default(throwaway_chunk_config, real_client_for_cleanup):
    """--rebuild-on-fingerprint-mismatch is strictly opt-in: the default
    `bootstrap` parse never sets it True."""
    args = ingest_cli.build_parser().parse_args(["bootstrap"])
    assert args.rebuild_on_fingerprint_mismatch is False


# --- review-round fix 1: canonical corpus identity protection --------------------------


def test_official_collection_is_correctly_idempotent_at_69_documents_287_chunks_288_points():
    """The exact official counts (39 EN / 14 TR / 16 AR -> 69 documents,
    287 chunks, 288 points with the fingerprint marker), proven through
    the real ingestion path (ingest_config, the real pinned E5 tokenizer,
    real heading-aware chunking) with `collection_name_override=
    'istanbul_rag_B_v2'` -- but against a real, EMBEDDED in-memory Qdrant
    client (qdrant_store.local_client), never a real server. The literal
    official name is therefore fully isolated here: nothing this test
    does can ever reach any persistent Qdrant instance, demo or
    otherwise."""
    client = qdrant_store.local_client(path=None)
    tok = embeddings.load_tokenizer()  # the real, pinned E5 tokenizer -- same one bootstrap uses
    fake_fp = {"model_name": "fake", "revision": "0", "dim": FAKE_DIM}

    r1 = ingest.ingest_config(client, "B", tok, fake_embed, fake_fp, collection_name_override="istanbul_rag_B_v2")
    assert len({p["source_id"] for p in r1.chunk_payloads}) == 69
    assert len(r1.chunk_ids) == 287
    assert client.count("istanbul_rag_B_v2").count == 288

    # Idempotent: re-running produces the identical chunk_id set and does
    # not duplicate points.
    r2 = ingest.ingest_config(client, "B", tok, fake_embed, fake_fp, collection_name_override="istanbul_rag_B_v2")
    assert r1.chunk_ids == r2.chunk_ids
    assert r1.fingerprint == r2.fingerprint
    assert client.count("istanbul_rag_B_v2").count == 288


#: `ingest --source` was disabled for this release (review-round fix):
#: a noncanonical collection populated incrementally would either wrongly
#: keep claiming the canonical 69-document fingerprint, or need a second
#: fingerprinting scheme this release does not implement. Disabled for
#: EVERY collection name, not only the three named here -- these three are
#: the ones explicitly required to be proven protected.
_NAMES_INGEST_SOURCE_MUST_NEVER_TOUCH = (
    "istanbul_rag_B", "istanbul_rag_B_r1_shadow", "istanbul_rag_B_v2",
)


@pytest.mark.parametrize("collection_name", _NAMES_INGEST_SOURCE_MUST_NEVER_TOUCH)
def test_ingest_source_refuses_unconditionally_for_every_named_collection(tmp_path, collection_name):
    """Refuses before any network connection is even attempted -- the
    guard runs before _connect() and before any file is loaded, so an
    unreachable/invalid qdrant_url and a nonexistent --source path both
    still prove the refusal (neither was needed to reach it)."""
    args = argparse.Namespace(
        qdrant_url="http://unreachable.invalid:1",
        chunk_config="B",
        collection_name=collection_name,
        source=str(tmp_path / "does_not_exist.json"),
    )
    exit_code = ingest_cli.cmd_ingest(args)
    assert exit_code == 2


@pytest.mark.parametrize("collection_name", _NAMES_INGEST_SOURCE_MUST_NEVER_TOUCH)
def test_ingest_source_refuses_even_when_the_name_is_supplied_via_env(monkeypatch, tmp_path, collection_name):
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", collection_name)
    parsed = ingest_cli.build_parser().parse_args(
        ["--qdrant-url", "http://unreachable.invalid:1", "ingest", "--source", str(tmp_path / "does_not_exist.json")]
    )
    assert parsed.collection_name == collection_name
    assert ingest_cli.cmd_ingest(parsed) == 2


def test_ingest_source_refuses_for_a_genuinely_arbitrary_noncanonical_collection_too(
    tmp_path, throwaway_chunk_config, real_client_for_cleanup
):
    """The disablement is unconditional, not scoped to the three named
    collections above -- proven against a brand-new throwaway name the
    command has never seen, confirming it is never created."""
    collection_name = f"PROMO_TEST_{uuid.uuid4().hex[:8]}"
    assert collection_name not in _LEGACY_COLLECTIONS and collection_name != "istanbul_rag_B_v2"
    doc_path = tmp_path / "demo.json"
    doc_path.write_text(
        json.dumps({
            "source_id": "promo_demo_doc", "title": "Promo Demo Doc", "language": "en",
            "content_type": "attraction", "text": "A short, safe demo fact for the noncanonical collection test.",
        }),
        encoding="utf-8",
    )
    args = _args(throwaway_chunk_config, collection_name, source=str(doc_path))
    exit_code = ingest_cli.cmd_ingest(args)
    assert exit_code == 2
    existing = {c.name for c in real_client_for_cleanup.get_collections().collections}
    assert collection_name not in existing  # never created -- ingest --source cannot mutate anything


def test_bootstrap_subcommand_accepts_no_source_argument_and_so_can_never_ingest_partial_content():
    """Structural proof that the CLI's only remaining write path
    (`bootstrap`) cannot be given a partial/incremental document set at
    all -- there is no `--source`-shaped argument on this subparser, so
    `ingest_config()` is always called with `documents=None` (the real
    full canonical corpus, rag/ingest.py::load_documents()) from
    cmd_bootstrap. Combined with `ingest --source` being disabled for
    every collection name (tests above), a noncanonical collection can
    never receive the official fingerprint through this CLI's reachable
    surface -- the only way to write that fingerprint is to genuinely
    ingest the exact canonical corpus."""
    bootstrap_args = ingest_cli.build_parser().parse_args(["bootstrap"])
    assert not hasattr(bootstrap_args, "source")


def test_the_ingest_config_library_function_itself_does_not_self_police_this_which_is_why_the_cli_command_is_disabled():
    """Documents, precisely, the residual fact that makes disabling the
    whole `ingest --source` CLI command (rather than trying to patch it)
    the correct, smallest-safe fix: rag/ingest.py::ingest_config() is a
    lower-level library function that always stores the CANONICAL
    fingerprint (compute_fingerprint(chunk_config, embedding_fp), always
    the full corpus) regardless of what partial `documents=` it is asked
    to actually embed -- this was itself a deliberate, correct design
    choice from the prior round (it is what makes bootstrap's fingerprint
    stable and lets `ensure_collection` compare it consistently), but it
    means the library has no built-in way to safely support a
    caller-supplied PARTIAL document set without silently mislabeling the
    result. No CLI code path calls ingest_config() with both a partial
    `documents=` override AND a persistent `collection_name_override`
    any more (cmd_ingest is disabled; cmd_bootstrap always passes the
    full corpus) -- this test exists so that fact stays visible and
    tested, not just asserted in a docstring."""
    embedding_fp = {"model_name": "fake", "revision": "0", "dim": FAKE_DIM}
    official_fp = ingest.compute_fingerprint("B", embedding_fp)

    client = qdrant_store.local_client(path=None)
    tok = SimpleWhitespaceTokenizer()
    one_doc = [_doc("only_one_demo_doc")]
    result = ingest.ingest_config(
        client, "B", tok, fake_embed, embedding_fp, documents=one_doc, collection_name_override="some_noncanonical_collection",
    )
    assert result.fingerprint == official_fp  # the library stamps the canonical fingerprint regardless...
    assert len(result.chunk_payloads) == 1  # ...even though the actual content is not the canonical corpus (287 chunks)


# --- review-round fix 2: legacy-collection deletion protection -------------------------


@pytest.mark.parametrize("protected_name", sorted(_LEGACY_COLLECTIONS))
def test_rebuild_flag_refuses_a_protected_legacy_name_supplied_via_flag(protected_name):
    """Refuses before any network connection is even attempted (same
    unreachable-URL proof as the canonical-protection test above) --
    proves the guard cannot possibly touch real data, by construction."""
    args = argparse.Namespace(
        qdrant_url="http://unreachable.invalid:1", chunk_config="B",
        collection_name=protected_name, rebuild_on_fingerprint_mismatch=True,
    )
    assert ingest_cli.cmd_bootstrap(args) == 2


@pytest.mark.parametrize("protected_name", sorted(_LEGACY_COLLECTIONS))
def test_rebuild_flag_refuses_a_protected_legacy_name_supplied_via_env(monkeypatch, protected_name):
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", protected_name)
    parsed = ingest_cli.build_parser().parse_args(
        ["--qdrant-url", "http://unreachable.invalid:1", "bootstrap", "--rebuild-on-fingerprint-mismatch"]
    )
    assert parsed.collection_name == protected_name
    assert ingest_cli.cmd_bootstrap(parsed) == 2


@pytest.mark.parametrize("protected_name", sorted(_LEGACY_COLLECTIONS))
def test_rebuild_flag_refuses_a_protected_legacy_name_and_leaves_it_byte_for_byte_unchanged_on_the_real_server(
    protected_name, real_client_for_cleanup
):
    """The strongest available proof: pointed at the REAL live Qdrant
    instance that actually holds these two real, populated legacy
    collections, --rebuild-on-fingerprint-mismatch still refuses and
    the collection's real point count and fingerprint marker are
    identical before and after -- because the guard returns before ever
    calling _connect(), so no read or write of any kind reaches this
    collection."""
    existing = {c.name for c in real_client_for_cleanup.get_collections().collections}
    if protected_name not in existing:
        pytest.skip(f"{protected_name!r} does not exist on this Qdrant instance -- nothing to protect here")

    before_count = real_client_for_cleanup.count(protected_name).count
    before_fp = real_client_for_cleanup.retrieve(
        collection_name=protected_name, ids=[qdrant_store.FINGERPRINT_POINT_ID], with_payload=True
    )[0].payload.get("__fingerprint__")

    args = argparse.Namespace(
        qdrant_url=QDRANT_TEST_URL, chunk_config="B",
        collection_name=protected_name, rebuild_on_fingerprint_mismatch=True,
    )
    exit_code = ingest_cli.cmd_bootstrap(args)
    assert exit_code == 2

    after_count = real_client_for_cleanup.count(protected_name).count
    after_fp = real_client_for_cleanup.retrieve(
        collection_name=protected_name, ids=[qdrant_store.FINGERPRINT_POINT_ID], with_payload=True
    )[0].payload.get("__fingerprint__")
    assert after_count == before_count
    assert after_fp == before_fp
