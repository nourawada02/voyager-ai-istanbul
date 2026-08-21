"""Real, committed CLI for RAG ingestion and readiness (Manual QA
remediation Q.1, §A). Replaces the earlier ad hoc, undocumented,
manually-run-once population of Qdrant with a reproducible, idempotent,
one-command operation any fresh clone / empty volume can run.

Two working subcommands, both against a real Qdrant server, both reusing
the exact frozen Phase 3 ingestion path (rag/ingest.py::ingest_config,
rag/qdrant_store.py) -- never a separate/duplicated ingestion codepath:

    python -m rag.ingest_cli status    [--qdrant-url URL] [--chunk-config B] [--collection-name NAME]
    python -m rag.ingest_cli bootstrap [--qdrant-url URL] [--chunk-config B] [--collection-name NAME]

`bootstrap` ingests the full existing approved Istanbul corpus
(rag/documents/*.json, built by rag/build_documents.py from
rag/corpus_sources.py) if the target collection is missing or empty; a
correctly-populated collection is left untouched and reported as such
(idempotent, never a duplicate). `--collection-name` / $QDRANT_COLLECTION_NAME
selects the target collection (default: istanbul_rag_B_v2, the official
collection).

`ingest --source PATH` (RAG official-promotion checkpoint, review-round
fix) is DISABLED for this release -- it always refuses, for every
collection name, and mutates nothing. A collection's stored fingerprint
is a single, fixed identity representing exactly one corpus definition;
this CLI has no correct, non-misleading way yet to represent "a canonical
collection plus N incremental documents" as its own distinct identity,
so rather than leave a collection able to silently drift from what its
fingerprint claims, the command is disabled outright. See
DISABLED_INGEST_SOURCE_MESSAGE for the exact reason and the supported
alternative (define a new versioned corpus manifest and bootstrap a new,
separately named versioned collection from it).

`status` reports one of a small closed set of states -- unreachable,
collection_missing, collection_empty, fingerprint_mismatch, ready -- plus
the real point count, never a bare boolean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient

from rag import embeddings, ingest, qdrant_store
from rag.ids import sha256_hex

_SUPPORTED_LANGUAGES = frozenset({"en", "tr", "ar"})
_MAX_TEXT_CHARS = 20000
_MIN_TEXT_CHARS = 1
_REQUIRED_FIELDS = ("source_id", "title", "language", "content_type", "text")

#: Official RAG-promotion checkpoint default -- rag-ingest's own target
#: collection, independent of `--chunk-config` (which still selects the
#: chunking config, "B", the frozen Phase 3 retrieval winner). No special
#: case for the legacy `istanbul_rag_B`/`istanbul_rag_B_r1_shadow`
#: collections exists anywhere in this module: this is a plain, generic
#: env-overridable default like every other setting here.
DEFAULT_COLLECTION_NAME = "istanbul_rag_B_v2"

#: The two frozen, pre-existing collections this checkpoint must never
#: mutate, rename, or delete, no matter how they are named on the command
#: line -- protected even when supplied explicitly via --collection-name
#: or $QDRANT_COLLECTION_NAME, and even under an explicit
#: --rebuild-on-fingerprint-mismatch. `istanbul_rag_B` is the original
#: frozen 20-document Phase 3 production collection; `istanbul_rag_B_r1_shadow`
#: is the R.1 development shadow collection this checkpoint formally
#: superseded. Both remain queryable rollback/evaluation artifacts.
PROTECTED_LEGACY_COLLECTIONS = frozenset({"istanbul_rag_B", "istanbul_rag_B_r1_shadow"})


class DocumentRejected(ValueError):
    """A demo document failed validation -- carries a specific, safe
    reason, never a raw exception/stack trace."""


@dataclass(frozen=True)
class ValidatedDocument:
    source_id: str
    title: str
    language: str
    content_type: str
    poi_id: Optional[str]
    district_id: Optional[str]
    checksum: str
    text: str


def _validate_document_record(raw: dict[str, Any], path: Path) -> ValidatedDocument:
    if not isinstance(raw, dict):
        raise DocumentRejected(f"{path}: not a JSON object")
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise DocumentRejected(f"{path}: missing required field(s) {missing}")

    source_id = raw["source_id"]
    if not isinstance(source_id, str) or not source_id.strip():
        raise DocumentRejected(f"{path}: source_id must be a non-empty string")

    title = raw["title"]
    if not isinstance(title, str) or not title.strip():
        raise DocumentRejected(f"{path}: title must be a non-empty string")

    language = raw["language"]
    if language not in _SUPPORTED_LANGUAGES:
        raise DocumentRejected(f"{path}: language {language!r} not in supported set {sorted(_SUPPORTED_LANGUAGES)}")

    content_type = raw["content_type"]
    if not isinstance(content_type, str) or not content_type.strip():
        raise DocumentRejected(f"{path}: content_type must be a non-empty string")

    text = raw["text"]
    if not isinstance(text, str):
        raise DocumentRejected(f"{path}: text must be a string")
    if len(text) < _MIN_TEXT_CHARS:
        raise DocumentRejected(f"{path}: text is empty")
    if len(text) > _MAX_TEXT_CHARS:
        raise DocumentRejected(f"{path}: text exceeds the {_MAX_TEXT_CHARS}-character safe demo-ingestion limit")

    poi_id = raw.get("poi_id")
    if poi_id is not None and not isinstance(poi_id, str):
        raise DocumentRejected(f"{path}: poi_id must be a string or null")
    district_id = raw.get("district_id")
    if district_id is not None and not isinstance(district_id, str):
        raise DocumentRejected(f"{path}: district_id must be a string or null")

    extra = set(raw.keys()) - set(_REQUIRED_FIELDS) - {"poi_id", "district_id", "checksum", "schema_version"}
    if extra:
        raise DocumentRejected(f"{path}: unrecognized field(s) {sorted(extra)} -- not the approved document-record shape")

    # Checksum is always computed here, never trusted from the file --
    # this is what makes "only new or changed content is added" honest.
    checksum = sha256_hex(text)
    return ValidatedDocument(
        source_id=source_id, title=title, language=language, content_type=content_type,
        poi_id=poi_id, district_id=district_id, checksum=checksum, text=text,
    )


def _load_demo_documents(source: Path) -> list[ValidatedDocument]:
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.glob("*.json"))
        if not paths:
            raise DocumentRejected(f"{source}: no *.json files found")
    else:
        raise DocumentRejected(f"{source}: not a file or directory")

    documents: list[ValidatedDocument] = []
    for path in paths:
        if path.suffix != ".json":
            raise DocumentRejected(f"{path}: unsupported file type {path.suffix!r} -- only .json document records are accepted")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DocumentRejected(f"{path}: not valid UTF-8 JSON ({exc})") from exc
        documents.append(_validate_document_record(raw, path))
    return documents


def _connect(qdrant_url: str) -> QdrantClient:
    try:
        return qdrant_store.server_client(qdrant_url, timeout=30.0)
    except qdrant_store.QdrantUnavailableError as exc:
        print(f"status: qdrant_unreachable ({exc})", file=sys.stderr)
        raise SystemExit(2) from exc


def _collection_status(client: QdrantClient, collection_name: str, expected_fingerprint: str) -> dict[str, Any]:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        return {"state": "collection_missing", "point_count": 0}

    count = client.count(collection_name).count
    marker = client.retrieve(collection_name=collection_name, ids=[qdrant_store.FINGERPRINT_POINT_ID], with_payload=True)
    stored_fingerprint = marker[0].payload.get("__fingerprint__") if marker else None

    if count <= 1:  # only the fingerprint marker point (or nothing) -- no real content
        return {"state": "collection_empty", "point_count": count, "stored_fingerprint": stored_fingerprint}
    if stored_fingerprint != expected_fingerprint:
        return {"state": "fingerprint_mismatch", "point_count": count, "stored_fingerprint": stored_fingerprint}
    return {"state": "ready", "point_count": count, "stored_fingerprint": stored_fingerprint}


def _expected_fingerprint(chunk_config: str) -> str:
    return ingest.compute_fingerprint(chunk_config, embeddings.fingerprint().as_dict())


def cmd_status(args: argparse.Namespace) -> int:
    client = _connect(args.qdrant_url)
    try:
        collection_name = args.collection_name
        expected_fp = _expected_fingerprint(args.chunk_config)
        result = _collection_status(client, collection_name, expected_fp)
    finally:
        client.close()
    result["collection"] = collection_name
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "ready" else 1


def cmd_bootstrap(args: argparse.Namespace) -> int:
    collection_name = args.collection_name
    # Checked before any connection or status lookup, unconditionally:
    # --rebuild-on-fingerprint-mismatch can never delete or rebuild either
    # protected legacy collection, no matter how `collection_name` got its
    # value (default, --collection-name, or $QDRANT_COLLECTION_NAME). This
    # is the one and only gate on the delete-and-recreate branch below, so
    # there is no code path that reaches `client.delete_collection(...)`
    # for a protected name.
    if args.rebuild_on_fingerprint_mismatch and collection_name in PROTECTED_LEGACY_COLLECTIONS:
        print(
            f"refusing: {collection_name!r} is a protected legacy collection -- "
            "--rebuild-on-fingerprint-mismatch can never delete or rebuild it, even when the name was supplied "
            "explicitly via --collection-name or $QDRANT_COLLECTION_NAME. Target a different collection name.",
            file=sys.stderr,
        )
        return 2

    client = _connect(args.qdrant_url)
    try:
        expected_fp = _expected_fingerprint(args.chunk_config)
        status = _collection_status(client, collection_name, expected_fp)
        if status["state"] == "ready":
            print(f"already bootstrapped: {status['point_count']} points in {collection_name!r} (fingerprint matches) -- no-op")
            return 0
        if status["state"] == "fingerprint_mismatch":
            if not args.rebuild_on_fingerprint_mismatch:
                print(
                    f"refusing to bootstrap: {collection_name!r} exists with a DIFFERENT ingestion fingerprint "
                    f"({status.get('stored_fingerprint')!r} != {expected_fp!r}) -- never silently mixing "
                    "configurations; re-run with --rebuild-on-fingerprint-mismatch to explicitly delete and "
                    "rebuild ONLY this configured collection",
                    file=sys.stderr,
                )
                return 2
            # Explicit, clearly named, opt-in rebuild -- deletes and
            # recreates ONLY `collection_name` (the one this exact
            # invocation is configured for, via --collection-name /
            # QDRANT_COLLECTION_NAME). There is no code path here or
            # anywhere else in this module that can name a different
            # collection -- and the protected-legacy-name guard above has
            # already returned before this point for either legacy name.
            print(
                f"--rebuild-on-fingerprint-mismatch set: deleting and rebuilding {collection_name!r} "
                f"(stored fingerprint {status.get('stored_fingerprint')!r} != expected {expected_fp!r})",
                file=sys.stderr,
            )
            client.delete_collection(collection_name)

        print(f"bootstrapping {collection_name!r} (state was {status['state']!r})...")
        tokenizer = embeddings.load_tokenizer()
        result = ingest.ingest_config(
            client, args.chunk_config, tokenizer, embeddings.embed_passages, embeddings.fingerprint().as_dict(),
            collection_name_override=collection_name,
        )
        count = client.count(collection_name).count
        print(f"bootstrapped {len(result.chunk_ids)} chunks from {len(set(p['source_id'] for p in result.chunk_payloads))} documents; {count} total points")
        return 0
    finally:
        client.close()


#: Disabled for this release -- see cmd_ingest()'s own docstring for why.
DISABLED_INGEST_SOURCE_MESSAGE = (
    "`ingest --source` is disabled for this release: a collection's stored fingerprint is a single, "
    "fixed identity (rag/ingest.py::compute_fingerprint) representing exactly one corpus definition, and "
    "this CLI has no correct, non-misleading way yet to represent 'a canonical collection plus N "
    "incremental documents' as a distinct, honestly-labeled identity -- an incrementally-mutated "
    "noncanonical collection would either wrongly keep claiming the canonical 69-document fingerprint, or "
    "need a second fingerprinting scheme this release does not implement. Never targetable, for the same "
    "reason, regardless of --collection-name / $QDRANT_COLLECTION_NAME: istanbul_rag_B_v2 (the official "
    "collection), istanbul_rag_B and istanbul_rag_B_r1_shadow (the two legacy collections), and every "
    "other collection name -- this command refuses unconditionally. To add sources: define a new, "
    "explicitly versioned corpus manifest (extend rag/corpus_sources.py's DOCUMENTS, or add a new "
    "manifest alongside it) and bootstrap a new, separately named versioned collection "
    "(--collection-name / $QDRANT_COLLECTION_NAME) from it with `bootstrap` -- never `ingest --source` "
    "against an existing one."
)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Disabled for this release (RAG official-promotion checkpoint,
    review-round fix). Refuses unconditionally, before any file loading
    or Qdrant connection, for every collection name -- there is
    deliberately no code path left in this function that can mutate any
    collection. See DISABLED_INGEST_SOURCE_MESSAGE for the full reason
    and the supported alternative."""
    print(DISABLED_INGEST_SOURCE_MESSAGE, file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.ingest_cli", description=__doc__)
    # QDRANT_URL env var sets the default (matches every other service's
    # own "env var configures the default, an explicit flag always wins"
    # convention) -- inside a container "localhost" would otherwise mean
    # the container itself, never the real vector-db service.
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL") or "http://localhost:6333",
        help="Qdrant server URL (default: $QDRANT_URL, else http://localhost:6333)",
    )
    parser.add_argument("--chunk-config", default="B", choices=sorted(ingest.CHUNK_CONFIGS), help="chunking config (default: B, the frozen Phase 3 retrieval winner)")
    # QDRANT_COLLECTION_NAME env var sets the default (same convention as
    # --qdrant-url above) -- the write path (this CLI) and the read path
    # (services/istanbul-expert-b/phase4/config.py) now share the exact
    # same variable name and default, so a fresh clone's `rag-ingest
    # bootstrap` populates precisely the collection System B reads.
    parser.add_argument(
        "--collection-name", default=os.environ.get("QDRANT_COLLECTION_NAME") or DEFAULT_COLLECTION_NAME,
        help=f"target Qdrant collection (default: $QDRANT_COLLECTION_NAME, else {DEFAULT_COLLECTION_NAME!r})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="report Qdrant/collection readiness")
    bootstrap_parser = subparsers.add_parser("bootstrap", help="ingest the full approved corpus if the collection is missing/empty")
    bootstrap_parser.add_argument(
        "--rebuild-on-fingerprint-mismatch", action="store_true",
        help=(
            "explicit, opt-in: if the configured collection (--collection-name / $QDRANT_COLLECTION_NAME) "
            "exists with a stale/incompatible fingerprint, delete and rebuild ONLY that exact collection "
            "instead of refusing. Never affects any other collection."
        ),
    )

    ingest_parser = subparsers.add_parser(
        "ingest", help="DISABLED for this release -- always refuses; see cmd_ingest()'s own docstring",
    )
    ingest_parser.add_argument("--source", required=True, help="a directory of *.json document records, or a single .json file (unused -- this command always refuses before reading it)")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "bootstrap":
        return cmd_bootstrap(args)
    if args.command == "ingest":
        return cmd_ingest(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover -- argparse already restricts choices
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
