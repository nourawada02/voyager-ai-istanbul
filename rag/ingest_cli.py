"""Real, committed CLI for RAG ingestion and readiness (Manual QA
remediation Q.1, §A). Replaces the earlier ad hoc, undocumented,
manually-run-once population of Qdrant with a reproducible, idempotent,
one-command operation any fresh clone / empty volume can run.

Three subcommands, all against a real Qdrant server, all reusing the
exact frozen Phase 3 ingestion path (rag/ingest.py::ingest_config,
rag/qdrant_store.py) -- never a separate/duplicated ingestion codepath:

    python -m rag.ingest_cli status    [--qdrant-url URL] [--chunk-config B]
    python -m rag.ingest_cli bootstrap [--qdrant-url URL] [--chunk-config B]
    python -m rag.ingest_cli ingest --source PATH [--qdrant-url URL] [--chunk-config B]

`bootstrap` ingests the full existing approved Istanbul corpus
(rag/documents/*.json, built by rag/build_documents.py from
rag/corpus_sources.py) if the target collection is missing or empty; a
correctly-populated collection is left untouched and reported as such
(idempotent, never a duplicate).

`ingest --source PATH` adds ADDITIONAL documents from a directory or
single file of normalized document records -- the exact same JSON shape
rag/documents/*.json already uses (source_id/title/language/content_type/
text, optional poi_id/district_id). This is the project's own reviewed,
safe format; nothing is scraped and no unreviewed external corpus is ever
pulled in. Every file is validated and a bad one is rejected with a clear,
specific reason, never silently skipped or silently accepted. Checksums
are always computed by this CLI itself (never trusted from the file), so
"only new or changed content" is added: an unchanged (source_id, checksum)
pair already present in the collection is a no-op.

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
from qdrant_client.http import models as qmodels

from rag import embeddings, ingest, qdrant_store
from rag.ids import sha256_hex

_SUPPORTED_LANGUAGES = frozenset({"en", "tr", "ar"})
_MAX_TEXT_CHARS = 20000
_MIN_TEXT_CHARS = 1
_REQUIRED_FIELDS = ("source_id", "title", "language", "content_type", "text")


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
    from rag.ids import config_fingerprint

    cfg = ingest._ingestion_config_dict(chunk_config, embeddings.fingerprint().as_dict())
    return config_fingerprint(cfg)


def cmd_status(args: argparse.Namespace) -> int:
    client = _connect(args.qdrant_url)
    try:
        collection_name = ingest.collection_name(args.chunk_config)
        expected_fp = _expected_fingerprint(args.chunk_config)
        result = _collection_status(client, collection_name, expected_fp)
    finally:
        client.close()
    result["collection"] = collection_name
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "ready" else 1


def cmd_bootstrap(args: argparse.Namespace) -> int:
    client = _connect(args.qdrant_url)
    try:
        collection_name = ingest.collection_name(args.chunk_config)
        expected_fp = _expected_fingerprint(args.chunk_config)
        status = _collection_status(client, collection_name, expected_fp)
        if status["state"] == "ready":
            print(f"already bootstrapped: {status['point_count']} points in {collection_name!r} (fingerprint matches) -- no-op")
            return 0
        if status["state"] == "fingerprint_mismatch":
            print(
                f"refusing to bootstrap: {collection_name!r} exists with a DIFFERENT ingestion fingerprint "
                f"({status.get('stored_fingerprint')!r} != {expected_fp!r}) -- never silently mixing configurations",
                file=sys.stderr,
            )
            return 2

        print(f"bootstrapping {collection_name!r} (state was {status['state']!r})...")
        tokenizer = embeddings.load_tokenizer()
        result = ingest.ingest_config(client, args.chunk_config, tokenizer, embeddings.embed_passages, embeddings.fingerprint().as_dict())
        count = client.count(collection_name).count
        print(f"bootstrapped {len(result.chunk_ids)} chunks from {len(set(p['source_id'] for p in result.chunk_payloads))} documents; {count} total points")
        return 0
    finally:
        client.close()


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        documents = _load_demo_documents(Path(args.source))
    except DocumentRejected as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2

    client = _connect(args.qdrant_url)
    try:
        collection_name = ingest.collection_name(args.chunk_config)
        expected_fp = _expected_fingerprint(args.chunk_config)
        status = _collection_status(client, collection_name, expected_fp)
        if status["state"] == "fingerprint_mismatch":
            print(
                f"refusing to ingest: {collection_name!r} exists with a DIFFERENT ingestion fingerprint -- "
                "run `bootstrap` first or resolve the mismatch",
                file=sys.stderr,
            )
            return 2

        # "add only new or changed content": skip any document whose
        # (source_id, checksum) already has at least one point in the
        # collection -- never re-embeds/re-uploads unchanged content.
        to_ingest = []
        skipped_unchanged = []
        for doc in documents:
            existing_points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(key="source_id", match=qmodels.MatchValue(value=doc.source_id)),
                    qmodels.FieldCondition(key="source_checksum", match=qmodels.MatchValue(value=doc.checksum)),
                ]),
                limit=1,
            ) if status["state"] != "collection_missing" else ([], None)
            if existing_points:
                skipped_unchanged.append(doc.source_id)
            else:
                to_ingest.append(doc)

        if not to_ingest:
            print(f"nothing to do: {len(skipped_unchanged)} document(s) already ingested unchanged ({skipped_unchanged})")
            return 0

        loaded = [
            ingest.LoadedDocument(
                source_id=d.source_id, title=d.title, language=d.language, content_type=d.content_type,
                poi_id=d.poi_id, district_id=d.district_id, checksum=d.checksum, text=d.text,
            )
            for d in to_ingest
        ]
        tokenizer = embeddings.load_tokenizer()
        result = ingest.ingest_config(
            client, args.chunk_config, tokenizer, embeddings.embed_passages, embeddings.fingerprint().as_dict(),
            documents=loaded,
        )
        count = client.count(collection_name).count
        print(
            f"ingested {len(result.chunk_ids)} chunks from {len(to_ingest)} new/changed document(s) "
            f"({[d.source_id for d in to_ingest]}); skipped {len(skipped_unchanged)} unchanged; {count} total points in {collection_name!r}"
        )
        return 0
    finally:
        client.close()


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
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="report Qdrant/collection readiness")
    subparsers.add_parser("bootstrap", help="ingest the full approved corpus if the collection is missing/empty")

    ingest_parser = subparsers.add_parser("ingest", help="ingest additional documents from --source")
    ingest_parser.add_argument("--source", required=True, help="a directory of *.json document records, or a single .json file")

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
