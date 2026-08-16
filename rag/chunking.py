"""Heading-aware recursive token chunking (Checkpoint Phase 3,
architecture.md §9.2/§9.5).

Chunking is a pure function of (document text, tokenizer, chunk_tokens,
overlap_tokens) -- no randomness, no wall-clock, no external state. The
tokenizer is injected via `Tokenizer` (a two-method protocol) so:
- production ingestion uses the real pinned `intfloat/multilingual-e5-small`
  tokenizer (see `embeddings.py`) for exact token counts;
- unit tests use `SimpleWhitespaceTokenizer` (deterministic, no model
  download) to test chunk-boundary/overlap/heading logic in isolation.

Document format: sections are delimited by Markdown-style `# Heading`
lines (see `corpus_sources.py`). `heading_path` for a chunk is
`[document_title, section_heading]` -- this corpus uses a single heading
level; the recursive splitter itself does not assume only one level and
will preserve deeper heading paths if a document ever has them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class Tokenizer(Protocol):
    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Returns (token_ids, char_offsets) with len(token_ids) ==
        len(char_offsets); char_offsets[i] = (start, end) into `text`."""
        ...


class SimpleWhitespaceTokenizer:
    """Deterministic, dependency-free tokenizer for hermetic unit tests.
    Never used for the real corpus/experiment -- production ingestion
    always uses the real pinned E5 tokenizer (embeddings.py)."""

    _TOKEN_RE = re.compile(r"\S+")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        offsets = [(m.start(), m.end()) for m in self._TOKEN_RE.finditer(text)]
        ids = list(range(len(offsets)))
        return ids, offsets


@dataclass(frozen=True)
class ChunkRecord:
    heading_path: tuple[str, ...]
    text: str
    token_count: int
    char_start: int
    char_end: int
    chunk_index: int


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, ...]
    text: str
    char_start: int  # offset into the full document text


def split_into_sections(document_text: str, document_title: str) -> list[Section]:
    """Splits on `# Heading` markers into (heading_path, section_text)
    pairs. Text before the first heading (if any) gets heading_path
    (document_title,) with an empty final segment. heading_path is always
    (document_title, section_heading) for this corpus's single-level
    format."""
    matches = list(_HEADING_RE.finditer(document_text))
    if not matches:
        return [Section((document_title,), document_text, 0)]

    sections: list[Section] = []
    for i, m in enumerate(matches):
        heading = m.group(2)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
        body = document_text[body_start:body_end]
        sections.append(Section((document_title, heading), body, body_start))
    return sections


def _split_section_tokens(
    section: Section, tokenizer: Tokenizer, chunk_tokens: int, overlap_tokens: int
) -> list[tuple[int, int, int]]:
    """Returns (token_start_idx, token_end_idx_exclusive, char_start,
    char_end) windows over the section's own token stream, using a fixed
    stride so overlap is exact and deterministic."""
    token_ids, offsets = tokenizer.encode_with_offsets(section.text)
    n = len(token_ids)
    if n == 0:
        return []
    stride = max(chunk_tokens - overlap_tokens, 1)
    windows: list[tuple[int, int, int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_tokens, n)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        windows.append((start, end, char_start, char_end))
        if end == n:
            break
        start += stride
    return windows


def chunk_document(
    document_text: str,
    document_title: str,
    tokenizer: Tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[ChunkRecord]:
    """Heading-aware recursive chunking: split into heading sections
    first, then recursively token-window any section exceeding
    chunk_tokens. A section at or under chunk_tokens becomes exactly one
    chunk (no unnecessary splitting). Deterministic: identical input
    always produces an identical, identically-ordered chunk list."""
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    sections = split_into_sections(document_text, document_title)
    chunks: list[ChunkRecord] = []
    index = 0
    for section in sections:
        windows = _split_section_tokens(section, tokenizer, chunk_tokens, overlap_tokens)
        for token_start, token_end, local_char_start, local_char_end in windows:
            text = section.text[local_char_start:local_char_end]
            if not text.strip():
                continue
            chunks.append(
                ChunkRecord(
                    heading_path=section.heading_path,
                    text=text,
                    token_count=token_end - token_start,
                    char_start=section.char_start + local_char_start,
                    char_end=section.char_start + local_char_end,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks
