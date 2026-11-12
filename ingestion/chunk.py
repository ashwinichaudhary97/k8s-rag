"""
Chunker for Kubernetes documentation.

Splits a ParsedDocument (from ingestion.parse) into typed chunks ready for
embedding. The chunker:

- Splits on heading boundaries (each ## / ### starts a new chunk).
- Preserves code blocks and tables as atomic (never splits inside them).
- Merges consecutive small paragraphs under the same heading.
- Enforces a token budget (default 512), splitting on paragraph boundaries
  when exceeded.
- Repeats the heading context at the start of continuation chunks.

The output is a list of Chunk objects with text and metadata.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from ingestion.parse import ParsedDocument, Token


_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate LLM token count using tiktoken."""
    return len(_ENCODER.encode(text))


@dataclass
class Chunk:
    """A chunk ready for embedding."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(file_path: str, section_path: list[str], chunk_index: int) -> str:
    key = f"{file_path}::{'/'.join(section_path)}::{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _make_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_token(token: Token) -> str:
    """Render a token back to text form for embedding."""
    if token.type == "heading":
        prefix = "#" * token.level
        return f"{prefix} {token.text}"
    if token.type == "code":
        lang = token.language or ""
        return f"```{lang}\n{token.text}\n```"
    if token.type == "table":
        return token.text
    return token.text


def chunk_document(
    doc: ParsedDocument,
    version: str,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Split a parsed document into chunks."""
    chunks: list[Chunk] = []

    buffer_tokens: list[Token] = []
    buffer_text_parts: list[str] = []
    buffer_section_path: list[str] = []
    buffer_heading: str = ""

    def flush_buffer() -> None:
        nonlocal buffer_tokens, buffer_text_parts
        if not buffer_tokens:
            return

        text = "\n\n".join(buffer_text_parts).strip()
        if not text:
            buffer_tokens = []
            buffer_text_parts = []
            return

        has_code = any(t.type == "code" for t in buffer_tokens)
        chunk_index = sum(1 for c in chunks if c.metadata["section_path"] == buffer_section_path)
        chunk_id = _make_chunk_id(doc.file_path, buffer_section_path, chunk_index)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    "file_path": doc.file_path,
                    "section_path": list(buffer_section_path),
                    "doc_type": doc.doc_type,
                    "version": version,
                    "has_code_block": has_code,
                    "content_hash": _make_content_hash(text),
                    "chunk_index": chunk_index,
                    "title": doc.title,
                },
            )
        )

        buffer_tokens = []
        buffer_text_parts = []

    for token in doc.tokens:
        if token.type == "heading":
            flush_buffer()
            buffer_section_path = list(token.section_path)
            buffer_heading = _render_token(token)
            buffer_text_parts = [buffer_heading]
            buffer_tokens = [token]
            continue

        rendered = _render_token(token)
        candidate_text = "\n\n".join(buffer_text_parts + [rendered])
        candidate_tokens = count_tokens(candidate_text)

        if candidate_tokens > max_chunk_tokens and buffer_tokens:
            flush_buffer()
            if buffer_heading:
                buffer_text_parts = [buffer_heading]
                buffer_tokens = []

        buffer_text_parts.append(rendered)
        buffer_tokens.append(token)

    flush_buffer()
    return chunks


def chunk_files(
    docs: list[ParsedDocument],
    version: str,
    max_chunk_tokens: int = 512,
) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, version, max_chunk_tokens))
    return all_chunks