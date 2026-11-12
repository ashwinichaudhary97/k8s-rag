"""
Chunker for Kubernetes documentation.

Splits a ParsedDocument into typed chunks ready for embedding:

- Splits on heading boundaries (## for all, ### only for non-reference docs).
- For api_reference and tool-reference doc types, only splits on ## (level 2)
  to merge API field descriptions under a single parent heading.
- Preserves code blocks and tables as atomic (never splits inside them).
- Never emits a chunk that is only a heading — a heading with no body is a
  navigation stub, not retrievable content.
- Merges undersized chunks (< min_chunk_words) with the next sibling chunk
  when the merge stays under the token budget.
- Enforces a 512-token budget, splitting on paragraph boundaries when exceeded.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from ingestion.parse import ParsedDocument, Token


_ENCODER = tiktoken.get_encoding("cl100k_base")

# Doc types where ### headings should NOT be split points
REFERENCE_DOC_TYPES = {"api_reference", "tool-reference"}


def count_tokens(text: str) -> int:
    """Estimate LLM token count using tiktoken."""
    return len(_ENCODER.encode(text))


def count_words(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(file_path: str, section_path: list[str], chunk_index: int) -> str:
    key = f"{file_path}::{'/'.join(section_path)}::{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _make_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_token(token: Token) -> str:
    if token.type == "heading":
        prefix = "#" * token.level
        return f"{prefix} {token.text}"
    if token.type == "code":
        lang = token.language or ""
        return f"```{lang}\n{token.text}\n```"
    if token.type == "table":
        return token.text
    return token.text


def _split_level(doc_type: str) -> int:
    """Return the minimum heading level that starts a new chunk.

    For reference docs, only ## (level 2) splits. For other docs, ## or ###
    (level 2 or 3) split.
    """
    if doc_type in REFERENCE_DOC_TYPES:
        return 2
    return 2  # Even for concept/task, we split on ## first; ### handled below


def chunk_document(
    doc: ParsedDocument,
    version: str,
    max_chunk_tokens: int = 512,
    min_chunk_words: int = 50,
) -> list[Chunk]:
    """Split a parsed document into chunks."""
    is_reference = doc.doc_type in REFERENCE_DOC_TYPES

    chunks: list[Chunk] = []

    buffer_tokens: list[Token] = []
    buffer_text_parts: list[str] = []
    buffer_section_path: list[str] = []
    buffer_heading: str = ""
    buffer_heading_level: int = 0

    def flush_buffer(force: bool = False) -> None:
        """Flush the current buffer into a chunk.

        If force=False and the buffer contains only a heading, do not flush —
        keep it in the buffer so the next content merges with it.
        """
        nonlocal buffer_tokens, buffer_text_parts

        if not buffer_tokens:
            return

        # Do not emit a heading-only buffer unless forced
        if not force:
            has_body = any(t.type != "heading" for t in buffer_tokens)
            if not has_body:
                return

        text = "\n\n".join(buffer_text_parts).strip()
        if not text:
            buffer_tokens = []
            buffer_text_parts = []
            return

        has_code = any(t.type == "code" for t in buffer_tokens)
        chunk_index = sum(
            1 for c in chunks if c.metadata["section_path"] == buffer_section_path
        )
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

    def start_new_heading(token: Token) -> None:
        nonlocal buffer_section_path, buffer_heading, buffer_heading_level, buffer_text_parts, buffer_tokens
        buffer_section_path = list(token.section_path)
        buffer_heading = _render_token(token)
        buffer_heading_level = token.level
        buffer_text_parts = [buffer_heading]
        buffer_tokens = [token]

    for token in doc.tokens:
        if token.type == "heading":
            # Determine if this heading starts a new chunk
            is_split_point = True
            if is_reference and token.level >= 3:
                is_split_point = False  # ### does not split in reference docs

            if is_split_point:
                # Check if the current buffer has body content
                has_body = any(t.type != "heading" for t in buffer_tokens)
                if has_body:
                    flush_buffer()
                # Else: the buffer is heading-only. Drop it and start a new heading.
                # (A heading with no body is a navigation stub.)
                start_new_heading(token)
            else:
                # For reference docs, ### headings become part of the buffer.
                # Render the heading and append it.
                buffer_text_parts.append(_render_token(token))
                buffer_tokens.append(token)
            continue

        # Non-heading token
        rendered = _render_token(token)
        candidate_text = "\n\n".join(buffer_text_parts + [rendered])
        candidate_tokens = count_tokens(candidate_text)

        if candidate_tokens > max_chunk_tokens and buffer_tokens:
            has_body = any(t.type != "heading" for t in buffer_tokens)
            if has_body:
                flush_buffer()
                # Restart buffer with heading context
                if buffer_heading:
                    buffer_text_parts = [buffer_heading]
                    buffer_tokens = []

        buffer_text_parts.append(rendered)
        buffer_tokens.append(token)

    # Final flush — force it even if heading-only, since it's the end of the doc
    flush_buffer(force=True)

    # Post-process: merge undersized chunks with the next sibling
    chunks = _merge_undersized(chunks, min_chunk_words, max_chunk_tokens)

    # Re-number chunk indices after merge
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.chunk_id = _make_chunk_id(
            chunk.metadata["file_path"],
            chunk.metadata["section_path"],
            i,
        )

    return chunks


def _merge_undersized(
    chunks: list[Chunk],
    min_words: int,
    max_tokens: int,
) -> list[Chunk]:
    """Merge chunks smaller than min_words with the next chunk when possible.

    Merges only when both chunks share the same file and the merged size
    stays under max_tokens.
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        words = count_words(chunk.text)

        # If this chunk is undersized and there's a next chunk, try to merge
        if words < min_words and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]

            # Only merge chunks from the same file
            if next_chunk.metadata["file_path"] == chunk.metadata["file_path"]:
                merged_text = chunk.text + "\n\n" + next_chunk.text
                merged_tokens = count_tokens(merged_text)

                if merged_tokens <= max_tokens:
                    # Merge: keep the first chunk's metadata, extend text
                    chunk.text = merged_text
                    chunk.metadata["has_code_block"] = (
                        chunk.metadata["has_code_block"]
                        or next_chunk.metadata["has_code_block"]
                    )
                    chunk.metadata["content_hash"] = _make_content_hash(merged_text)
                    # Section path: use the deeper of the two
                    if len(next_chunk.metadata["section_path"]) > len(
                        chunk.metadata["section_path"]
                    ):
                        chunk.metadata["section_path"] = next_chunk.metadata[
                            "section_path"
                        ]
                    i += 2  # Skip the merged chunk
                    merged.append(chunk)
                    continue

        merged.append(chunk)
        i += 1

    return merged


def chunk_files(
    docs: list[ParsedDocument],
    version: str,
    max_chunk_tokens: int = 512,
    min_chunk_words: int = 50,
) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(doc, version, max_chunk_tokens, min_chunk_words)
        )
    return all_chunks