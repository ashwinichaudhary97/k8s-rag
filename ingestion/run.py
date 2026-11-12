"""
Ingestion CLI for the K8s RAG pipeline.

Orchestrates: parse -> chunk -> enrich -> mark state -> write JSONL.

Usage:
    python -m ingestion.run --version v1.36 --repo-path D:/k8s-rag/data/website \
        --output-dir D:/k8s-rag/data/chunks --db-path D:/k8s-rag/data/state.db

    # Incremental (skip already-indexed files):
    python -m ingestion.run --version v1.36 --repo-path ... --incremental

    # Limit (for testing):
    python -m ingestion.run --version v1.36 --repo-path ... --limit 20
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

from ingestion.parse import parse_file, iter_files, should_include
from ingestion.chunk import chunk_document
from ingestion.metadata import enrich_chunks
from ingestion.state_store import (
    init_db,
    mark_indexed,
    get_changed_files,
    get_file_count,
)


def _file_content_hash(file_path: Path) -> str:
    """SHA-256 of raw file bytes."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def ingest(
    version: str,
    repo_path: Path,
    output_dir: Path,
    db_path: Path,
    incremental: bool = False,
    limit: int | None = None,
) -> dict:
    """Run the ingestion pipeline. Returns a summary dict."""
    init_db(db_path)

    # Collect files to process
    if incremental:
        files = get_changed_files(db_path, repo_path, version)
        mode = "incremental"
    else:
        files = [f for f in iter_files(repo_path) if should_include(f, repo_path)]
        mode = "full"

    if limit is not None:
        files = files[:limit]
        mode += f" (limited to {limit})"

    print(f"[ingest] mode={mode}")
    print(f"[ingest] files to process: {len(files)}")

    all_chunks = []
    files_parsed = 0
    files_skipped_parse = 0
    files_skipped_empty = 0

    start = perf_counter()

    for i, file_path in enumerate(files, 1):
        relative = str(file_path.relative_to(repo_path)).replace("\\", "/")

        if i % 100 == 0 or i == len(files):
            elapsed = perf_counter() - start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  [{i}/{len(files)}] {rate:.1f} files/s - {relative}")

        doc = parse_file(file_path, repo_path)
        if doc is None:
            mark_indexed(db_path, relative, version, "__skipped__")
            files_skipped_parse += 1
            continue

        chunks = chunk_document(doc, version=version)
        if not chunks:
            mark_indexed(db_path, relative, version, "__empty__")
            files_skipped_empty += 1
            continue

        chunks = enrich_chunks(chunks, version=version, repo_path=repo_path)

        raw_hash = _file_content_hash(file_path)
        mark_indexed(db_path, relative, version, raw_hash)

        all_chunks.extend(chunks)
        files_parsed += 1

    elapsed = perf_counter() - start

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"chunks_{version}.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "version": version,
        "mode": mode,
        "files_processed": files_parsed,
        "files_skipped_parse": files_skipped_parse,
        "files_skipped_empty": files_skipped_empty,
        "total_chunks": len(all_chunks),
        "elapsed_seconds": round(elapsed, 1),
        "output_path": str(output_path),
        "state_db_files_indexed": get_file_count(db_path, version),
    }

    print()
    print("[ingest] DONE")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest K8s docs into chunks")
    parser.add_argument("--version", required=True, help="K8s version (e.g., v1.36)")
    parser.add_argument("--repo-path", required=True, type=Path, help="Path to cloned website repo")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for chunks_*.jsonl")
    parser.add_argument("--db-path", required=True, type=Path, help="Path to SQLite state DB")
    parser.add_argument("--incremental", action="store_true", help="Skip already-indexed files")
    parser.add_argument("--limit", type=int, default=None, help="Process only N files (for testing)")

    args = parser.parse_args()

    if not args.repo_path.exists():
        print(f"ERROR: repo path does not exist: {args.repo_path}")
        sys.exit(1)

    ingest(
        version=args.version,
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        db_path=args.db_path,
        incremental=args.incremental,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()