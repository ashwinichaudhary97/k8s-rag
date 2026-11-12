"""
Metadata enrichment for ingested chunks.

Adds fields that the parser and chunker don't know about:
- version (canonical form)
- commit_sha (Git state of the docs repo at ingestion time)
- source_url (deep link to the docs page)
- indexed_at (ISO timestamp)

Kept separate from the chunker so that URL construction and Git state
reading can change without touching chunking logic.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ingestion.chunk import Chunk


BASE_URL = "https://kubernetes.io/docs"


def _get_commit_sha(repo_path: Path) -> str:
    """Get the current Git commit SHA of the docs repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _make_source_url(file_path: str) -> str:
    """Convert a repo-relative file path to a kubernetes.io URL.

    Example:
        content/en/docs/concepts/workloads/pods/_index.md
        -> https://kubernetes.io/docs/concepts/workloads/pods/

        content/en/docs/concepts/workloads/pods/pod-lifecycle.md
        -> https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
    """
    # Strip content/en/docs/ prefix
    prefix = "content/en/docs/"
    if file_path.startswith(prefix):
        relative = file_path[len(prefix):]
    else:
        relative = file_path

    # Strip .md extension
    if relative.endswith(".md"):
        relative = relative[:-3]

    # _index becomes the directory URL
    if relative.endswith("_index"):
        relative = relative[: -len("_index")].rstrip("/")

    return f"{BASE_URL}/{relative}".rstrip("/")


def enrich_chunks(
    chunks: list[Chunk],
    version: str,
    repo_path: Path,
) -> list[Chunk]:
    """Add commit_sha, source_url, and indexed_at to every chunk.

    Mutates chunk metadata in place and returns the same list.
    """
    commit_sha = _get_commit_sha(repo_path)
    indexed_at = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        chunk.metadata["version"] = version
        chunk.metadata["commit_sha"] = commit_sha
        chunk.metadata["source_url"] = _make_source_url(chunk.metadata["file_path"])
        chunk.metadata["indexed_at"] = indexed_at

    return chunks