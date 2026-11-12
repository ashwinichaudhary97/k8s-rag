"""
SQLite state store for incremental ingestion.

Tracks which files have been indexed at which version, with their content
hash. On re-ingestion, only changed files are re-parsed and re-embedded.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ingestion.parse import iter_files, should_include


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_path TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_indexed_at TEXT NOT NULL,
    PRIMARY KEY (file_path, version)
);

CREATE INDEX IF NOT EXISTS idx_version ON files(version);
CREATE INDEX IF NOT EXISTS idx_last_indexed ON files(last_indexed_at);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path) -> None:
    """Create tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def get_indexed_hash(db_path: Path, file_path: str, version: str) -> str | None:
    """Return the content hash of a file at a version, or None if not indexed."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT content_hash FROM files WHERE file_path = ? AND version = ?",
            (file_path, version),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def mark_indexed(db_path: Path, file_path: str, version: str, content_hash: str) -> None:
    """Insert or update a file's index status."""
    conn = _connect(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO files (file_path, version, content_hash, last_indexed_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(file_path, version) DO UPDATE SET
                   content_hash = excluded.content_hash,
                   last_indexed_at = excluded.last_indexed_at""",
            (file_path, version, content_hash, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_changed_files(db_path: Path, repo_path: Path, version: str) -> list[Path]:
    """Return files whose current content hash differs from the indexed hash.

    If the state store is empty (first run), returns all included files.
    """
    import hashlib

    changed = []
    for md_file in iter_files(repo_path):
        if not should_include(md_file, repo_path):
            continue
        content_bytes = md_file.read_bytes()
        current_hash = hashlib.sha256(content_bytes).hexdigest()
        relative = str(md_file.relative_to(repo_path)).replace("\\", "/")

        indexed_hash = get_indexed_hash(db_path, relative, version)
        if indexed_hash != current_hash:
            changed.append(md_file)

    return changed


def get_all_versions(db_path: Path) -> list[str]:
    """Return all versions that have been ingested."""
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT version FROM files ORDER BY version").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_file_count(db_path: Path, version: str) -> int:
    """Return the number of files indexed at a version."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM files WHERE version = ?",
            (version,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()