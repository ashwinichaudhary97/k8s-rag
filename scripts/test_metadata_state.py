"""
Smoke test for metadata and state store.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parse import parse_file
from ingestion.chunk import chunk_document
from ingestion.metadata import enrich_chunks, _make_source_url, _get_commit_sha
from ingestion.state_store import init_db, mark_indexed, get_indexed_hash, get_changed_files, get_file_count, get_all_versions


REPO_ROOT = Path(r"D:\k8s-rag\data\website")
DB_PATH = Path(r"D:\k8s-rag\data\state.db")


def test_url_construction():
    print("=" * 60)
    print("TEST: URL construction")
    print("=" * 60)
    cases = [
        ("content/en/docs/concepts/workloads/pods/_index.md",
         "https://kubernetes.io/docs/concepts/workloads/pods"),
        ("content/en/docs/concepts/workloads/pods/pod-lifecycle.md",
         "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle"),
        ("content/en/docs/reference/glossary/_index.md",
         "https://kubernetes.io/docs/reference/glossary"),
    ]
    for inp, expected in cases:
        actual = _make_source_url(inp)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  [{status}] {inp}")
        print(f"         -> {actual}")
        if actual != expected:
            print(f"         expected: {expected}")


def test_commit_sha():
    print()
    print("=" * 60)
    print("TEST: commit SHA")
    print("=" * 60)
    sha = _get_commit_sha(REPO_ROOT)
    print(f"  commit_sha: {sha}")
    print(f"  length: {len(sha)} (should be 40 for real SHA, 7 for 'unknown')")


def test_enrich():
    print()
    print("=" * 60)
    print("TEST: enrich_chunks")
    print("=" * 60)
    file_path = REPO_ROOT / "content/en/docs/concepts/workloads/pods/_index.md"
    doc = parse_file(file_path, REPO_ROOT)
    chunks = chunk_document(doc, version="v1.36", max_chunk_tokens=512)
    enriched = enrich_chunks(chunks, version="v1.36", repo_path=REPO_ROOT)

    print(f"  chunks: {len(enriched)}")
    print(f"  sample metadata keys: {sorted(enriched[0].metadata.keys())}")
    print(f"  sample source_url: {enriched[0].metadata['source_url']}")
    print(f"  sample version: {enriched[0].metadata['version']}")
    print(f"  sample commit_sha: {enriched[0].metadata['commit_sha'][:12]}...")


def test_state_store():
    print()
    print("=" * 60)
    print("TEST: state store")
    print("=" * 60)
    init_db(DB_PATH)
    print(f"  db initialized: {DB_PATH.exists()}")

    test_file = "content/en/docs/concepts/workloads/pods/_index.md"
    test_hash = "abc123def456"

    mark_indexed(DB_PATH, test_file, "v1.36", test_hash)
    print(f"  marked as indexed")

    retrieved = get_indexed_hash(DB_PATH, test_file, "v1.36")
    print(f"  retrieved hash: {retrieved}")
    print(f"  match: {retrieved == test_hash}")

    count = get_file_count(DB_PATH, "v1.36")
    print(f"  file count for v1.36: {count}")

    versions = get_all_versions(DB_PATH)
    print(f"  versions: {versions}")

    # Test changed files (should include everything since only 1 file is indexed)
    print()
    print("  computing changed files (this reads all 1,672 files)...")
    changed = get_changed_files(DB_PATH, REPO_ROOT, "v1.36")
    # Expected: 1,625 files pass should_include; 1 of them is marked indexed;
    # so 1,624 changed. Note: ~68 of these produce zero tokens when parsed
    # (stub files, front-matter-only). The ingestion CLI marks those __skipped__.
    expected = 1624
    status = "OK" if abs(len(changed) - expected) <= 5 else f"MISMATCH (expected ~{expected})"
    print(f"  changed files: {len(changed)} [{status}]")


if __name__ == "__main__":
    test_url_construction()
    test_commit_sha()
    test_enrich()
    test_state_store()
    print()
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)