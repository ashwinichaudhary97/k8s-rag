"""
Quick test of the chunker on 3 sample files.

Prints chunk count, token distribution, and previews for each file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parse import parse_file
from ingestion.chunk import chunk_document, count_tokens


REPO_ROOT = Path(r"D:\k8s-rag\data\website")

SAMPLE_FILES = [
    "content/en/docs/concepts/workloads/pods/_index.md",
    "content/en/docs/concepts/architecture/cloud-controller.md",
    "content/en/docs/concepts/overview/working-with-objects/namespaces.md",
]


def main():
    for rel_path in SAMPLE_FILES:
        file_path = REPO_ROOT / rel_path
        if not file_path.exists():
            print(f"[SKIP] {rel_path}")
            continue

        doc = parse_file(file_path, REPO_ROOT)
        if doc is None:
            print(f"[EXCLUDED] {rel_path}")
            continue

        chunks = chunk_document(doc, version="v1.36", max_chunk_tokens=512)

        print(f"\n{'=' * 80}")
        print(f"FILE: {rel_path}")
        print(f"Title: {doc.title} | Doc type: {doc.doc_type} | Structural tokens: {len(doc.tokens)}")
        print(f"Chunks: {len(chunks)}")
        print("=" * 80)

        for i, chunk in enumerate(chunks[:5]):
            tokens = count_tokens(chunk.text)
            section = "/".join(chunk.metadata["section_path"]) or "(root)"
            has_code = chunk.metadata["has_code_block"]
            preview = chunk.text[:180].replace("\n", " ")
            if len(chunk.text) > 180:
                preview += "..."

            print(f"\n  [{i}] id={chunk.chunk_id} | tokens={tokens} | has_code={has_code}")
            print(f"      section: {section}")
            print(f"      preview: {preview}")

        if len(chunks) > 5:
            print(f"\n  ... and {len(chunks) - 5} more chunks")

        token_counts = [count_tokens(c.text) for c in chunks]
        if token_counts:
            print(f"\n  LLM token stats: min={min(token_counts)} max={max(token_counts)} "
                  f"avg={sum(token_counts) // len(token_counts)} total={sum(token_counts)}")
            oversize = sum(1 for t in token_counts if t > 512)
            undersize = sum(1 for t in token_counts if t < 50)
            print(f"  Oversized (>512): {oversize} | Undersized (<50): {undersize}")


if __name__ == "__main__":
    main()