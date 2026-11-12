"""
Test the parser on 5 sample files and print the typed tokens.

Usage:
    python scripts/test_parse.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parse import parse_file, iter_files


REPO_ROOT = Path(r"D:\k8s-rag\data\website")

# 5 files to inspect:
# 1. A concept page (we already saw this one)
# 2. A task page with code blocks
# 3. A reference page
# 4. A page with tables
# 5. A page with unusual Hugo shortcodes
SAMPLE_FILES = [
    "content/en/docs/concepts/workloads/pods/_index.md",
    "content/en/docs/concepts/architecture/cloud-controller.md",  # has yaml code
    "content/en/docs/concepts/cluster-administration/flow-control.md",  # has table
    "content/en/docs/concepts/cluster-administration/node-shutdown.md",  # has table
    "content/en/docs/concepts/overview/working-with-objects/namespaces.md",
]


def main():
    print("=" * 80)
    print("PARSER TEST — 5 SAMPLE FILES")
    print("=" * 80)

    for rel_path in SAMPLE_FILES:
        file_path = REPO_ROOT / rel_path
        if not file_path.exists():
            print(f"\n[SKIP] File does not exist: {rel_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"FILE: {rel_path}")
        print("=" * 80)

        doc = parse_file(file_path, REPO_ROOT)
        if doc is None:
            print("[EXCLUDED by should_include]")
            continue

        print(f"Title:    {doc.title}")
        print(f"Doc type: {doc.doc_type}")
        print(f"Tokens:   {len(doc.tokens)}")

        token_counts = {"heading": 0, "paragraph": 0, "code": 0, "table": 0}
        for token in doc.tokens:
            token_counts[token.type] += 1
        print(f"  headings:   {token_counts['heading']}")
        print(f"  paragraphs: {token_counts['paragraph']}")
        print(f"  code:       {token_counts['code']}")
        print(f"  tables:     {token_counts['table']}")

        print("\nFIRST 10 TOKENS:")
        for i, token in enumerate(doc.tokens[:10]):
            section_str = "/".join(token.section_path) if token.section_path else "(none)"
            text_preview = token.text[:100].replace("\n", " ")
            if len(token.text) > 100:
                text_preview += "..."
            print(f"  [{i}] {token.type:9} | section: {section_str}")
            print(f"      {text_preview}")

    print(f"\n{'=' * 80}")
    print("COUNTING ALL FILES")
    print("=" * 80)

    total_files = 0
    included_files = 0
    total_tokens = 0
    for md_file in iter_files(REPO_ROOT):
        total_files += 1
        doc = parse_file(md_file, REPO_ROOT)
        if doc is not None:
            included_files += 1
            total_tokens += len(doc.tokens)

    print(f"Total .md files:        {total_files}")
    print(f"Included after filter:  {included_files}")
    print(f"Excluded:               {total_files - included_files}")
    print(f"Total tokens (all):     {total_tokens}")
    if included_files > 0:
        print(f"Avg tokens per file:    {total_tokens // included_files}")


if __name__ == "__main__":
    main()