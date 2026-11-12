"""
BM25 sparse retrieval over the K8s docs chunks.

Builds a BM25 index from the chunks JSONL, saves it to disk, and provides
a search function that returns ranked chunk IDs.
"""

import json
from pathlib import Path

import bm25s


CHUNKS_PATH = Path("D:/k8s-rag/data/chunks/chunks_v1.36.jsonl")
INDEX_DIR = Path("D:/k8s-rag/data/bm25_index")


def build_index(chunks_path: Path = CHUNKS_PATH, index_dir: Path = INDEX_DIR) -> dict:
    """Build and save a BM25 index over the chunks."""
    index_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = []
    texts = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunk_ids.append(record["chunk_id"])
            texts.append(record["text"])

    print(f"Tokenizing {len(texts)} chunks")
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)

    print("Building BM25 index")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    print(f"Saving index to {index_dir}")
    retriever.save(str(index_dir), corpus=chunk_ids)

    return {
        "num_chunks": len(texts),
        "index_dir": str(index_dir),
        "chunk_ids_saved": len(chunk_ids),
    }


def load_index(index_dir: Path = INDEX_DIR):
    """Load the BM25 index and chunk IDs."""
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=True)
    return retriever


def search(query: str, retriever, top_k: int = 50) -> list[tuple[str, float]]:
    """Search BM25 index. Returns list of (chunk_id, score) tuples."""
    query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=top_k, show_progress=False)

    hits = []
    for i in range(results.shape[1]):
        chunk_id = results[0, i]
        score = float(scores[0, i])
        hits.append((chunk_id, score))
    return hits


if __name__ == "__main__":
    info = build_index()
    print(json.dumps(info, indent=2))