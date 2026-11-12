"""
Hybrid retrieval: dense (Qdrant) + sparse (BM25) with RRF fusion.
"""

from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from retrieval.bm25 import load_index, search as bm25_search


DB_PATH = Path("D:/k8s-rag/data/qdrant/v1.36_db")
COLLECTION_NAME = "chunks_v1.36"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RRF_K = 60
RETRIEVAL_TOP_K = 50


def _normalize_chunk_id(item) -> str:
    """bm25s returns a dict like {'id': N, 'text': 'chunk_id'}."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text") or item.get("chunk_id") or item.get("id") or str(item)
    return str(item)


def rrf_fuse(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(dense_results, start=1):
        chunk_id = _normalize_chunk_id(chunk_id)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_id, _) in enumerate(sparse_results, start=1):
        chunk_id = _normalize_chunk_id(chunk_id)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused


class HybridRetriever:
    def __init__(self):
        print("Loading Qdrant client")
        self.qdrant = QdrantClient(path=str(DB_PATH))
        print("Loading MiniLM embedding model")
        self.embedder = SentenceTransformer(MODEL_NAME, device="cpu")
        print("Loading BM25 index")
        self.bm25 = load_index()

    def dense_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[str, float]]:
        query_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
        ).points
        return [(h.payload["chunk_id"], h.score) for h in hits]

    def sparse_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[str, float]]:
        raw = bm25_search(query, self.bm25, top_k=top_k)
        return [(_normalize_chunk_id(cid), score) for cid, score in raw]

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[tuple[str, float]]:
        dense = self.dense_search(query, top_k)
        sparse = self.sparse_search(query, top_k)
        return rrf_fuse(dense, sparse)[:top_k]

    def get_payload(self, chunk_id: str) -> dict | None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=[0.0] * 384,
            query_filter=Filter(
                must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
            ),
            limit=1,
        ).points
        return results[0].payload if results else None

    def close(self):
        self.qdrant.close()