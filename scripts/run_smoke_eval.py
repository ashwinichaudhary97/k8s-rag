"""
Run the 10-question smoke set against the Qdrant index.

Metrics: Recall@5, MRR, and per-question retrieval results.
Output:  evaluation/results/stage1_smoke.json
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


DB_PATH = Path("D:/k8s-rag/data/qdrant/v1.36_db")
COLLECTION_NAME = "chunks_v1.36"
SMOKE_PATH = Path("D:/k8s-rag/evaluation/smoke_set.jsonl")
RESULTS_DIR = Path("D:/k8s-rag/evaluation/results")
RESULTS_PATH = RESULTS_DIR / "stage1_smoke.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def matches_expected(payload: dict, expected: dict) -> bool:
    """Check if a retrieved chunk matches an expected (file_path, section_path)."""
    if payload.get("file_path") != expected["file_path"]:
        return False
    expected_section = expected.get("section_path") or []
    if not expected_section:
        return True
    actual_section = payload.get("section_path") or []
    return actual_section[: len(expected_section)] == expected_section


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = []
    with SMOKE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"Loaded {len(questions)} smoke questions")

    print(f"Loading model {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Connecting to Qdrant at {DB_PATH}")
    client = QdrantClient(path=str(DB_PATH))

    results = []
    recall_hits = 0
    reciprocal_ranks = []

    for q in questions:
        query = q["question"]
        expected_list = q["expected"]

        query_vec = model.encode(query, normalize_embeddings=True).tolist()
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=TOP_K,
        ).points

        hit_rank = None
        for rank, h in enumerate(hits, 1):
            for exp in expected_list:
                if matches_expected(h.payload, exp):
                    hit_rank = rank
                    break
            if hit_rank:
                break

        if hit_rank:
            recall_hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)

        results.append({
            "question_id": q["question_id"],
            "question": query,
            "category": q["category"],
            "hit": hit_rank is not None,
            "hit_rank": hit_rank,
            "expected": expected_list,
            "top_5": [
                {
                    "rank": i + 1,
                    "score": round(h.score, 4),
                    "file_path": h.payload.get("file_path"),
                    "section_path": h.payload.get("section_path"),
                    "chunk_id": h.payload.get("chunk_id"),
                }
                for i, h in enumerate(hits)
            ],
        })

        status = f"rank {hit_rank}" if hit_rank else "MISS"
        print(f"  [{q['question_id']}] {status} | {query[:60]}")

    recall_at_5 = recall_hits / len(questions)
    mrr = sum(reciprocal_ranks) / len(questions) if reciprocal_ranks else 0.0

    summary = {
        "stage": "stage1_smoke",
        "num_questions": len(questions),
        "top_k": TOP_K,
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "hits": recall_hits,
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "results": results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Recall@5: {recall_at_5:.1%} ({recall_hits}/{len(questions)})")
    print(f"MRR:      {mrr:.4f}")
    print(f"Saved to: {RESULTS_PATH}")

    client.close()


if __name__ == "__main__":
    main()