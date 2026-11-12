"""
Run the smoke set against the Qdrant index.

Supports two modes:
- dense: MiniLM dense-only retrieval (Stage 1 baseline)
- hybrid: MiniLM + BM25 with RRF fusion (Stage 2)

Usage:
    python scripts/run_smoke_eval.py --mode dense --output stage1_smoke.json
    python scripts/run_smoke_eval.py --mode hybrid --output stage2_smoke.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so `retrieval` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


DB_PATH = Path("D:/k8s-rag/data/qdrant/v1.36_db")
COLLECTION_NAME = "chunks_v1.36"
SMOKE_PATH = Path("D:/k8s-rag/evaluation/smoke_set.jsonl")
RESULTS_DIR = Path("D:/k8s-rag/evaluation/results")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def matches_expected(payload: dict, expected: dict) -> bool:
    if payload.get("file_path") != expected["file_path"]:
        return False
    expected_section = expected.get("section_path") or []
    if not expected_section:
        return True
    actual_section = payload.get("section_path") or []
    return actual_section[: len(expected_section)] == expected_section


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--output", default="stage1_smoke.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / args.output

    questions = []
    with SMOKE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"Loaded {len(questions)} smoke questions")
    print(f"Mode: {args.mode}")

    if args.mode == "dense":
        print(f"Loading model {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print(f"Connecting to Qdrant at {DB_PATH}")
        client = QdrantClient(path=str(DB_PATH))
        retriever = None
    else:
        from retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever()
        client = None
        model = None

    results = []
    recall_hits = 0
    reciprocal_ranks = []

    for q in questions:
        query = q["question"]
        expected_list = q["expected"]

        if args.mode == "dense":
            query_vec = model.encode(query, normalize_embeddings=True).tolist()
            hits = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vec,
                limit=TOP_K,
            ).points
            top_payloads = [h.payload for h in hits]
            top_scores = [h.score for h in hits]
        else:
            fused = retriever.search(query, top_k=TOP_K * 4)
            top_payloads = []
            top_scores = []
            for chunk_id, score in fused[:TOP_K * 4]:
                payload = retriever.get_payload(chunk_id)
                if payload:
                    top_payloads.append(payload)
                    top_scores.append(score)
                if len(top_payloads) >= TOP_K:
                    break

        hit_rank = None
        for rank, payload in enumerate(top_payloads[:TOP_K], 1):
            for exp in expected_list:
                if matches_expected(payload, exp):
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
                    "score": round(top_scores[i], 4),
                    "file_path": top_payloads[i].get("file_path"),
                    "section_path": top_payloads[i].get("section_path"),
                    "chunk_id": top_payloads[i].get("chunk_id"),
                }
                for i in range(min(TOP_K, len(top_payloads)))
            ],
        })

        status = f"rank {hit_rank}" if hit_rank else "MISS"
        print(f"  [{q['question_id']}] {status} | {query[:60]}")

    recall_at_5 = recall_hits / len(questions)
    mrr = sum(reciprocal_ranks) / len(questions) if reciprocal_ranks else 0.0

    summary = {
        "stage": f"{args.mode}_smoke",
        "mode": args.mode,
        "num_questions": len(questions),
        "top_k": TOP_K,
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "hits": recall_hits,
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "results": results,
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Recall@5: {recall_at_5:.1%} ({recall_hits}/{len(questions)})")
    print(f"MRR:      {mrr:.4f}")
    print(f"Saved to: {results_path}")

    if client:
        client.close()
    if retriever:
        retriever.close()


if __name__ == "__main__":
    main()