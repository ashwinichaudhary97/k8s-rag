"""
Build a Qdrant collection from the embedded chunks.

Uses QdrantClient(path=...) for local persistence (no server needed).
Data is written to disk automatically and reloaded on next connect.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


EMBEDDINGS_PATH = Path("D:/k8s-rag/data/embeddings/embeddings_v1.36.parquet")
CHUNKS_PATH = Path("D:/k8s-rag/data/chunks/chunks_v1.36.jsonl")
OUTPUT_DIR = Path("D:/k8s-rag/data/qdrant")
DB_PATH = OUTPUT_DIR / "v1.36_db"
MANIFEST_PATH = OUTPUT_DIR / "v1.36_build_manifest.json"

VERSION = "v1.36"
COLLECTION_NAME = f"chunks_{VERSION}"
EMBEDDING_DIM = 384

UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "k8s-rag")


def chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, chunk_id))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {EMBEDDINGS_PATH}")
    emb_df = pd.read_parquet(EMBEDDINGS_PATH)
    print(f"  {len(emb_df)} embeddings, dim {len(emb_df['embedding'].iloc[0])}")

    print(f"Loading metadata from {CHUNKS_PATH}")
    metadata_by_id = {}
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            metadata_by_id[record["chunk_id"]] = record["metadata"]
    print(f"  {len(metadata_by_id)} metadata records")

    print(f"\nOpening persistent Qdrant at {DB_PATH}")
    client = QdrantClient(path=str(DB_PATH))

    # Drop existing collection if re-running
    if client.collection_exists(COLLECTION_NAME):
        print(f"  Collection {COLLECTION_NAME} exists, deleting")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    print(f"\nInserting {len(emb_df)} vectors")
    points = []
    for _, row in emb_df.iterrows():
        chunk_id = row["chunk_id"]
        metadata = metadata_by_id.get(chunk_id, {})
        point_uuid = chunk_id_to_uuid(chunk_id)
        points.append(
            PointStruct(
                id=point_uuid,
                vector=row["embedding"],
                payload={
                    **metadata,
                    "text": row["text"],
                    "chunk_id": chunk_id,
                },
            )
        )

    BATCH_SIZE = 1000
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Inserted {i + len(batch)}/{len(points)}")

    info = client.get_collection(COLLECTION_NAME)
    print(f"  Collection count: {info.points_count}")

    manifest = {
        "version": VERSION,
        "collection_name": COLLECTION_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "num_points": len(points),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_embeddings": str(EMBEDDINGS_PATH),
        "source_chunks": str(CHUNKS_PATH),
        "db_path": str(DB_PATH),
        "id_mapping": "uuid5(NAMESPACE, chunk_id); original chunk_id in payload",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest to {MANIFEST_PATH}")
    print("\nBuild complete.")

    # Sanity check
    print("\nSanity check: searching for 'what is a pod'")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    query_vec = model.encode("what is a pod", normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=3,
    ).points

    for i, r in enumerate(results):
        print(f"  [{i}] score={r.score:.4f} | chunk_id={r.payload.get('chunk_id', '?')}")
        print(f"      file: {r.payload.get('file_path', '?')}")
        print(f"      text: {r.payload.get('text', '')[:120]}...")

    client.close()


if __name__ == "__main__":
    main()