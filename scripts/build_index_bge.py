import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


EMBEDDINGS_PATH = Path("D:/k8s-rag/data/embeddings/embeddings_v1.36_bge_m3.parquet")
CHUNKS_PATH = Path("D:/k8s-rag/data/chunks/chunks_v1.36.jsonl")
OUTPUT_DIR = Path("D:/k8s-rag/data/qdrant")
DB_PATH = OUTPUT_DIR / "v1.36_bge_db"
MANIFEST_PATH = OUTPUT_DIR / "v1.36_bge_build_manifest.json"

VERSION = "v1.36"
COLLECTION_NAME = "chunks_v1.36_bge"
EMBEDDING_DIM = 1024

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
        "model": "BAAI/bge-m3",
        "num_points": len(points),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_embeddings": str(EMBEDDINGS_PATH),
        "source_chunks": str(CHUNKS_PATH),
        "db_path": str(DB_PATH),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest to {MANIFEST_PATH}")
    print("\nBuild complete.")

    client.close()


if __name__ == "__main__":
    main()