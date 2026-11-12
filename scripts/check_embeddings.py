import pandas as pd

df = pd.read_parquet("D:/k8s-rag/data/embeddings/embeddings_v1.36.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"First chunk_id: {df['chunk_id'].iloc[0]}")
print(f"First text preview: {df['text'].iloc[0][:100]}")
print(f"First embedding dim: {len(df['embedding'].iloc[0])}")
print(f"First embedding first 5 values: {df['embedding'].iloc[0][:5]}")