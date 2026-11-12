# CHANGELOG

Every increment logged with: Stage | Change | Baseline | New | Delta | Kept/Reverted

| # | Stage | Change | Baseline | New | Delta | Kept/Reverted |
|---|-------|--------|----------|-----|-------|---------------|
| 1 | Stage 1 | Heading-aware chunking + MiniLM + dense-only | — | Recall@5 = 66.7%, MRR = 0.6000 | baseline | — |
| 2 | Stage 2 | Hybrid BM25 + dense (RRF, k=60) | Recall@5 = 66.7%, MRR = 0.6000 | Recall@5 = 80.0%, MRR = 0.6800 | +13.3pp, +0.08 | KEPT |