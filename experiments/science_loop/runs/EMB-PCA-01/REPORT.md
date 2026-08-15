# EMB-PCA-01 — Embedding PCA / separability

**Generated:** 2026-08-13T20:01:04.164120+00:00  
**Model:** `gemma2:2b`  
**Pairs:** 400 eval (F10 length-clipped)  
**Embedding:** MiniLM idea mean-pool + poetry boundary (measurement stack)

## Verdict
**STRONG — embeddings carry substantial human-vs-LLM class information**

## PCA variance
| Component | Explained | Cumulative |
|-----------|-----------|------------|
| PC1 | 13.51% | 13.51% |
| PC2 | 6.80% | 20.31% |
| PC3 | 5.15% | 25.47% |

## Separability
| Metric | Value |
|--------|------:|
| Linear probe CV acc (full dim) | 0.9863 ± 0.0092 (sklearn_logreg) |
| Linear probe CV acc (PCA-3) | 0.9625 ± 0.0088 |
| Chance | 0.50 |
| Silhouette (PCA-2) | 0.3937 |
| Centroid distance (full) | 0.1928 |
| Separation ratio (full) | 0.6736 |
| Mean paired cosine(human, llm) | 0.6812 |
| Mean paired ‖Δ‖₂ | 0.4155 |

## Artifacts
- `pca_report.json` — machine-readable summary
- `pca_coords.jsonl` — PC1–3 per example
- `pca_scatter.png` — PC1/PC2 scatter (if matplotlib available)

## Science-loop notes
- If CV ≈ 0.5 and silhouette ≤ 0: measurement may be asking R_D/CUE to separate classes that **share the same embedding manifold** — consider richer encoders or features beyond MiniLM idea pools.
- If CV ≫ 0.5 but E4/R_D still fail: class signal exists in embeddings but **is not used** by current creativity metrics (metric-design problem, not representation collapse).
- Does **not** fit δ_D; diagnostic only.
