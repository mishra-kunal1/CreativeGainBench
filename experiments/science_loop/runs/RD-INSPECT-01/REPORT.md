# RD-INSPECT-01 — Why \(R_D\) does not separate human vs LLM

**Date:** 2026-08-15  
**Data:** `data/poems_20260813T171429Z.sql.gz` → `paired_eval.jsonl` (1,489 eval pairs)  
**Metric:** poetry_v2 CountNgram \(R_D^{\mathrm{norm}}\) already stored in `scores`  
**No new LLM inference.** Equivalence-first tests via `creativegainbench.stats`.

Scripts: `extract_paired_eval.py`, `analyze_rd_separation.py`. Machine-readable: `report.json`.

## Verdict

The comparison is **well-defined as a generator test and poorly defined as a ranking of “more creative.”** Humans form one population on \(R_D\) (split-half energy **equivalent**). The LLM shift sits inside that scatter (**SNR ≈ 0.02–0.04**). Location (Hedges’ \(g\)) is **equivalent** for all four models. Embeddings still separate the same texts (PC1–3 nearest-centroid CV **0.94**; EMB-PCA-01 logreg **0.99**), but those directions explain **1.7%** of \(R_D\) variance. Class lives in dense MiniLM; \(R_D\) does not use it.

Do **not** retune \(\delta_D\) or the compressor to force human > LLM. That would fit the metric to the label.

## Is the comparison meaningful?

There is **no human rater matrix** (one published body per prompt). Krippendorff’s \(\alpha\) / ICC cannot run. Proxies:

| Check | Result |
|--------|--------|
| Human split-half energy distance | \(3.7\times10^{-6}\), CI inside margin \(2.1\times10^{-4}\) → **equivalent** |
| Provenance (poetrydb − gutenberg) | \(\Delta=-4.0\times10^{-5}\), CI includes 0 |
| SNR \(\lvert\Delta\rvert/\mathrm{SD}_H\) vs gemma2:2b | **0.021** |
| Lin’s CCC (paired, \(\gamma=0.80\)) | 0.09–0.21 → **different** (inadequate agreement) |

Humans agree with *each other as a pool*. They do not agree with the model *item-wise*, and the mean shift is ~2% of human SD. Ranking “the human poem is more creative on \(R_D\)” is not a powered claim.

## Location (human − model on \(R_D^{\mathrm{norm}}\))

All CIs include 0. Win rate < 0.5 (LLM wins slightly more pairs). AUC ~0.45–0.47.

| Model | \(n\) | Mean \(\Delta\) | 95% CI | Win (H>M) | AUC | Hedges’ \(g\) (M−H) | Verdict |
|-------|------:|----------------:|--------|----------:|----:|--------------------:|---------|
| gemma2:2b | 1489 | \(+2.24\times10^{-5}\) | \([−4.2, +9.0]\times10^{-5}\) | 0.46 | 0.455 | −0.023 | **equivalent** |
| mistral:latest | 1489 | \(+2.14\times10^{-5}\) | \([−4.3, +8.0]\times10^{-5}\) | 0.46 | 0.454 | −0.023 | **equivalent** |
| llama3.1:8b | 1489 | \(+3.84\times10^{-5}\) | \([−2.1, +10.5]\times10^{-5}\) | 0.46 | 0.465 | −0.041 | **equivalent** |
| phi4:14b | 1489 | \(+2.65\times10^{-5}\) | \([−3.2, +8.7]\times10^{-5}\) | 0.47 | 0.455 | −0.029 | **equivalent** |

D-gate (negative-bank \(\delta_D\)): human pass **23.5%**. Gemma **27.1%** and phi4 **26.6%** pass *more* often (CI on Δ excludes 0). The gate does not rescue ranking.

## Shape and dispersion

KS is significant (\(p\sim10^{-8}\)) because \(n=1489\); it is **not** the decision statistic.

Energy distance is detectable but **inside** the \(0.2\times\) pooled-SD margin → **trivially different** for all four models. Standardized W1 is ~0.10 pooled-SD / ~0.21 human IQR.

Dispersion is the clearer non-location failure: models are **tighter**.

| Model | \(\log(\sigma_M^2/\sigma_H^2)\) | Verdict | SD ratio \(M/H\) | Brown–Forsythe \(p\) | LLM in human \(q_{05}\)–\(q_{95}\) |
|-------|--------------------------------:|---------|-----------------:|---------------------:|-----------------------------------:|
| gemma2:2b | −0.47 | indeterminate | 0.79 | 0.014 | 0.93 |
| mistral | −0.77 | **different** | 0.68 | 0.007 | 0.93 |
| llama3.1:8b | −0.63 | **different** | 0.73 | 0.005 | 0.93 |
| phi4:14b | −0.73 | **different** | 0.70 | 0.009 | 0.93 |

## Pooled cancellation (domains)

Gemma2:2b paired \(\Delta\) by cluster (positive = human higher):

| Domain | n | \(\Delta\) | Win H |
|--------|--:|-----------:|------:|
| 0 | 141 | \(+2.9\times10^{-5}\) | 0.55 |
| 2 | 292 | \(+2.8\times10^{-4}\) | 0.58 |
| 3 | 120 | \(+4.5\times10^{-4}\) | 0.48 |
| 9 | 114 | \(+9.4\times10^{-5}\) | 0.67 |
| 6 | 124 | \(−2.1\times10^{-4}\) | 0.33 |
| 7 | 81 | \(−5.8\times10^{-4}\) | 0.42 |
| 8 | 158 | \(−2.3\times10^{-4}\) | **0.17** |

Stratum `{0,2,9}` \(\Delta=+1.7\times10^{-4}\) vs `{6,8}` \(\Delta=−2.2\times10^{-4}\). Pooled null is partly **sign cancellation**, not only overlap. Domains 4 and 11 have no eval.

## PCA vs \(R_D\) (sparsity / unused directions)

Join of EMB-PCA-01 coords (\(n=400\) gemma pairs) to dump \(R_D\):

| \(k\) PCs | Class CV (nearest centroid) | \(R^2\) of \(r_{D\mathrm{norm}}\) |
|----------:|----------------------------:|----------------------------------:|
| 1 | 0.895 | 0.001 |
| 2 | 0.928 | 0.005 |
| 3 | **0.939** | **0.017** |

Spearman(\(\lVert\Delta\mathrm{PC}\rVert_2\), \(\Delta R_D\)) \(=0.047\) (\(p=0.35\)).  
Spearman(PC1, \(R_D\)) \(=−0.046\) (\(p=0.19\)).

EMB-PCA-01 full-dim logreg CV was 0.986. The same geometry that classifies source **does not predict** probe-relative deformation. That is a **functional / quantization gap**, not representation collapse.

## Length / symbol bands

Human mean symbols **29.4** vs gemma **21.9** (mistral 20.5, llama 22.1, phi4 25.2). Spearman(symbols, \(R_D\)) on humans \(=0.14\).

Gemma matched mid-symbol quintiles (mean \(\Delta\times10^{4}\)): Q1 (shortest) **−0.51**, Q5 (longest) **+1.01**. Residual length exists; even Q5 is small vs human SD (\(\sim10\) in these units). Length matching does not create a usable ranking.

## What this implies (later; not this pass)

1. \(R_D\) is answering “does \(y\) help compress probes given \(\mathcal{H}\)?”, not “is this human-like / unlike a 2B sample.”
2. If a later metric should use class-discriminative geometry, it has to act **before** VQ→trigram (or on a different functional of the codebook).
3. Math FormalMATH is still out of scope: no local dump, poetry_v2 contexts are the wrong \(\mathcal{H}/\mathcal{P}\).

## Reproduce

```bash
python3 experiments/science_loop/runs/RD-INSPECT-01/extract_paired_eval.py
.venv/bin/python experiments/science_loop/runs/RD-INSPECT-01/analyze_rd_separation.py
```
