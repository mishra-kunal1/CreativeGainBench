# RD-KERNEL-01 — Parzen / kernel probe CE for R_D

## Diagnosis (from RD-SOFT-01)

Soft order-1 SoftCountLM failed **P1** (R² of soft \(r_d\) on EMB-PCA PC1–3 ≈ **0.017**, same as hard VQ). Soft assignments still route through a frozen \(K\)-way alphabet; on large \(\mathcal{H}\), adding one \(y\) barely moves \(p(k)\), so \(D\sim 10^{-6}\) and class geometry is lost.

**Do not** retune \(\tau\) / \(\delta_D\) for human>LLM. Next compressor is geometry-native and still label-free.

## Target equation (Lean-compatible)

Keep `ProbeCompressor.corpusDeformationGain`:

\[
D(y,\mathcal{H},\mathcal{P})=\sum_{s'\in\mathcal{P}}\bigl(N(s'\mid\mathcal{H})-N(s'\mid\mathcal{H}\cup\{y\})\bigr)
\]

Define \(N\) via a Gaussian Parzen density over **raw MiniLM idea embeddings** (no codebook):

\[
\hat p_{\mathcal{H}}(e)=\frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}}
\exp\Bigl(-\frac{\|e-h\|_2^2}{2\sigma^2}\Bigr)
\]

\[
N(s'\mid\mathcal{H})=-\sum_t\log_2\max\bigl(\hat p_{\mathcal{H}}(e'_t),\varepsilon\bigr)
\]

Efficient update without rebuilding the bank:

\[
\hat p_{\mathcal{H}\cup\{y\}}(e)=
\frac{|\mathcal{H}|\,\hat p_{\mathcal{H}}(e)+\sum_j K_\sigma(e,y_j)}{|\mathcal{H}|+|y|}
\]

Normalize with existing \(\lambda_D\) (use embedding dim as continuous \(|\Sigma|\) proxy). Gate \(1[\mathrm{feasible}]\cdot 1[R_D^{\mathrm{ker}}>\delta_D]\) with \(\delta_D\) and \(\sigma\) from the **negative bank only**.

## Implementation

1. `src/creativegainbench/metrics/kernel_probe_ce.py` — `KernelProbeLM`, `kernel_deformation_gain`
2. Extend `deformation.py` — `KernelDomainDeformationContext`, `compute_kernel_deformation`
3. Config `rd_backend = "kernel_parzen"` (hard CountNgram remains default)
4. `scripts/build_kernel_poetry_contexts.py` — reuse `soft_emb_cache`; optional bank cap; σ grid on negatives
5. `experiments/science_loop/runs/RD-KERNEL-01/` — score + analyze + `REPORT.md`
6. Unit tests + `math_backing/docs/kernel_probe_compressor.md`

## Pre-registered success criteria (gemma2:2b)

| Criterion | Pass |
|-----------|------|
| P1 Geometry link | \(R^2\) of ker \(r_d\) on PC1–3 **≥ 0.15** (soft/hard ≈ 0.017) |
| P2 Ranking | Paired Δ CI excludes 0 **or** ROC-AUC ≥ 0.58; if null, report honestly — **do not** retune σ |
| P3 Negatives | Ker \(\delta_D\) neg pass-rate ∈ [0.03, 0.08] |
| P4 Axioms | Copy/pad correct-reject ≥ 95% |

If P1 fails again → escalate to leave-one-out / kNN CE or revisit idea segmentation — still no supervised class term.

## Out of scope

- Supervised Fisher / class MI
- Refitting codebook \(K\) or σ on human/LLM
- Ollama / CUE changes
