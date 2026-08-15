# Findings: geometry-aware R_D (RD-INSPECT → RD-SOFT → RD-KERNEL)

**Date:** 2026-08-15  
**Scope:** poetry_v2 eval, gemma2:2b (+ 3 other LLMs), label-free calibration only.

## Bottom line

1. Hard CountNgram \(R_D\) fails because the **alphabet** (VQ codebook) discards MiniLM class geometry — not because the Lean deformation functional is wrong.
2. Soft order-1 SoftCount still fails P1 (same \(R^2\approx 0.017\)).
3. **Parzen / kernel probe CE** over raw MiniLM fixes the geometry link (\(R^2\approx 0.35\)) and **discriminates** human vs LLM without forcing human>LLM.
4. Direction is consistently **model_higher** (LLMs sit off the human train manifold). That is a construct finding, not a bug — do not retune \(\sigma\)/\(\delta_D\) to flip the sign.
5. Smoke test: codebook does **not** collapse \(R_D^{\mathrm{ker}}\), \(R_D^{\mathrm{hard}}\), and \(R_B\) into one signal. Codebook remains optional for \(R_D\); still used by \(R_B\) until a live-receiver retest.

## Results table

| Run | What | P1 \(R^2\) PC1–3 | Discrimination | Notes |
|-----|------|------------------|----------------|-------|
| RD-INSPECT-01 | hard \(R_D\) | ~0.017 | null ranking | PCA class CV ~0.94 |
| RD-SOFT-01 | soft unigram | ~0.017 | null | τ=0.25, neg OK |
| RD-KERNEL-01 | Parzen CE | **~0.345** | **yes** (AUC≈0.21–0.30, model_higher) | σ≈0.326, P3/P4 pass |

Discrimination (not “human better”): CV acc with ker \(r_d\) ≈ 0.64–0.71; **not** a length artifact (humans longer, lower \(r_d\); residualized AUC still separates). Hardest LLM to detect: llama3.1:8b; easiest: phi4:14b.

## Code shipped

- `metrics/kernel_probe_ce.py`, `metrics/soft_count_ngram.py`
- `rd_backend`: `count_ngram` (default) | `soft_count` | `kernel_parzen`
- `load_kernel_backend()` + hash-verified JSON meta/thresholds
- Rebuild: `scripts/build_kernel_poetry_contexts.py` (GPU recommended) → writes `domain_*_kernel_ctx.pt` locally (not in git; ~100MB)

## Out of scope / next

- Live Ollama/OpenAI receiver before dropping codebook from \(R_B\)
- Wire `kernel_parzen` as poetry_v2 scoring default in CLI/eval if product wants the discriminator
- Do **not** fit \(\sigma\)/\(\delta_D\) to human/LLM labels
