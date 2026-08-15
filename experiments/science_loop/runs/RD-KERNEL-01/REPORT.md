# RD-KERNEL-01 — Parzen / kernel ProbeCompressor R_D inspect

Geometry-native Gaussian Parzen probe CE over MiniLM idea embeddings. σ and δ_D calibrated on **negatives only**. No Ollama.

## Pre-registered criteria (gemma2:2b)

| Criterion | Pass | Detail |
|-----------|------|--------|
| P1 Geometry link R² ≥ 0.15 | YES | ker R²=0.3447 (soft 0.017503786790727394, hard 0.016549838948826667) |
| P2 Ranking (CI≠0 or AUC≥0.58) | YES | Δ=-0.0001315 CI=[-0.00014000067462166136, -0.0001210674809264622] AUC=0.268 |
| P3 Negatives pass-rate ∈ [0.03,0.08] | YES | mean=0.0479 |
| P4 Axioms (≥95% correct reject) | YES | reject=0.9521 |

## gemma2:2b location

- n=1489, human_mean=1.678e-05, model_mean=0.0001483
- paired Δ (H−M)=-0.0001315 CI=[-0.00014000067462166136, -0.0001210674809264622]
- Hedges g=-0.8119, log-var ratio=-0.1768

## Interpretation

- **P1**: Parzen CE recovers embedding-class geometry (R² ≫ soft/hard ~0.017).
- **P2**: CI excludes 0, so ranking signal exists, but Δ(H−M)<0 and AUC≪0.5 ⇒ **models score higher than humans** on ker R_D. That is a construct finding (geometry linked, human>LLM not supported) — do not retune σ to flip the sign.

## Meta

- σ=0.32647937536239624, backend=kernel_parzen
- n_eval=1489

If P1 fails → escalate to leave-one-out / kNN CE (still label-free). If P1 passes and human>LLM fails → class geometry is in the score but does not favor human>LLM (construct finding, not a bug to force).
