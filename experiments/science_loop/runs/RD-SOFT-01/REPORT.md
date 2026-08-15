# RD-SOFT-01 — Soft ProbeCompressor R_D inspect

Geometry-aware soft unigram R_D (order-1 SoftCountLM). τ and δ_D calibrated on **negatives only**. No Ollama.

## Pre-registered criteria (gemma2:2b)

| Criterion | Pass | Detail |
|-----------|------|--------|
| P1 Geometry link R² ≥ 0.15 | NO | soft R²=0.0175 (hard baseline 0.016549838948826667) |
| P2 Ranking (CI≠0 or AUC≥0.58) | NO | Δ=-6.887e-08 CI=[-6.47292882684852e-07, 5.980961146462324e-07] AUC=0.514 |
| P3 Negatives pass-rate ∈ [0.03,0.08] | YES | mean=0.0326 |
| P4 Axioms (≥95% correct reject) | YES | reject=0.9674 |

## gemma2:2b location

- n=1489, human_mean=2.213e-06, model_mean=2.282e-06
- paired Δ (H−M)=-6.887e-08 CI=[-6.47292882684852e-07, 5.980961146462324e-07]
- Hedges g=-0.006534, log-var ratio=0.1753

## Meta

- τ=0.25, backend=soft_count order=1
- n_eval=1489

If P1 fails → next iteration is kernel/Parzen probe CE (still label-free). If P1 passes and P2 fails → class geometry is in the score but does not favor human>LLM (construct finding, not a bug to force).
