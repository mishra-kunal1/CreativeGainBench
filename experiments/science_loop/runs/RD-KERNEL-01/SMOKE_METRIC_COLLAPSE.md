# Smoke: R_D / R_B / R_creativity collapse?

n=80 texts (40 human↔gemma2:2b pairs), HashReceiver, stub CUE, 3 tasks × 4 samples.

## Hypothesis

Codebook may be unnecessary for R_D, but shared codebook soft-clustering in R_B could make metrics measure the **same** thing. Collapse threshold: |Spearman| ≥ 0.7.

## Correlations (Spearman / Pearson)

| Pair | Spearman | Pearson |
|------|----------|---------|
| r_d_ker vs r_d_hard | 0.359 | 0.159 |
| r_d_ker vs r_b_cb | -0.361 | -0.248 |
| r_d_ker vs r_b_cont | 0.069 | 0.155 |
| r_d_hard vs r_b_cb | -0.036 | 0.176 |
| r_b_cb vs r_b_cont | -0.041 | -0.110 |
| r_d_ker vs r_creativity | 0.500 | 0.498 |
| r_b_cb vs r_creativity | -0.209 | -0.298 |

## Human vs LLM separation

| Metric | AUC(H>M) | Direction | human μ | model μ | nonzero |
|--------|----------|-----------|---------|---------|---------|
| r_d_ker | 0.276 | model_higher | 0.0001181 | 0.0002859 | 1.00 |
| r_d_hard | 0.391 | model_higher | 0.000319 | 0.0001744 | 1.00 |
| r_b_cb | 0.826 | human_higher | 0.9999 | 0.9998 | 1.00 |
| r_b_cont | 0.538 | human_higher | 1 | 1 | 1.00 |
| r_creativity | 0.387 | model_higher | 7.102e-06 | 4.397e-05 | 0.23 |

## Verdict

- ker vs hard collapse: **False**
- ker vs R_B(codebook) collapse: **False**
- R_B codebook ≈ continuous: **False**
- ker tracks continuous R_B: **False**
- ker d_gate open rate: 0.23; hard: 0.29; R_creativity nonzero: 0.23

## Read (hypothesis)

**The codebook is not forcing metrics to measure the same thing.**

1. **R_D^ker ⊀ R_D^hard** (ρ≈0.36): dropping the codebook alphabet changes the construct — consistent with RD-KERNEL-01 P1 (ker recovers geometry; hard does not).
2. **R_D^ker ⊀ R_B(codebook)** (ρ≈−0.36): deformation and receiver expansion are **anti**-aligned here, not duplicates. Codebook soft-clustering in R_B does not collapse into Parzen R_D.
3. **R_B codebook ⊀ continuous self-entropy** (ρ≈−0.04): on HashReceiver samples the two R_B variants are near-uncorrelated — but both sit at a **ceiling (~1.0)**. Offline HashReceiver synthetic variants over-saturate entropy; treat R_B numbers here as a smoke probe, not a calibrated receiver study. A real Ollama/OpenAI receiver is needed before dropping the codebook from R_B.
4. **R_creativity** is mostly **d_gate(ker)** × stub-CUE×(1+R_B): only 23% nonzero (gate open rate), ρ(ker, R_creativity)≈0.50 from the gate; it inherits ker’s **model_higher** direction.

**Bottom line:** for \(R_D\), codebook is optional and *should* stay off (Parzen). For \(R_B\), codebook is still the Lean/product path, but this smoke does **not** show codebook-induced collapse with \(R_D\). Next calibration step for R_B: live receiver (not HashReceiver) before deciding codebook is dispensable there.

collapse := |Spearman|≥0.7. If ker≢hard and ker≢r_b_cb, codebook is not forcing all metrics to one signal. If r_b_cb≈r_b_cont, the codebook soft-cluster is redundant for R_B on this receiver.

