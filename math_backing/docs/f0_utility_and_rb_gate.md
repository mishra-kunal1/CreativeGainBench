# F0.1–F0.3 — Utility gate and R_B gating status

## F0.1 — Feasibility / \(U(q,y)\)

Python: `creativegainbench.metrics.feasibility.feasibility_bit`.

Minimal poetry proxy for Lean utility gate \(1[U(q,y)\ge\tau]\):

- non-empty text
- ≥ 2 non-blank lines

Wired into the **R_D gate** as:

\[
\mathbf{1}[\mathrm{feasible}]\cdot\mathbf{1}[R_D>\delta_D]
\]

in `d_gate(..., feasible=...)`, `compute_r_creativity`, and experiment1 `03_score_rd` payloads (`feasibility_bit`, `r_d_gate`).

## F0.2 — \(z^*\) source

CUE diagnostics include:

- `outcome_source`: `"external"` | `"self_classify"` (legacy)
- `z_star_source`: `"external"` | `"self"` (canonical for validation stratification)

Persisted on CUE score payloads from experiment1 `04_score_cue`.

## F0.3 — R_B feasibility gate

**Status: ungated.**

`compute_receiver_expansion` always returns normalized entropy expansion in
\([0,1]\). Flag: `R_B_FEASIBILITY_GATED = False` in
`metrics/receiver_expansion.py`.

Interpret R_B results as a **BBase-style** ungated contrast, not as Lean
`RewardB` (which is utility-gated). Do not claim `RewardB` theorems for
current R_B numbers until a U-gate is added there.

## F8 — G_k semantics

Default path in `compute_r_creativity` / `mas_outputs_from_row` is
**`G_k_surface`**: entropy of raw agent/joint texts (diversity proxy).

Lean-closer estimator: `compute_interaction_gain_conditioned` —
downstream-conditioned receiver samples (same shape as R_B). Not yet the
default in `R_creativity`; report `g_k_kind` on score rows.

## F10 — Length protocol

Negative bank constructs are length-matched to each domain’s eval median
char length before δ_D calibration. Experiment1 R_D scoring clips human and
model texts to that same median (capped by `max_chars`).
