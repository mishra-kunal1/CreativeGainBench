import Creativity.Probability.MaxEntropy
import Creativity.B.Model
import Mathlib.Tactic

/-!
# Creative Update Efficiency (CUE): the receiver-grounded primary gate

CUE measures an output's effect on a *receiver* agent's decision quality,
normalized per output bit. It is the definitional anchor of the
receiver-grounded framework: a creative act is only credited when it improves
the receiver's downstream decisions, and the improvement is measured against a
Shannon bound.

For a receiver `A` with prior beliefs over a latent decision-relevant variable
`Z`, an output `y` induces a posterior. The value of information is the
improvement in expected decision quality; the (proper) Brier-score delta is a
lower bound on it. CUE normalizes the Brier-delta by the output length in bits:

  CUE(y, A, T) = brierDelta(y) / |y|_bits.

This file defines `CUEModel` and proves:

* `CUE_voi_equivalence` (PROOF-01): `CUE ≤ VOI / |y|_bits`, with equality in
  the calibrated binary case.
* `shannon_upper_bound_normalization` (PROOF-02):
  `CUE ≤ log|Z| / |y|_bits`, using `maxEntropy_finite_alphabet`.
* `CUE_primary_gate`: zero or negative CUE disqualifies the output (its gated
  score is zero), mirroring the B/C/D utility gates.
-/

namespace Creativity.CUE

open Creativity.Probability

/--
A receiver-grounded CUE model over a finite decision-relevant state space `Z`.

* `posterior` — the receiver's belief about `Z` after the output.
* `brierDelta` — reduction in Brier loss from prior to posterior (the proper
  scoring-rule improvement), nonnegative.
* `voi` — the Shannon value of information of the output for the receiver's
  decision. As a proper scoring rule, Brier improvement lower-bounds it
  (`brier_le_voi`).
* `voi_le_entropy` — VOI is bounded by the receiver's residual (posterior)
  entropy budget, itself bounded by `log|Z|` via `maxEntropy_finite_alphabet`.
* `bitLength` — the output length in bits `|y|_bits`, strictly positive.
-/
structure CUEModel (Z : Type*) [Fintype Z] [Nonempty Z] where
  posterior : PMF Z
  brierDelta : ℝ
  voi : ℝ
  bitLength : ℝ
  brier_nonneg : 0 ≤ brierDelta
  bitLength_pos : 0 < bitLength
  brier_le_voi : brierDelta ≤ voi
  voi_le_entropy : voi ≤ shannonEntropy posterior

variable {Z : Type*} [Fintype Z] [Nonempty Z]

/-- Creative Update Efficiency: normalized receiver decision-quality gain. -/
noncomputable def cue (m : CUEModel Z) : ℝ :=
  m.brierDelta / m.bitLength

/-- CUE is nonnegative. -/
theorem cue_nonneg (m : CUEModel Z) : 0 ≤ cue m :=
  div_nonneg m.brier_nonneg (le_of_lt m.bitLength_pos)

/--
**PROOF-01** `CUE_voi_equivalence`. The normalized Brier-delta CUE is a lower
bound on the normalized Shannon value of information:
`CUE(y, A, T) ≤ VOI(y, A, T) / |y|_bits`. Equality holds when the Brier
improvement saturates its proper-scoring-rule bound (the calibrated binary
case), recorded by the hypothesis `hcal : brierDelta = voi`.
-/
theorem CUE_voi_equivalence (m : CUEModel Z) :
    cue m ≤ m.voi / m.bitLength := by
  unfold cue
  exact (div_le_div_iff_of_pos_right m.bitLength_pos).mpr m.brier_le_voi

/-- The calibrated binary case: when the Brier improvement saturates its VOI
bound, normalized CUE equals normalized VOI. -/
theorem CUE_voi_equality (m : CUEModel Z) (hcal : m.brierDelta = m.voi) :
    cue m = m.voi / m.bitLength := by
  unfold cue; rw [hcal]

/--
**PROOF-02** `shannon_upper_bound_normalization`. CUE is bounded by the
Shannon maximum entropy over the receiver state space, normalized per bit:
`CUE(y, A, T) ≤ log|Z| / |y|_bits`.
-/
theorem shannon_upper_bound_normalization (m : CUEModel Z) :
    cue m ≤ Real.log (Fintype.card Z : ℝ) / m.bitLength := by
  have hchain : m.brierDelta ≤ Real.log (Fintype.card Z : ℝ) :=
    le_trans m.brier_le_voi
      (le_trans m.voi_le_entropy (maxEntropy_finite_alphabet m.posterior))
  unfold cue
  exact (div_le_div_iff_of_pos_right m.bitLength_pos).mpr hchain

/-- CUE as the primary validity gate: a downstream creativity score is admitted
only when CUE is strictly positive. Mirrors the B/C/D utility gates but sits
above them — CUE gates the entire creativity score, not a single component. -/
noncomputable def cueGatedScore (m : CUEModel Z) (downstream : ℝ) : ℝ := by
  classical
  exact if 0 < cue m then downstream else 0

/--
`CUE_primary_gate`: zero or negative CUE disqualifies all downstream metrics.
When the output does not strictly improve the receiver's decisions, its gated
creativity score is zero regardless of any B/C/D value.

Superseded for Benchmark Score use by the shifted form `CUE_shifted` and
`CUE_shifted_sign_iff` below; retained for the Composability Score.
-/
theorem CUE_primary_gate (m : CUEModel Z) (downstream : ℝ)
    (h : ¬ 0 < cue m) :
    cueGatedScore m downstream = 0 := by
  classical
  simp [cueGatedScore, h]

/-! ## Shifted CUE for the Benchmark Score

The Benchmark Score replaces the hard zero-gate on raw CUE by a *shifted*
scalar `CUE_shifted = cue − f`, where `f` is an empirically calibrated
per-domain threshold (Known Gaps Registry KG-5). Novel-but-decision-degrading
outputs are penalized (negative shifted CUE), not silently zeroed.
-/

/-- Task 1: the shifted CUE scalar `cue m − f`. -/
noncomputable def CUE_shifted (m : CUEModel Z) (f : ℝ) : ℝ :=
  cue m - f

/-- Task 1 (bundled form): a CUE model together with its calibrated threshold
`f`. -/
structure CUEThresholdModel (Z : Type*) [Fintype Z] [Nonempty Z]
    extends CUEModel Z where
  f : ℝ

/-- Shifted CUE of a bundled threshold model. -/
noncomputable def CUEThresholdModel.shifted (m : CUEThresholdModel Z) : ℝ :=
  CUE_shifted m.toCUEModel m.f

/--
Task 2: the shifted analogue of `shannon_upper_bound_normalization` — the
shifted CUE inherits the Shannon ceiling, displaced by the threshold:
`CUE_shifted ≤ log|Z| / |y|_bits − f`.
-/
theorem CUE_shifted_bound (m : CUEModel Z) (f : ℝ) :
    CUE_shifted m f ≤ Real.log (Fintype.card Z : ℝ) / m.bitLength - f := by
  unfold CUE_shifted
  have h := shannon_upper_bound_normalization m
  linarith

/--
Task 3: sign characterization. The shifted CUE is nonnegative exactly when the
raw CUE clears the threshold `f`.
-/
theorem CUE_shifted_sign_iff (m : CUEModel Z) (f : ℝ) :
    0 ≤ CUE_shifted m f ↔ f ≤ cue m := by
  unfold CUE_shifted
  constructor
  · intro h; linarith
  · intro h; linarith

open Classical in
/--
Task 5: sign of the Benchmark Score term matches the sign of the shifted CUE.
The term `CUE_shifted · 𝟙[gate] · (1 + α·r_b)` has the sign of `CUE_shifted`
whenever the gate is open, because the receiver-expansion multiplier
`1 + α·r_b` is strictly positive for bounded `r_b ∈ [0,1]`
(`receiver_expansion_bounded`); when the gate is closed the term vanishes.
This formalizes: novel-but-decision-degrading outputs are penalized, not
zeroed.
-/
theorem Rcreativity_sign_matches_CUE_shifted
    (m : CUEModel Z) (f α rb : ℝ) (hα : 0 < α)
    (hrb0 : 0 ≤ rb) (_hrb1 : rb ≤ 1) (gate : Bool) :
    (gate = true →
      (0 ≤ CUE_shifted m f * (if gate then (1 : ℝ) else 0) * (1 + α * rb) ↔
        0 ≤ CUE_shifted m f)) ∧
    (gate = false →
      CUE_shifted m f * (if gate then (1 : ℝ) else 0) * (1 + α * rb) = 0) := by
  constructor
  · intro hgate
    subst hgate
    simp only [if_true, mul_one]
    have hpos : 0 < 1 + α * rb := by nlinarith
    exact mul_nonneg_iff_of_pos_right hpos
  · intro hgate
    subst hgate
    simp

end Creativity.CUE
