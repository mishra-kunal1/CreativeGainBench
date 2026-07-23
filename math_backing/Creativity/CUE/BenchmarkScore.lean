import Creativity.CUE.Receiver
import Creativity.CUE.ExpansionLabel
import Creativity.CUE.MASBridge
import Mathlib.Tactic

/-!
# The Benchmark Score: canonical \(R_{\mathrm{creativity}}\)

The *reported* Benchmark Score is defined by

  `R_creativity(q,y,s)
     = 𝟙[CUE > 0] · 𝟙[R_D > δ_D]
       · ( CUE · (1 + α·R_B^{→A}) + λ_G · G_k )`

Lean name: `Rcreativity` (abbrev `benchmarkScore`). There is no \(R_C\)
term. Both CUE and \(R_D\) are *multiplicative safety floors*: either gate
failure zeroes the **entire** score, including the interaction-gain term.
Leaving \(λ_G G_k\) outside the gates would create an additive escape hatch
around the primary constraints (cf. multiplicative safety floors in Safe RL).

Proofs of the multiplicative regime:

* `Rcreativity_cue_gate_closed` — nonpositive CUE ⇒ score = 0.
* `Rcreativity_d_gate_closed` — \(R_D ≤ δ_D\) ⇒ score = 0.
* `Rcreativity_nonneg` — score ≥ 0 under standard sign hypotheses.
* `Rcreativity_bounded` — CUE-as-multiplier Shannon ceiling (gates open).
* `Rcreativity_monotone_in_CUE` — magnitude of CUE is load-bearing when both gates open.
* `unit_conversion_sign_and_order_invariant` — bits/nats reconciliation; `α` absorbs residual scale.

An optional shifted variant `Rcreativity_shifted` (threshold `f`) is retained
for calibration experiments (Known Gaps KG-5) and is *not* the canonical
definition; it uses the same double-gate floor on the shifted scalar.
-/

namespace Creativity.CUE

open Classical

variable {Z : Type*} [Fintype Z] [Nonempty Z]

/-- The CUE gate of the Benchmark Score: `𝟙[CUE > 0]`. -/
noncomputable def cueGate (c : ℝ) : ℝ :=
  if 0 < c then 1 else 0

theorem cueGate_nonneg (c : ℝ) : 0 ≤ cueGate c := by
  unfold cueGate; split <;> norm_num

theorem cueGate_le_one (c : ℝ) : cueGate c ≤ 1 := by
  unfold cueGate; split <;> norm_num

theorem cueGate_mul_self (c : ℝ) : cueGate c * c = max c (0 : ℝ) := by
  unfold cueGate
  split_ifs with h
  · rw [one_mul, max_eq_left h.le]
  · push_neg at h
    rw [zero_mul, max_eq_right h]

/-- The D gate of the Benchmark Score: `𝟙[R_D > δ_D]`. Magnitude of
`R_D` is discarded — only threshold clearance matters. -/
noncomputable def dGate (rd δD : ℝ) : ℝ :=
  if δD < rd then 1 else 0

theorem dGate_nonneg (rd δD : ℝ) : 0 ≤ dGate rd δD := by
  unfold dGate; split <;> norm_num

theorem dGate_le_one (rd δD : ℝ) : dGate rd δD ≤ 1 := by
  unfold dGate; split <;> norm_num

/--
Canonical Benchmark Score / \(R_{\mathrm{creativity}}\):

`𝟙[CUE > 0] · 𝟙[R_D > δ_D] · (CUE · (1 + α·r_b) + λ_G · g_k)`.

The interaction-gain term sits *inside* both gates: either gate failure
nullifies the total reward (multiplicative safety floor).
-/
noncomputable def Rcreativity (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ) : ℝ :=
  cueGate (cue m) * dGate rd δD * (cue m * (1 + α * rb) + lamG * gk)

/-- Alias: the reported Benchmark Score is \(R_{\mathrm{creativity}}\). -/
noncomputable abbrev benchmarkScore (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ) : ℝ :=
  Rcreativity m α lamG rb gk rd δD

/-- Optional calibrated variant with threshold `f` (not the canonical def).
Uses the same double-gate floor on the shifted CUE scalar. -/
noncomputable def Rcreativity_shifted (m : CUEModel Z)
    (f α lamG rb gk rd δD : ℝ) : ℝ :=
  cueGate (CUE_shifted m f) * dGate rd δD *
    (CUE_shifted m f * (1 + α * rb) + lamG * gk)

theorem Rcreativity_eq_shifted_at_f_zero (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ) :
    Rcreativity m α lamG rb gk rd δD =
      Rcreativity_shifted m 0 α lamG rb gk rd δD := by
  simp [Rcreativity, Rcreativity_shifted, CUE_shifted]

/-- Multiplicative safety floor: if `CUE ≤ 0`, the entire score is zero
(including `λ_G · G_k`). -/
theorem Rcreativity_cue_gate_closed (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ) (h : cue m ≤ 0) :
    Rcreativity m α lamG rb gk rd δD = 0 := by
  unfold Rcreativity cueGate
  simp [not_lt.mpr h]

/-- Multiplicative safety floor: if `R_D ≤ δ_D`, the entire score is zero
(including `λ_G · G_k`). -/
theorem Rcreativity_d_gate_closed (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ) (h : rd ≤ δD) :
    Rcreativity m α lamG rb gk rd δD = 0 := by
  unfold Rcreativity dGate
  simp [not_lt.mpr h]

/-- Gate-closed reduction (D-gate form); alias of `Rcreativity_d_gate_closed`. -/
theorem benchmarkScore_gate_closed (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ) (h : rd ≤ δD) :
    benchmarkScore m α lamG rb gk rd δD = 0 :=
  Rcreativity_d_gate_closed m α lamG rb gk rd δD h

/-- Nonnegativity under standard sign hypotheses on the free parameters. -/
theorem Rcreativity_nonneg (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb) (_hrb1 : rb ≤ 1)
    (hlamG : 0 ≤ lamG) (hgk : 0 ≤ gk) :
    0 ≤ Rcreativity m α lamG rb gk rd δD := by
  unfold Rcreativity
  have hmul : (0 : ℝ) < 1 + α * rb := by nlinarith
  have hinner : 0 ≤ cue m * (1 + α * rb) + lamG * gk :=
    add_nonneg (mul_nonneg (cue_nonneg m) hmul.le) (mul_nonneg hlamG hgk)
  exact mul_nonneg (mul_nonneg (cueGate_nonneg _) (dGate_nonneg rd δD)) hinner

theorem benchmarkScore_nonneg (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb) (hrb1 : rb ≤ 1)
    (hlamG : 0 ≤ lamG) (hgk : 0 ≤ gk) :
    0 ≤ benchmarkScore m α lamG rb gk rd δD :=
  Rcreativity_nonneg m α lamG rb gk rd δD hα hrb0 hrb1 hlamG hgk

/--
CUE-as-multiplier ceiling (with gates open or closed):
`R_creativity ≤ (log|Z|/|y|_bits)·(1+α) + λ_G·g_k`.
-/
theorem Rcreativity_bounded (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb) (hrb1 : rb ≤ 1)
    (hlamG : 0 ≤ lamG) (hgk : 0 ≤ gk) :
    Rcreativity m α lamG rb gk rd δD ≤
      (Real.log (Fintype.card Z : ℝ) / m.bitLength) * (1 + α) +
        lamG * gk := by
  have hcueB := shannon_upper_bound_normalization m
  have hB0 : 0 ≤ Real.log (Fintype.card Z : ℝ) / m.bitLength := by
    have hlog : 0 ≤ Real.log (Fintype.card Z : ℝ) :=
      Real.log_nonneg (by exact_mod_cast Fintype.card_pos)
    exact div_nonneg hlog m.bitLength_pos.le
  have hpos : (0 : ℝ) < 1 + α * rb := by nlinarith
  have hceil : (0 : ℝ) ≤
      (Real.log (Fintype.card Z : ℝ) / m.bitLength) * (1 + α) + lamG * gk := by
    nlinarith [mul_nonneg hlamG hgk]
  unfold Rcreativity cueGate dGate
  split_ifs with hcue hd <;> simp only [one_mul, mul_one, zero_mul, mul_zero]
  · -- both gates open
    have hkey : cue m * (1 + α * rb) ≤
        (Real.log (Fintype.card Z : ℝ) / m.bitLength) * (1 + α) := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hcueB) hpos.le,
        mul_nonneg hB0 (mul_nonneg hα.le (sub_nonneg.mpr hrb1)),
        cue_nonneg m]
    linarith
  · exact hceil
  · exact hceil
  · exact hceil

theorem benchmarkScore_bounded (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb) (hrb1 : rb ≤ 1)
    (hlamG : 0 ≤ lamG) (hgk : 0 ≤ gk) :
    benchmarkScore m α lamG rb gk rd δD ≤
      (Real.log (Fintype.card Z : ℝ) / m.bitLength) * (1 + α) +
        lamG * gk :=
  Rcreativity_bounded m α lamG rb gk rd δD hα hrb0 hrb1 hlamG hgk

/-- Strict monotonicity in CUE when both gates are open. -/
theorem Rcreativity_monotone_in_CUE (m₁ m₂ : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb)
    (hcue1 : 0 < cue m₁) (hcue2 : 0 < cue m₂)
    (hgate : δD < rd)
    (hlt : cue m₁ < cue m₂) :
    Rcreativity m₁ α lamG rb gk rd δD <
      Rcreativity m₂ α lamG rb gk rd δD := by
  unfold Rcreativity cueGate dGate
  simp only [if_pos hcue1, if_pos hcue2, if_pos hgate, one_mul, mul_one]
  have hpos : (0 : ℝ) < 1 + α * rb := by nlinarith
  nlinarith [mul_pos (sub_pos.mpr hlt) hpos]

theorem benchmarkScore_monotone_in_CUE (m₁ m₂ : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb)
    (hcue1 : 0 < cue m₁) (hcue2 : 0 < cue m₂)
    (hgate : δD < rd)
    (hlt : cue m₁ < cue m₂) :
    benchmarkScore m₁ α lamG rb gk rd δD <
      benchmarkScore m₂ α lamG rb gk rd δD :=
  Rcreativity_monotone_in_CUE m₁ m₂ α lamG rb gk rd δD hα hrb0
    hcue1 hcue2 hgate hlt

/--
With both gates open and `λ_G·g_k = 0`, the score is positive iff CUE is.
-/
theorem Rcreativity_pos_iff_CUE_pos (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb)
    (hgate : δD < rd) (hz : lamG * gk = 0) :
    0 < Rcreativity m α lamG rb gk rd δD ↔ 0 < cue m := by
  unfold Rcreativity cueGate dGate
  simp only [if_pos hgate, one_mul, mul_one, hz, add_zero]
  have hpos : (0 : ℝ) < 1 + α * rb := by nlinarith
  constructor
  · intro h
    by_contra hc
    push_neg at hc
    simp [not_lt.mpr hc] at h
  · intro hcue
    simp only [if_pos hcue, one_mul]
    exact mul_pos hcue hpos

theorem benchmarkScore_sign_matches_CUE (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb)
    (hgate : δD < rd) (hz : lamG * gk = 0) :
    0 < benchmarkScore m α lamG rb gk rd δD ↔ 0 < cue m :=
  Rcreativity_pos_iff_CUE_pos m α lamG rb gk rd δD hα hrb0 hgate hz

/-- Deprecated name retained for call-site migration. -/
theorem benchmarkScore_sign_matches_CUE_shifted (m : CUEModel Z)
    (α lamG rb gk rd δD : ℝ)
    (hα : 0 < α) (hrb0 : 0 ≤ rb)
    (hgate : δD < rd) (hz : lamG * gk = 0) :
    0 < benchmarkScore m α lamG rb gk rd δD ↔ 0 < cue m :=
  benchmarkScore_sign_matches_CUE m α lamG rb gk rd δD hα hrb0 hgate hz

/--
Unit-reconciliation lemma: positive rescaling preserves sign and order.
CUE is bits-normalized; \(R_B^{\to A}\) is nats-based; `α` absorbs residual scale.
-/
theorem unit_conversion_sign_and_order_invariant (c x y : ℝ) (hc : 0 < c) :
    (0 ≤ c * x ↔ 0 ≤ x) ∧ (c * x < c * y ↔ x < y) := by
  constructor
  · constructor
    · intro h
      by_contra hx
      push_neg at hx
      nlinarith
    · intro h
      exact mul_nonneg hc.le h
  · exact mul_lt_mul_left hc

theorem log_two_pos : (0 : ℝ) < Real.log 2 :=
  Real.log_pos (by norm_num)

end Creativity.CUE
