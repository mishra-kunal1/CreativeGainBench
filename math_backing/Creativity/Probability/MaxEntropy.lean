import Creativity.Probability.Entropy
import Creativity.Probability.KLDivergence
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

/-!
# Maximum entropy on a finite alphabet

`maxEntropy_finite_alphabet`: for any PMF `p` over a finite nonempty type `Z`,
`shannonEntropy p ≤ log |Z|`. This is the finite maximum-entropy bound used by
`CUE` normalization (PROOF-02) and by the commensurability normalization proof
(PROOF-08).

The proof does **not** introduce a new axiom: it reuses the existing Gibbs
inequality `klDivergence_nonneg` by computing the KL divergence of `p` against
the uniform distribution, which equals `log|Z| - H(p)`.
-/

namespace Creativity.Probability

open scoped ENNReal

variable {Z : Type*} [Fintype Z]

/-- The masses of a finite PMF sum to one over the finite index type. -/
lemma sum_toReal_eq_one (p : PMF Z) : ∑ x : Z, (p x).toReal = 1 := by
  have h : ∑ x : Z, p x = 1 := by
    rw [← tsum_fintype]; exact PMF.tsum_coe p
  calc
    ∑ x : Z, (p x).toReal
        = (∑ x : Z, p x).toReal :=
          (ENNReal.toReal_sum (fun x _ => p.apply_ne_top x)).symm
    _ = (1 : ℝ≥0∞).toReal := by rw [h]
    _ = 1 := ENNReal.toReal_one

/-- KL divergence of `p` against the uniform distribution equals
`log|Z| - H(p)`. -/
lemma klDivergence_uniform_eq (p : PMF Z) [Nonempty Z] :
    klDivergence p (PMF.uniformOfFintype Z) =
      Real.log (Fintype.card Z : ℝ) - shannonEntropy p := by
  classical
  have hcard : (Fintype.card Z : ℝ) ≠ 0 := by
    exact_mod_cast (Fintype.card_ne_zero)
  have huniform : ∀ x : Z,
      (PMF.uniformOfFintype Z x).toReal = (Fintype.card Z : ℝ)⁻¹ := by
    intro x
    rw [PMF.uniformOfFintype_apply]
    rw [ENNReal.toReal_inv, ENNReal.toReal_natCast]
  -- Rewrite each KL summand as `p_x log p_x + p_x log |Z|`.
  have hterm : ∀ x : Z,
      (if (p x).toReal = 0 then 0
        else (p x).toReal *
          Real.log ((p x).toReal / (PMF.uniformOfFintype Z x).toReal)) =
        (p x).toReal * Real.log (p x).toReal +
          (p x).toReal * Real.log (Fintype.card Z : ℝ) := by
    intro x
    rw [huniform x]
    by_cases hx : (p x).toReal = 0
    · simp [hx]
    · rw [if_neg hx]
      have hdiv : (p x).toReal / (Fintype.card Z : ℝ)⁻¹ =
          (p x).toReal * (Fintype.card Z : ℝ) := by
        field_simp
      rw [hdiv, Real.log_mul hx hcard]
      ring
  unfold klDivergence shannonEntropy
  rw [Finset.sum_congr rfl (fun x _ => hterm x)]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, sum_toReal_eq_one]
  ring

/--
**Finite maximum-entropy bound** (PROOF-02 dependency). For any PMF over a
finite nonempty alphabet, Shannon entropy is bounded by the log of the
alphabet size. Proved from `klDivergence_nonneg` (Gibbs), no new axiom.
-/
theorem maxEntropy_finite_alphabet (p : PMF Z) [Nonempty Z] :
    shannonEntropy p ≤ Real.log (Fintype.card Z : ℝ) := by
  have hkl : 0 ≤ klDivergence p (PMF.uniformOfFintype Z) :=
    klDivergence_nonneg _ _
  rw [klDivergence_uniform_eq p] at hkl
  linarith

end Creativity.Probability
