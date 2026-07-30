import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMassFunction.Monad
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog
import Mathlib.Data.ENNReal.Real
import Mathlib.Data.Fintype.Basic

namespace Creativity.Probability

noncomputable def shannonEntropy {α : Type*} [Fintype α] (p : PMF α) : ℝ :=
  -∑ x : α, (p x).toReal * Real.log (p x).toReal

lemma shannonEntropy_nonneg {α : Type*} [Fintype α] (p : PMF α) :
    0 ≤ shannonEntropy p := by
  classical
  unfold shannonEntropy
  rw [neg_nonneg]
  refine Finset.sum_nonpos fun x _ => ?_
  have hx0 : 0 ≤ (p x).toReal := ENNReal.toReal_nonneg
  have hx1 : (p x).toReal ≤ 1 := by
    rw [← ENNReal.toReal_one]
    exact (ENNReal.toReal_le_toReal (p.apply_ne_top x) ENNReal.one_ne_top).2 (p.coe_le_one x)
  exact Real.mul_log_nonpos hx0 hx1

lemma shannonEntropy_point_mass {α : Type*} [Fintype α] (x₀ : α) :
    shannonEntropy (PMF.pure x₀) = 0 := by
  classical
  unfold shannonEntropy
  rw [neg_eq_zero]
  refine Finset.sum_eq_zero fun x _ => ?_
  by_cases h : x = x₀
  · simp [PMF.pure_apply, h]
  · simp [PMF.pure_apply, h]

/-- Shannon entropy as a sum of `Real.negMulLog` terms. -/
lemma shannonEntropy_eq_sum_negMulLog {α : Type*} [Fintype α] (p : PMF α) :
    shannonEntropy p = ∑ x : α, Real.negMulLog (p x).toReal := by
  unfold shannonEntropy
  simp only [Real.negMulLog, neg_mul, Finset.sum_neg_distrib]

lemma shannonEntropy_lt_iff {α : Type*} [Fintype α] (p q : PMF α) :
    shannonEntropy p < shannonEntropy q ↔
      -∑ x : α, (p x).toReal * Real.log (p x).toReal <
        -∑ x : α, (q x).toReal * Real.log (q x).toReal := by
  simp [shannonEntropy]

end Creativity.Probability
