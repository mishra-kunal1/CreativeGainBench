import Creativity.B.Theorems
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.ENNReal.Real
import Mathlib.Tactic

namespace Creativity.B.DPO

open Creativity.B

variable {Q Y Z : Type*} [Fintype Z]

noncomputable def softmaxPolicy {Y : Type*} [Fintype Y] [Nonempty Y]
    (logits : Y → ℝ) : PMF Y :=
  PMF.ofFintype
    (fun y => ENNReal.ofReal
      (Real.exp (logits y) / (∑ y' : Y, Real.exp (logits y'))))
    (by
      let denom : ℝ := ∑ y' : Y, Real.exp (logits y')
      have hden_pos : 0 < denom := by
        dsimp [denom]
        exact Finset.sum_pos (fun y _ => Real.exp_pos _) Finset.univ_nonempty
      have hsum : (∑ y : Y, Real.exp (logits y) / denom) = 1 := by
        rw [← Finset.sum_div]
        dsimp [denom]
        exact div_self (ne_of_gt hden_pos)
      rw [← ENNReal.ofReal_one, ← hsum]
      rw [← ENNReal.ofReal_sum_of_nonneg]
      intro y _
      exact div_nonneg (le_of_lt (Real.exp_pos _)) (le_of_lt hden_pos))

noncomputable def dpoLogOdds (logits : Y → ℝ) (yw yl : Y) : ℝ :=
  logits yw - logits yl

theorem DPO_B_correct_direction (m : BModel Q Y Z)
    [DecidableEq Y]
    (q : Q) (yw yl : Y)
    (_hfw : feasible m q yw) (_hfl : feasible m q yl)
    (hdom : downstreamH m q yl < downstreamH m q yw)
    (logits : Y → ℝ) (η : ℝ) (hη : 0 < η) :
    let logits' := fun y =>
      if y = yw then logits y + η
      else if y = yl then logits y - η
      else logits y
    dpoLogOdds logits yw yl < dpoLogOdds logits' yw yl := by
  classical
  intro logits'
  have hne' : yw ≠ yl := by
    intro h
    rw [h] at hdom
    exact (lt_irrefl (downstreamH m q yl)) hdom
  have hne : yl ≠ yw := hne'.symm
  simp [dpoLogOdds, logits', hne, hne']
  linarith

end Creativity.B.DPO
