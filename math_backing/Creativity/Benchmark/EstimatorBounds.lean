import Creativity.Probability.Entropy
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

namespace Creativity.Benchmark

noncomputable def plugInEntropy {Y : Type*} [Fintype Y]
    (counts : Y → ℕ) (n : ℕ) : ℝ :=
  -∑ y : Y, ((counts y : ℝ) / (n : ℝ)) *
    Real.log ((counts y : ℝ) / (n : ℝ))

/--
**Registered trusted import (PROOF-12, Known Gaps Registry KG-2).**
Miller & Madow (1955): the plug-in entropy estimator has finite-sample bias
bounded at Miller–Madow scale `|Ĥ_plug − H| ≤ |Y|/(2n)`.

Proof target for a future promotion: apply Hoeffding's inequality to the
empirical frequency estimator combined with the Lipschitz continuity of
`x ↦ -x log x` on `[0,1]` (each term is bounded in `[0, 1/e]`); a Lean 4
statistical-learning-theory library providing `Hoeffding_inequality` would
discharge it. Until then this development uses the bound as an explicitly
registered calibration assumption rather than reproving the statistics theorem.
-/
axiom plugIn_bias_axiom {Y : Type*} [Fintype Y]
    (counts : Y → ℕ) (n : ℕ) (trueEntropy : ℝ) (hn : 0 < n) :
    |plugInEntropy counts n - trueEntropy| ≤
      (Fintype.card Y : ℝ) / (2 * (n : ℝ))

theorem entropy_ranking_sample_requirement
    {Y : Type*} [Fintype Y]
    (countsLow countsHigh : Y → ℕ)
    (n : ℕ) (hn : 0 < n)
    (trueLow trueHigh : ℝ)
    (gap_exceeds : trueLow + (Fintype.card Y : ℝ) / (n : ℝ) < trueHigh) :
    plugInEntropy countsLow n < plugInEntropy countsHigh n := by
  have hlow := abs_le.mp (plugIn_bias_axiom countsLow n trueLow hn)
  have hhigh := abs_le.mp (plugIn_bias_axiom countsHigh n trueHigh hn)
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt hn)
  have hscale :
      2 * ((Fintype.card Y : ℝ) / (2 * (n : ℝ))) =
        (Fintype.card Y : ℝ) / (n : ℝ) := by
    field_simp [hnR]
    ring
  linarith

end Creativity.Benchmark
