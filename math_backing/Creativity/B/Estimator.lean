import Creativity.B.Model
import Mathlib.Tactic

namespace Creativity.B.Estimator

open Creativity.B

variable {Q Y Z : Type*} [Fintype Z]

structure BEstimator (m : BModel Q Y Z) where
  estimate : Q → Y → ℝ
  error_bound : ℝ
  hbound : ∀ q y, |estimate q y - downstreamH m q y| ≤ error_bound

/-!
`B_rank_preserved` is the benchmark bridge for entropy estimates: if the true
entropy gap is greater than twice the uniform estimator error, the measured
scores preserve the true rank ordering.
-/
theorem B_rank_preserved (m : BModel Q Y Z) (est : BEstimator m)
    (q : Q) (y₁ y₂ : Y)
    (_hf1 : feasible m q y₁) (_hf2 : feasible m q y₂)
    (hgap : downstreamH m q y₂ + 2 * est.error_bound < downstreamH m q y₁) :
    est.estimate q y₂ < est.estimate q y₁ := by
  have h2_abs := abs_le.mp (est.hbound q y₂)
  have h1_abs := abs_le.mp (est.hbound q y₁)
  linarith

end Creativity.B.Estimator
