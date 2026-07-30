import Creativity.C.Compression
import Mathlib.Tactic

namespace Creativity.C.Estimator

open Creativity.C

structure CEstimator (α : Type*) [AbstractCompressor α] where
  estimate : List α → List α → ℝ
  error_bound : ℝ
  hbound : ∀ s corpus, |estimate s corpus - novelty s corpus| ≤ error_bound

/-!
`C_rank_preserved` is the benchmark bridge for novelty estimates: if the true
novelty gap is greater than twice the uniform estimator error, the measured
scores preserve the true rank ordering.
-/
theorem C_rank_preserved {α : Type*} [AbstractCompressor α]
    (est : CEstimator α) (s₁ s₂ corpus : List α)
    (hgap : novelty s₂ corpus + 2 * est.error_bound < novelty s₁ corpus) :
    est.estimate s₂ corpus < est.estimate s₁ corpus := by
  have h2_abs := abs_le.mp (est.hbound s₂ corpus)
  have h1_abs := abs_le.mp (est.hbound s₁ corpus)
  linarith

end Creativity.C.Estimator
