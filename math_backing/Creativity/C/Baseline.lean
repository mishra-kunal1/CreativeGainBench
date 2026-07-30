import Creativity.C.Theorems

/-!
Composability Score only; not used in Benchmark Score after R_C removal.
`CBase_infeasible_can_maximize` and the novelty-only baseline contrast remain
valid theorems, scoped out of the active benchmark pipeline.
-/

namespace Creativity.C.Baseline

open Creativity.C

noncomputable def rewardNovOnly {α : Type*} [AbstractCompressor α]
    (s corpus : List α) : ℝ :=
  novelty s corpus

theorem CBase_infeasible_can_maximize
    {α : Type*} [AbstractCompressor α]
    (s corpus : List α) (u τ : ℝ) (hu : u < τ)
    (hpos : 0 < novelty s corpus) :
    rewardNovOnly s corpus > 0 ∧ rewardC s corpus u τ = 0 := by
  classical
  constructor
  · simpa [rewardNovOnly] using hpos
  · simp [rewardC, not_le.mpr hu]

end Creativity.C.Baseline
