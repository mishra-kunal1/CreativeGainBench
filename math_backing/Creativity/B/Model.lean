import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMassFunction.Monad
import Mathlib.Data.Fintype.Basic
import Creativity.Probability.Entropy

namespace Creativity.B

structure BModel (Q Y Z : Type*) where
  kernel : Q → Y → PMF Z
  utility : Q → Y → ℝ
  threshold : ℝ

variable {Q Y Z : Type*} [Fintype Z]

noncomputable def downstreamH (m : BModel Q Y Z) (q : Q) (y : Y) : ℝ :=
  Creativity.Probability.shannonEntropy (m.kernel q y)

noncomputable def mixtureKernel (m : BModel Q Y Z) [Fintype Y]
    (q : Q) (π : PMF Y) : PMF Z :=
  π.bind (m.kernel q)

def feasible (m : BModel Q Y Z) (q : Q) (y : Y) : Prop :=
  m.threshold ≤ m.utility q y

noncomputable def rewardB (m : BModel Q Y Z) (q : Q) (y : Y) : ℝ := by
  classical
  exact if feasible m q y then downstreamH m q y else 0

noncomputable def feasibleSet (m : BModel Q Y Z) [Fintype Y] (q : Q) : Finset Y := by
  classical
  exact Finset.univ.filter (feasible m q)

lemma rewardB_of_feasible (m : BModel Q Y Z) (q : Q) (y : Y) (h : feasible m q y) :
    rewardB m q y = downstreamH m q y := by
  classical
  simp [rewardB, h]

lemma rewardB_of_infeasible (m : BModel Q Y Z) (q : Q) (y : Y) (h : ¬ feasible m q y) :
    rewardB m q y = 0 := by
  classical
  simp [rewardB, h]

lemma rewardB_nonneg (m : BModel Q Y Z) (q : Q) (y : Y) :
    0 ≤ rewardB m q y := by
  classical
  by_cases h : feasible m q y
  · simp [rewardB, h, downstreamH, Creativity.Probability.shannonEntropy_nonneg]
  · simp [rewardB, h]

omit [Fintype Z] in
lemma mem_feasibleSet_iff (m : BModel Q Y Z) [Fintype Y] (q : Q) (y : Y) :
    y ∈ feasibleSet m q ↔ feasible m q y := by
  classical
  simp [feasibleSet]

end Creativity.B
