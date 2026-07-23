import Creativity.B.Theorems

namespace Creativity.B.Baseline

open Creativity.B
open Creativity.B.Theorems

variable {Q Y Z : Type*} [Fintype Z]

noncomputable def rewardUtil (m : BModel Q Y Z) (q : Q) (y : Y) : ℝ := by
  classical
  exact if feasible m q y then m.utility q y else 0

theorem BBase_indistinguishable (m : BModel Q Y Z) (q : Q) (y₁ y₂ : Y)
    (hf1 : feasible m q y₁) (hf2 : feasible m q y₂)
    (hU : m.utility q y₁ = m.utility q y₂)
    (_hH : downstreamH m q y₁ ≠ downstreamH m q y₂) :
    rewardUtil m q y₁ = rewardUtil m q y₂ := by
  classical
  simp [rewardUtil, hf1, hf2, hU]

theorem BContrast_distinguishes (m : BModel Q Y Z) (q : Q) (y₁ y₂ : Y)
    (hf1 : feasible m q y₁) (hf2 : feasible m q y₂)
    (hU : m.utility q y₁ = m.utility q y₂)
    (hH : downstreamH m q y₂ < downstreamH m q y₁) :
    rewardUtil m q y₁ = rewardUtil m q y₂ ∧
      rewardB m q y₂ < rewardB m q y₁ := by
  classical
  constructor
  · simp [rewardUtil, hf1, hf2, hU]
  · exact B1_dominance m q y₁ y₂ hf1 hf2 hH

end Creativity.B.Baseline
