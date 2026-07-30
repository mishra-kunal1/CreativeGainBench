import Creativity.Probability.Entropy
import Mathlib.Tactic

namespace Creativity.MAS

structure TwoAgentModel (Q Y₁ Y₂ Z : Type*) where
  kernel₁ : Q → Y₁ → PMF Z
  kernel₂ : Q → Y₂ → PMF Z
  kernelJoint : Q → Y₁ → Y₂ → PMF Z
  utility₁ : Q → Y₁ → ℝ
  utility₂ : Q → Y₂ → ℝ
  threshold : ℝ

variable {Q Y₁ Y₂ Z : Type*} [Fintype Z]

noncomputable def H₁ (m : TwoAgentModel Q Y₁ Y₂ Z) (q : Q) (y₁ : Y₁) : ℝ :=
  Creativity.Probability.shannonEntropy (m.kernel₁ q y₁)

noncomputable def H₂ (m : TwoAgentModel Q Y₁ Y₂ Z) (q : Q) (y₂ : Y₂) : ℝ :=
  Creativity.Probability.shannonEntropy (m.kernel₂ q y₂)

noncomputable def Hjoint (m : TwoAgentModel Q Y₁ Y₂ Z)
    (q : Q) (y₁ : Y₁) (y₂ : Y₂) : ℝ :=
  Creativity.Probability.shannonEntropy (m.kernelJoint q y₁ y₂)

noncomputable def interactionGain (m : TwoAgentModel Q Y₁ Y₂ Z)
    (q : Q) (y₁ : Y₁) (y₂ : Y₂) : ℝ :=
  Hjoint m q y₁ y₂ - max (H₁ m q y₁) (H₂ m q y₂)

theorem MAS_irreducibility (m : TwoAgentModel Q Y₁ Y₂ Z)
    (q : Q) (y₁ : Y₁) (y₂ : Y₂)
    (hgain : 0 < interactionGain m q y₁ y₂) :
    H₁ m q y₁ < Hjoint m q y₁ y₂ ∧
      H₂ m q y₂ < Hjoint m q y₁ y₂ := by
  constructor
  · simp [interactionGain] at hgain
    linarith [le_max_left (H₁ m q y₁) (H₂ m q y₂)]
  · simp [interactionGain] at hgain
    linarith [le_max_right (H₁ m q y₁) (H₂ m q y₂)]

structure TwoAgentEstimator (m : TwoAgentModel Q Y₁ Y₂ Z) where
  estimate₁ : Q → Y₁ → ℝ
  estimate₂ : Q → Y₂ → ℝ
  estimateJoint : Q → Y₁ → Y₂ → ℝ
  error_bound : ℝ
  hbound₁ : ∀ q y₁, |estimate₁ q y₁ - H₁ m q y₁| ≤ error_bound
  hbound₂ : ∀ q y₂, |estimate₂ q y₂ - H₂ m q y₂| ≤ error_bound
  hboundJoint : ∀ q y₁ y₂,
    |estimateJoint q y₁ y₂ - Hjoint m q y₁ y₂| ≤ error_bound

theorem MAS_benchmark_detects_irreducibility
    (m : TwoAgentModel Q Y₁ Y₂ Z)
    (est : TwoAgentEstimator m)
    (q : Q) (y₁ : Y₁) (y₂ : Y₂)
    (hgain : 2 * est.error_bound < interactionGain m q y₁ y₂) :
    max (est.estimate₁ q y₁) (est.estimate₂ q y₂) <
      est.estimateJoint q y₁ y₂ := by
  have h1_abs := abs_le.mp (est.hbound₁ q y₁)
  have h2_abs := abs_le.mp (est.hbound₂ q y₂)
  have hj_abs := abs_le.mp (est.hboundJoint q y₁ y₂)
  simp [interactionGain] at hgain
  rw [max_lt_iff]
  constructor
  · linarith [le_max_left (H₁ m q y₁) (H₂ m q y₂)]
  · linarith [le_max_right (H₁ m q y₁) (H₂ m q y₂)]

end Creativity.MAS
