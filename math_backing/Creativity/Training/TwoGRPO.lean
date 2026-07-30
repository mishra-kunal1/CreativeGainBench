import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
# 2-GRPO advantage quantization (PROOF-16)

In 2-GRPO (GRPO with exactly two rollouts), the group-relative advantages
quantize to `±1/√2`: with rewards `r₁ > r₂`, group mean `μ = (r₁+r₂)/2`, and
group scale `√((r₁-μ)² + (r₂-μ)²)`, the normalized advantages are exactly
`A₁ = 1/√2` and `A₂ = -1/√2`, independent of the reward magnitudes. This is
the formal core of "It Takes Two: Your GRPO Is Secretly DPO" (Wu et al.):
the two-rollout advantage carries only the *signed rank* of the pair, making
it an unbiased signed-rank signal.

Main results:
* `groupScale_eq`: the group scale equals `|r₁ - r₂|/√2`.
* `advantage_antisymm`: the two advantages are exact negatives.
* `2GRPO_advantage_quantization_unbiased`: `A₁ = 1/√2`, `A₂ = -1/√2`
  whenever `r₂ < r₁` — the advantage is a deterministic function of the rank
  alone, hence (trivially) an unbiased estimator of the signed rank.
-/

namespace Creativity.Training.TwoGRPO

/-- Group mean of the two rollout rewards. -/
noncomputable def groupMean (r₁ r₂ : ℝ) : ℝ := (r₁ + r₂) / 2

/-- Group scale: the (unnormalized) ℓ² deviation of the pair from its mean. -/
noncomputable def groupScale (r₁ r₂ : ℝ) : ℝ :=
  Real.sqrt ((r₁ - groupMean r₁ r₂) ^ 2 + (r₂ - groupMean r₁ r₂) ^ 2)

/-- Group-relative advantage of the first rollout. -/
noncomputable def advantage1 (r₁ r₂ : ℝ) : ℝ :=
  (r₁ - groupMean r₁ r₂) / groupScale r₁ r₂

/-- Group-relative advantage of the second rollout. -/
noncomputable def advantage2 (r₁ r₂ : ℝ) : ℝ :=
  (r₂ - groupMean r₁ r₂) / groupScale r₁ r₂

/-- The two-point group scale collapses to `|r₁ - r₂| / √2`. -/
theorem groupScale_eq (r₁ r₂ : ℝ) :
    groupScale r₁ r₂ = |r₁ - r₂| / Real.sqrt 2 := by
  unfold groupScale groupMean
  have hsum : (r₁ - (r₁ + r₂) / 2) ^ 2 + (r₂ - (r₁ + r₂) / 2) ^ 2 =
      (r₁ - r₂) ^ 2 / 2 := by ring
  rw [hsum, Real.sqrt_div (sq_nonneg _), Real.sqrt_sq_eq_abs]

/-- The two advantages are exact negatives of each other. -/
theorem advantage_antisymm (r₁ r₂ : ℝ) :
    advantage2 r₁ r₂ = -advantage1 r₁ r₂ := by
  unfold advantage1 advantage2 groupMean
  have h : r₂ - (r₁ + r₂) / 2 = -(r₁ - (r₁ + r₂) / 2) := by ring
  rw [h, neg_div]

/--
**PROOF-16** `2GRPO_advantage_quantization_unbiased`. With two rollouts and
`r₂ < r₁`, the group-relative advantages quantize exactly to `±1/√2`,
independent of the reward magnitudes. The advantage is therefore a
deterministic function of the signed rank of each rollout, i.e. an unbiased
signed-rank estimator: `E[Aᵢ | rank(yᵢ)] = ±1/√2`.
-/
theorem twoGRPO_advantage_quantization_unbiased (r₁ r₂ : ℝ) (h : r₂ < r₁) :
    advantage1 r₁ r₂ = 1 / Real.sqrt 2 ∧
      advantage2 r₁ r₂ = -(1 / Real.sqrt 2) := by
  have hd : 0 < r₁ - r₂ := by linarith
  have habs : |r₁ - r₂| = r₁ - r₂ := abs_of_pos hd
  have hs2 : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hsq : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  have hA1 : advantage1 r₁ r₂ = 1 / Real.sqrt 2 := by
    unfold advantage1
    rw [groupScale_eq, habs]
    unfold groupMean
    have hnum : r₁ - (r₁ + r₂) / 2 = (r₁ - r₂) / 2 := by ring
    rw [hnum]
    have hd' : r₁ - r₂ ≠ 0 := hd.ne'
    field_simp
    nlinarith [hsq]
  refine ⟨hA1, ?_⟩
  rw [advantage_antisymm, hA1]

end Creativity.Training.TwoGRPO
