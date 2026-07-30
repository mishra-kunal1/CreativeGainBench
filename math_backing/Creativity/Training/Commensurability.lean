import Creativity.B.Model
import Creativity.D.Deformation
import Creativity.Probability.MaxEntropy
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Tactic

namespace Creativity.Training.Commensurability

open Creativity.B
open Creativity.D
open ProbeCompressor

/-!
Normalization weights for the optional additive B–D diagnostic
`benchmarkReward = R_B + λ_D R_D`.

The reported Benchmark Score is multiplicative (`Rcreativity`) and needs no
λ_C. The weight `lambdaD` equalizes maximal D contribution against the B
Shannon ceiling `log|Z|` (`lambda_D_normalization_standalone`).
-/

/-- Canonical weight on D: `log |𝒵| / (|P| · log |Σ| · 𝔼[|s'|])`. -/
noncomputable def lambdaD (Z_card Sigma_card probe_size expected_probe_len : ℝ) : ℝ :=
  Real.log Z_card / (probe_size * Real.log Sigma_card * expected_probe_len)

/-- B reward normalized to unit maximum downstream entropy `log |𝒵|`. -/
noncomputable def normalizedRewardB {Q Y Z : Type*} [Fintype Z]
    (m : BModel Q Y Z) (q : Q) (y : Y) : ℝ :=
  rewardB m q y / Real.log (Fintype.card Z : ℝ)

/-- D term normalized to unit maximum at the reference deformation scale. -/
noncomputable def normalizedRewardD {α : Type*} [ProbeCompressor α]
    (Z_card : ℝ) (lambdaD_val : ℝ)
    (s corpus : List α) (probes : List (List α)) (u τ : ℝ) : ℝ :=
  lambdaD_val * rewardD s corpus probes u τ / Real.log Z_card

theorem normalizedRewardB_eq_one
    {Q Y Z : Type*} [Fintype Z]
    (m : BModel Q Y Z) (q : Q) (y : Y)
    (hfeas : feasible m q y)
    (hmax : downstreamH m q y = Real.log (Fintype.card Z : ℝ))
    (hlogZ : 0 < Real.log (Fintype.card Z : ℝ)) :
    normalizedRewardB m q y = 1 := by
  unfold normalizedRewardB
  rw [rewardB_of_feasible m q y hfeas, hmax]
  exact div_self hlogZ.ne'

theorem normalizedRewardD_eq_one
    {α : Type*} [ProbeCompressor α]
    (Z_card Sigma_card probe_size expected_probe_len : ℝ)
    (s corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (hmax : rewardD s corpus probes u τ =
      probe_size * Real.log Sigma_card * expected_probe_len)
    (hlogZ : 0 < Real.log Z_card)
    (hprobe : 0 < probe_size)
    (hSigma : 0 < Real.log Sigma_card)
    (hlen : 0 < expected_probe_len) :
    normalizedRewardD Z_card (lambdaD Z_card Sigma_card probe_size expected_probe_len)
      s corpus probes u τ = 1 := by
  unfold normalizedRewardD lambdaD
  rw [hmax]
  field_simp

theorem rewardB_le_maxEntropy
    {Q Y Z : Type*} [Fintype Z] [Nonempty Z]
    (m : BModel Q Y Z) (q : Q) (y : Y) :
    rewardB m q y ≤ Real.log (Fintype.card Z : ℝ) := by
  have hlog : 0 ≤ Real.log (Fintype.card Z : ℝ) := by
    apply Real.log_nonneg
    exact_mod_cast Fintype.card_pos
  by_cases hfeas : feasible m q y
  · rw [rewardB_of_feasible m q y hfeas]
    exact Creativity.Probability.maxEntropy_finite_alphabet _
  · rw [rewardB_of_infeasible m q y hfeas]
    exact hlog

/--
R_C-free λ_D normalization. Scaling the maximal D contribution
`|P| · log|Σ| · E[|s'|]` by `λ_D` yields the B ceiling `log|Z|`.
-/
theorem lambda_D_normalization_standalone
    (Z_card Sigma_card probe_size expected_probe_len : ℝ)
    (hprobe : 0 < probe_size)
    (hSigma : 0 < Real.log Sigma_card)
    (hlen : 0 < expected_probe_len) :
    lambdaD Z_card Sigma_card probe_size expected_probe_len *
        (probe_size * Real.log Sigma_card * expected_probe_len) =
      Real.log Z_card := by
  unfold lambdaD
  field_simp

end Creativity.Training.Commensurability
