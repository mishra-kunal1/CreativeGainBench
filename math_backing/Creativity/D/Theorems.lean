import Creativity.D.Deformation
import Mathlib.Tactic

namespace Creativity.D.Theorems

open Creativity.C
open Creativity.D
open ProbeCompressor

theorem D1_utility_gate {α : Type*} [ProbeCompressor α]
    (s corpus : List α) (probes : List (List α)) (u τ : ℝ) (h : u < τ) :
    rewardD s corpus probes u τ = 0 := by
  classical
  simp [rewardD, not_le.mpr h]

/--
`R_D` magnitude tracks the deformation-gain ordering.

Scope note: in the Benchmark Score, `R_D` enters only as the binary gate
`𝟙[R_D > δ_D]` (`Creativity.CUE.dGate`), which discards magnitude — this
ordinal property is therefore *diagnostic-only* there. It remains
load-bearing for the Composability Score and for D-side DPO preference
pairs.
-/
theorem D2_dominance {α : Type*} [ProbeCompressor α]
    (s₁ s₂ corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (hfeas : τ ≤ u)
    (hdom : corpusDeformationGain s₂ corpus probes <
      corpusDeformationGain s₁ corpus probes) :
    rewardD s₂ corpus probes u τ < rewardD s₁ corpus probes u τ := by
  classical
  simp [rewardD, hfeas]
  exact hdom

-- Composability Score only; not used in Benchmark Score after R_C removal.
theorem D3_rewardC_bounded {α : Type*} [AbstractCompressor α]
    (s corpus : List α) (u τ : ℝ) (hfeas : τ ≤ u) :
    rewardC s corpus u τ ≤ AbstractCompressor.L s := by
  classical
  have h : novelty s corpus ≤ AbstractCompressor.L s :=
    AbstractCompressor.copy_minimal s corpus
  simp [rewardC, hfeas]
  exact h

theorem D3_deformation_nonneg {α : Type*} [ProbeCompressor α]
    (s corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (hfeas : τ ≤ u) :
    0 ≤ rewardD s corpus probes u τ := by
  classical
  simp [rewardD, hfeas]
  exact corpusDeformationGain_nonneg s corpus probes

-- Composability Score only; not used in Benchmark Score after R_C removal.
-- The Benchmark Score uses the R_C-free `D3_rewardD_bound_standalone` below.
theorem D3_exploration_transformation_bound {α : Type*} [ProbeCompressor α]
    (s corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (hfeas : τ ≤ u) :
    rewardC s corpus u τ + rewardD s corpus probes u τ ≤
      AbstractCompressor.L s + corpusDeformationGain s corpus probes := by
  classical
  have hC := D3_rewardC_bounded s corpus u τ hfeas
  have hD : rewardD s corpus probes u τ = corpusDeformationGain s corpus probes := by
    simp [rewardD, hfeas]
  linarith

/--
Task 8: R_C-free restatement of the D3 bound for the Benchmark Score. Once the
utility gate is open, `R_D` is bounded by the raw corpus deformation gain
`D(s, H, P)` — no compression-length (`L(s)` / `R_C`) term appears.
-/
theorem D3_rewardD_bound_standalone {α : Type*} [ProbeCompressor α]
    (s corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (hfeas : τ ≤ u) :
    rewardD s corpus probes u τ ≤ corpusDeformationGain s corpus probes := by
  classical
  simp [rewardD, hfeas]

theorem D4_zero_for_corpus_copy {α : Type*} [ProbeCompressor α]
    (corpus : List α) (probes : List (List α)) :
    corpusDeformationGain corpus corpus probes = 0 := by
  induction probes with
  | nil => rfl
  | cons p ps ih =>
    simp [corpusDeformationGain, ih, probeNoveltyDelta, append_self_invariant]

theorem D4_zero_reward_for_corpus_copy {α : Type*} [ProbeCompressor α]
    (corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (hfeas : τ ≤ u) :
    rewardD corpus corpus probes u τ = 0 := by
  classical
  simp [rewardD, hfeas, D4_zero_for_corpus_copy corpus probes]

end Creativity.D.Theorems
