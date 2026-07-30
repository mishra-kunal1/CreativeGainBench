import Creativity.D.ProxyReward
import Creativity.Training.DPOBridge
import Mathlib.Tactic

/-!
# DPO ascent for the proxy reward R̃_D

A DPO step on a D̃-valid preference pair strictly increases the expected proxy
reward `E_{y~π_θ}[R̃_D(q,y)]`. The proof template is identical to
`DPO_B_gradient_correct` (Thm 29) and `DPO_C_gradient_correct`
(`Creativity/C/DPO.lean`), substituting `D_tilde_dominance`
(`Creativity/D/ProxyReward.lean`) for `B1_dominance`; the generic pieces
`dpo_logit_direction` (Thm 25) and `expectedReward_pairwiseTransfer_eq` are
reused unchanged.
-/

namespace Creativity.D.DPOTilde

open Creativity.D.ProxyReward
open Creativity.Training.DPO

variable {Q S Z Y : Type*} [Fintype Z] [Fintype Y] [DecidableEq Y]

/-- The per-output proxy reward induced on policy outputs by the artifact map
`artifact : Y → S` and the freshly sampled probe prompt `qprobe`. -/
noncomputable def rewardDTildeOn (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : Y → S) (u τ : ℝ) : Y → ℝ :=
  fun y => rewardDTilde m qprobe (artifact y) u τ

/-- Expected proxy reward of a (locally represented) policy. -/
noncomputable def expectedRDTilde (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : Y → S) (u τ : ℝ) (policy : Y → ℝ) : ℝ :=
  expectedReward policy (rewardDTildeOn m qprobe artifact u τ)

/--
**DPO ascent for R̃_D** (H1 non-degeneracy: the local DPO transfer moves
strictly positive mass `δ > 0`; H2 small step: the transfer is pairwise, other
outputs fixed). A DPO step on a D̃-valid preference pair strictly increases
`E_{y~π_θ}[R̃_D(q,y)]`.

The preference pair is D̃-valid via `D_tilde_dominance`
(`dtildeValidPair m qprobe (artifact y_w) (artifact y_l) u τ`), and the
conclusion follows from `expectedReward_pairwiseTransfer_eq` exactly as in
`DPO_B_gradient_correct`.
-/
theorem DPO_D_tilde_gradient_correct (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : Y → S) (u τ : ℝ)
    (policyBefore policyAfter : Y → ℝ)
    (y_w y_l : Y)
    (hpair : dtildeValidPair m qprobe (artifact y_w) (artifact y_l) u τ)
    (hLocalDPO :
      ∃ δ : ℝ, 0 < δ ∧
        policyAfter = pairwiseTransfer policyBefore y_w y_l δ) :
    expectedRDTilde m qprobe artifact u τ policyBefore <
      expectedRDTilde m qprobe artifact u τ policyAfter := by
  obtain ⟨δ, hδ, rfl⟩ := hLocalDPO
  unfold expectedRDTilde
  exact dpo_step_increases_expected_RB policyBefore
    (rewardDTildeOn m qprobe artifact u τ) y_w y_l δ hδ hpair.2

end Creativity.D.DPOTilde
