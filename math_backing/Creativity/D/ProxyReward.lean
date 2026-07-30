import Creativity.Probability.KLDivergence
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic

/-!
# The trainable proxy reward R̃_D

`R_D` (probe-relative structural influence on a frozen hold-out probe set `P`)
is a benchmark-only quantity: it must never receive gradient updates, or the
frozen probe set leaks into training and every rank-preservation guarantee is
voided. Training instead uses the proxy

  R̃_D(s, q_probe) = D_KL( p(· | q_probe, s) ‖ p_ref(· | q_probe) ) · 1[u ≥ τ]

where `q_probe` is drawn fresh from a probe prompt distribution every batch and
`p_ref` is a frozen reference policy. This file defines the proxy model and
proves its Goodhart-resistance lemmas:

- `D_tilde_utility_gate` (mirrors `D1_utility_gate`)
- `D_tilde_nonneg` (KL nonnegativity through the gate)
- `D_tilde_zero_for_identical` (mirrors `D4_zero_for_corpus_copy`)
- `D_tilde_no_fixed_target` (the formal Goodhart-resistance condition:
  no probe prompt is fixed across all training steps)
- `D_tilde_dominance` (preference ordering; prerequisite for the DPO theorems
  in `Creativity/D/DPO_tilde.lean`)
-/

namespace Creativity.D.ProxyReward

open Creativity.Probability

/--
The proxy-reward model for R̃_D.

* `Qprobe` — probe prompt distribution; resampled per batch, never frozen.
* `p_ref` — frozen reference policy `Q → PMF Z`; never updated during training.
* `p_cond` — the trained policy's predictive distribution on the probe prompt,
  conditioned on having produced output `s : S`.
-/
structure DTildeModel (Q S Z : Type*) [Fintype Z] where
  Qprobe : PMF Q
  p_ref : Q → PMF Z
  p_cond : Q → S → PMF Z

variable {Q S Z : Type*} [Fintype Z]

/-- The KL shift induced by output `s` on probe prompt `qprobe`. -/
noncomputable def klShift (m : DTildeModel Q S Z) (qprobe : Q) (s : S) : ℝ :=
  klDivergence (m.p_cond qprobe s) (m.p_ref qprobe)

/--
The trainable proxy reward:
`R̃_D(s, q_probe) = D_KL(p(·|q_probe, s) ‖ p_ref(·|q_probe)) · 1[u ≥ τ]`.
-/
noncomputable def rewardDTilde (m : DTildeModel Q S Z) (qprobe : Q) (s : S)
    (utility_val threshold : ℝ) : ℝ := by
  classical
  exact if threshold ≤ utility_val then klShift m qprobe s else 0

/-- Utility gate: infeasible outputs earn zero proxy reward.
Mirrors `D1_utility_gate` (Thm 15). -/
theorem D_tilde_utility_gate (m : DTildeModel Q S Z) (qprobe : Q) (s : S)
    (u τ : ℝ) (h : u < τ) :
    rewardDTilde m qprobe s u τ = 0 := by
  classical
  simp [rewardDTilde, not_le.mpr h]

/-- The proxy reward is nonnegative: it is a gated KL divergence. -/
theorem D_tilde_nonneg (m : DTildeModel Q S Z) (qprobe : Q) (s : S)
    (u τ : ℝ) :
    0 ≤ rewardDTilde m qprobe s u τ := by
  classical
  unfold rewardDTilde
  split
  · exact klDivergence_nonneg _ _
  · exact le_refl 0

/-- KL divergence of a PMF against itself vanishes term by term. -/
lemma klDivergence_self {α : Type*} [Fintype α] (p : PMF α) :
    klDivergence p p = 0 := by
  unfold klDivergence
  apply Finset.sum_eq_zero
  intro x _
  by_cases h : (p x).toReal = 0
  · simp [h]
  · simp [h, div_self h]

/--
If the conditioned policy matches the frozen reference on the probe prompt,
the proxy reward is zero: reproducing the reference distribution earns nothing.
Mirrors `D4_zero_for_corpus_copy` (Thm 19).
-/
theorem D_tilde_zero_for_identical (m : DTildeModel Q S Z) (qprobe : Q) (s : S)
    (u τ : ℝ) (hid : m.p_cond qprobe s = m.p_ref qprobe) :
    rewardDTilde m qprobe s u τ = 0 := by
  classical
  unfold rewardDTilde klShift
  rw [hid, klDivergence_self]
  simp

/--
No probe prompt is fixed across training: for every prompt `q` there is no
guarantee that `q` appears in every batch — formally, no `q` belongs to the
probe batch at every training step.
-/
def NoFixedTarget {Q : Type*} (probeBatch : ℕ → Finset Q) : Prop :=
  ∀ q : Q, ¬ ∀ t : ℕ, q ∈ probeBatch t

/--
A valid R̃_D training protocol: the probe batch drawn at each training step,
together with the Goodhart-resistance certificate that the batch is resampled
fresh — no probe string is held fixed across all steps, so the policy cannot
memorize specific probe inputs.
-/
structure DTildeTrainingProtocol (Q : Type*) where
  probeBatch : ℕ → Finset Q
  no_fixed_target : NoFixedTarget probeBatch

/--
**Goodhart-resistance condition for R̃_D** (axiom-level protocol requirement):
`Q_probe` is drawn fresh each step — no string in `Q_probe` is fixed across
training steps. Any protocol admissible for 2-CGRPO carries this certificate;
this theorem exposes it for the summary conjunction
`Goodhart_resistance_summary`.
-/
theorem D_tilde_no_fixed_target {Q : Type*} (P : DTildeTrainingProtocol Q) :
    NoFixedTarget P.probeBatch :=
  P.no_fixed_target

/--
A D̃-valid preference pair: both outputs pass the utility gate and the winner
strictly exceeds the loser in proxy reward.
-/
def dtildeValidPair (m : DTildeModel Q S Z) (qprobe : Q)
    (s_w s_l : S) (u τ : ℝ) : Prop :=
  τ ≤ u ∧ rewardDTilde m qprobe s_l u τ < rewardDTilde m qprobe s_w u τ

/--
D̃ preference ordering: feasible outputs are strictly ordered by their KL shift.
Mirrors `B1_dominance` (Thm 8) and `D2_dominance` (Thm 16).
-/
theorem D_tilde_reward_dominance (m : DTildeModel Q S Z) (qprobe : Q)
    (s₁ s₂ : S) (u τ : ℝ) (hfeas : τ ≤ u)
    (hdom : klShift m qprobe s₁ < klShift m qprobe s₂) :
    rewardDTilde m qprobe s₁ u τ < rewardDTilde m qprobe s₂ u τ := by
  classical
  simpa [rewardDTilde, hfeas] using hdom

/--
If `u ≥ τ` and `R̃_D(s₁) < R̃_D(s₂)` then `(s₂, s₁)` is a D̃-valid preference
pair. This is the prerequisite for `DPO_D_tilde_gradient_correct`.
-/
theorem D_tilde_dominance (m : DTildeModel Q S Z) (qprobe : Q)
    (s₁ s₂ : S) (u τ : ℝ) (hfeas : τ ≤ u)
    (hlt : rewardDTilde m qprobe s₁ u τ < rewardDTilde m qprobe s₂ u τ) :
    dtildeValidPair m qprobe s₂ s₁ u τ :=
  ⟨hfeas, hlt⟩

end Creativity.D.ProxyReward
