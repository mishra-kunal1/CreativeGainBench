import Creativity.C.Theorems
import Creativity.Training.DPOBridge
import Mathlib.Tactic

/-!
# DPO ascent for R_C and the corpus stationarity condition

This file proves that a DPO step on a C-valid preference pair strictly
increases the expected exploratory-novelty reward `E_{y~π_θ}[R_C(q,y)]`,
reusing the generic machinery of `Creativity/Training/DPOBridge.lean`
(`dpo_logit_direction`, `pairwiseTransfer`,
`expectedReward_pairwiseTransfer_eq`) unchanged.

It also states `C_corpus_stationarity_condition`, the formal
Goodhart-resistance condition for R_C: the corpus `H` used to compute `N_KL`
is updated on a fixed external schedule and never by inserting policy outputs.
Violating this condition makes `H` self-referential and breaks the anti-copy
(`C2_copy_penalty`) and anti-padding (`C3_no_padding_exploit`) guarantees,
because the policy could then drive its own outputs into the corpus and
manipulate the novelty baseline it is scored against.
-/

namespace Creativity.C.DPO

open Creativity.C
open Creativity.Training.DPO

variable {Y α : Type*} [Fintype Y] [DecidableEq Y] [AbstractCompressor α]

/--
A C-valid preference pair for prompts with artifact map `artifact : Y → List α`:
both outputs pass the utility gate and the winner has strictly greater
algorithmic novelty against the corpus.
-/
def cValidPair (artifact : Y → List α) (corpus : List α)
    (y_w y_l : Y) (u τ : ℝ) : Prop :=
  τ ≤ u ∧ novelty (artifact y_l) corpus < novelty (artifact y_w) corpus

/-- The per-output C reward induced on policy outputs by the artifact map. -/
noncomputable def rewardCOn (artifact : Y → List α) (corpus : List α)
    (u τ : ℝ) : Y → ℝ :=
  fun y => rewardC (artifact y) corpus u τ

/-- Expected C reward of a (locally represented) policy. -/
noncomputable def expectedRC (artifact : Y → List α) (corpus : List α)
    (u τ : ℝ) (policy : Y → ℝ) : ℝ :=
  expectedReward policy (rewardCOn artifact corpus u τ)

omit [Fintype Y] [DecidableEq Y] in
/-- A C-valid pair is strictly ordered by `R_C`: the gate is open on both
sides (via the feasibility component of `cValidPair`, dual to
`C1_utility_gate`), so novelty ordering transfers to reward ordering. -/
theorem cValidPair_reward_ordered (artifact : Y → List α) (corpus : List α)
    (y_w y_l : Y) (u τ : ℝ)
    (hpair : cValidPair artifact corpus y_w y_l u τ) :
    rewardCOn artifact corpus u τ y_l < rewardCOn artifact corpus u τ y_w := by
  classical
  obtain ⟨hfeas, hnov⟩ := hpair
  simpa [rewardCOn, rewardC, hfeas] using hnov

/--
**DPO ascent for R_C** (H1 non-degeneracy: the local DPO transfer moves
strictly positive mass `δ > 0`; H2 small step: the transfer is pairwise, other
outputs fixed). A DPO step on a C-valid preference pair strictly increases
`E_{y~π_θ}[R_C(q,y)]`.

Proof template reused from `DPO_B_gradient_correct` (Thm 29):
1. C-valid pair gives `R_C(y_w) > R_C(y_l)` via the utility gate and novelty
   ordering (`cValidPair_reward_ordered`);
2. `dpo_logit_direction` (Thm 25) supplies the logit direction unchanged;
3. `expectedReward_pairwiseTransfer_eq` transfers the pairwise mass move to
   the expected reward unchanged;
4. conclude strict increase in `E[R_C]`.
-/
theorem DPO_C_gradient_correct (artifact : Y → List α) (corpus : List α)
    (u τ : ℝ) (policyBefore policyAfter : Y → ℝ)
    (y_w y_l : Y)
    (hpair : cValidPair artifact corpus y_w y_l u τ)
    (hLocalDPO :
      ∃ δ : ℝ, 0 < δ ∧
        policyAfter = pairwiseTransfer policyBefore y_w y_l δ) :
    expectedRC artifact corpus u τ policyBefore <
      expectedRC artifact corpus u τ policyAfter := by
  obtain ⟨δ, hδ, rfl⟩ := hLocalDPO
  unfold expectedRC
  exact dpo_step_increases_expected_RB policyBefore
    (rewardCOn artifact corpus u τ) y_w y_l δ hδ
    (cValidPair_reward_ordered artifact corpus y_w y_l u τ hpair)

/--
Corpus stationarity: the corpus at every training step equals a schedule
`externalSchedule` fixed before training (updates happen only on that external
schedule), and no policy output produced during training ever occurs inside
the corpus it is scored against.
-/
def CorpusStationarity {α : Type*} [DecidableEq α]
    (corpusAt : ℕ → List α)
    (externalSchedule : ℕ → List α)
    (policyOutputs : ℕ → List α) : Prop :=
  (∀ t : ℕ, corpusAt t = externalSchedule t) ∧
  (∀ t t' : ℕ, ¬ (policyOutputs t <:+: corpusAt t'))

/--
A valid R_C training protocol: the corpus trajectory, the pre-registered
external update schedule, the stream of policy outputs, and the
Goodhart-resistance certificate that the corpus is stationary with respect to
the policy.
-/
structure CTrainingProtocol (α : Type*) [DecidableEq α] where
  corpusAt : ℕ → List α
  externalSchedule : ℕ → List α
  policyOutputs : ℕ → List α
  corpus_stationary :
    CorpusStationarity corpusAt externalSchedule policyOutputs

/--
**Goodhart-resistance condition for R_C** (axiom-level protocol requirement):
the corpus `H` used to compute `N_KL` is updated on a fixed external schedule,
never by inserting policy outputs. Violating this axiom breaks the anti-copy
and anti-padding guarantees (`C2_copy_penalty`, `C3_no_padding_exploit`) by
making `H` self-referential: the policy would control the compression baseline
that defines its own novelty. Any protocol admissible for 2-CGRPO carries this
certificate; this theorem exposes it for `Goodhart_resistance_summary`.
-/
theorem C_corpus_stationarity_condition {α : Type*} [DecidableEq α]
    (P : CTrainingProtocol α) :
    CorpusStationarity P.corpusAt P.externalSchedule P.policyOutputs :=
  P.corpus_stationary

end Creativity.C.DPO
