import Creativity.B.Theorems
import Mathlib.Tactic

/-!
# R_B entropy-incoherence bound

The existing B theorems (`B1_dominance`, `B2_no_collapse`,
`B3_expected_exceeds_dominated`) prove anti-collapse but do not by themselves
block *entropy-maximizing incoherence*: outputs that achieve high downstream
entropy `H_q(y)` by being vague rather than expansive. If such an output slips
past the feasibility gate, R_B rewards its entropy like any other.

This file states the calibration requirement that closes the gap:

`B_utility_gate_tightness` — the utility threshold τ must satisfy
`τ > H_q(y_random)`, where `y_random` is a feasibility-passing but
semantically incoherent output. τ must be set above the entropy of incoherent
baseline outputs, not just above zero, and this calibration must be validated
against incoherent high-entropy baselines *before training*. Without it, R_B
can be Goodharted by entropy-maximizing incoherence.
-/

namespace Creativity.B.CoherenceBound

open Creativity.B

variable {Q Y Z : Type*} [Fintype Z]

/--
An incoherent high-entropy baseline: an output that passes the feasibility
gate (so R_B does not zero it out) while being semantically incoherent —
its entropy comes from vagueness, not epistemic expansiveness. Produced by the
τ-calibration study before training.
-/
structure IncoherentBaseline (m : BModel Q Y Z) (q : Q) where
  yRandom : Y
  passes_gate : feasible m q yRandom

/--
Gate tightness for an incoherent baseline: the utility threshold τ
(`m.threshold`, commensurable with entropy in nats after the normalization of
`Creativity/Training/Commensurability.lean`) strictly exceeds the downstream
entropy `H_q(y_random)` of the incoherent output.
-/
def BUtilityGateTight (m : BModel Q Y Z) (q : Q)
    (base : IncoherentBaseline m q) : Prop :=
  downstreamH m q base.yRandom < m.threshold

/--
A τ-calibration certificate: for every incoherent baseline exhibited by the
calibration study, τ exceeds its entropy. This is what a valid 2-CGRPO run
must produce before training begins.
-/
structure TauCalibration (m : BModel Q Y Z) (q : Q) where
  gate_tight : ∀ base : IncoherentBaseline m q, BUtilityGateTight m q base

/--
**Goodhart-resistance condition for R_B** (calibration requirement axiom):
the utility threshold τ satisfies `τ > H_q(y_random)` for every
feasibility-passing but semantically incoherent output `y_random` — τ is
calibrated above the entropy of incoherent baseline outputs, not just above
zero. Any protocol admissible for 2-CGRPO carries this certificate; this
theorem exposes it for `Goodhart_resistance_summary`.
-/
theorem B_utility_gate_tightness (m : BModel Q Y Z) (q : Q)
    (cal : TauCalibration m q) :
    ∀ base : IncoherentBaseline m q, BUtilityGateTight m q base :=
  cal.gate_tight

/--
Consequence of gate tightness: the R_B reward of any incoherent baseline is
strictly below the utility threshold τ. Entropy-maximizing incoherence cannot
push R_B to or past the calibrated bar, so it cannot dominate a genuinely
expansive output whose entropy reaches τ.
-/
theorem B_incoherence_reward_below_threshold (m : BModel Q Y Z) (q : Q)
    (cal : TauCalibration m q) (base : IncoherentBaseline m q) :
    rewardB m q base.yRandom < m.threshold := by
  rw [rewardB_of_feasible m q base.yRandom base.passes_gate]
  exact cal.gate_tight base

/--
Rank consequence: under a tight gate, any feasible output whose downstream
entropy reaches the calibrated threshold τ strictly dominates every incoherent
baseline in R_B.
-/
theorem B_coherent_dominates_incoherent (m : BModel Q Y Z) (q : Q)
    (cal : TauCalibration m q) (base : IncoherentBaseline m q)
    (y : Y) (hfeas : feasible m q y)
    (hH : m.threshold ≤ downstreamH m q y) :
    rewardB m q base.yRandom < rewardB m q y := by
  rw [rewardB_of_feasible m q base.yRandom base.passes_gate,
    rewardB_of_feasible m q y hfeas]
  exact lt_of_lt_of_le (cal.gate_tight base) hH

end Creativity.B.CoherenceBound
