import Creativity.CUE.Receiver
import Creativity.MAS.KAgentModel
import Mathlib.Tactic

/-!
# Interaction gain to decision-quality bridge (PROOF-07)

`Gk_implies_CUE_improvement` connects the existing entropy-level irreducibility
result `kAgent_irreducibility` to the receiver decision layer: under a receiver
whose CUE is calibrated to downstream entropy with bounded error `ε_cal`, a
sufficiently large interaction gain forces the joint output to strictly exceed
every single agent in CUE.

The calibration model carries the `brier_entropy_calibration_link` as bounded
error fields, exactly parallel to `KAgentEstimator`; the theorem is the
CUE analogue of `kAgent_benchmark_detects`.
-/

namespace Creativity.CUE.MASBridge

open Creativity.MAS

variable {k : ℕ} {Q Y Z : Type*} [Fintype Z]

/--
A receiver whose CUE readings are calibrated to downstream entropy with uniform
error `εcal`: each agent's CUE tracks its downstream entropy, and the joint CUE
tracks the joint downstream entropy. This is the formal
`brier_entropy_calibration_link`.
-/
structure ReceiverCalibratedMAS (m : KAgentModel k Q Y Z) (q : Q)
    (ys : Fin k → Y) where
  agentCUE : Fin k → ℝ
  jointCUE : ℝ
  εcal : ℝ
  cal_ind : ∀ i, |agentCUE i - agentEntropy m i q (ys i)| ≤ εcal
  cal_joint : |jointCUE - jointEntropy m q ys| ≤ εcal

/--
**PROOF-07** `Gk_implies_CUE_improvement`. Under the receiver calibration link,
if the interaction gain exceeds twice the calibration error then the joint
output achieves strictly higher CUE than every single agent:
`G_k > 2ε_cal ⇒ ∀ i, CUE(y_i) < CUE(y_joint)`.
-/
theorem Gk_implies_CUE_improvement (m : KAgentModel k Q Y Z) (q : Q)
    (ys : Fin k → Y) (cal : ReceiverCalibratedMAS m q ys)
    (hgap : 2 * cal.εcal < kInteractionGain m q ys) :
    ∀ i, cal.agentCUE i < cal.jointCUE := by
  intro i
  have hi := abs_le.mp (cal.cal_ind i)
  have hj := abs_le.mp cal.cal_joint
  have hi_true : agentEntropy m i q (ys i) ≤ maxAgentEntropy m q ys := by
    unfold maxAgentEntropy
    exact Finset.le_sup' (fun i => agentEntropy m i q (ys i)) (Finset.mem_univ i)
  unfold kInteractionGain at hgap
  linarith

/--
The joint output strictly dominates the best single agent in CUE:
`max_i CUE(y_i) < CUE(y_joint)`.
-/
theorem Gk_implies_CUE_improvement_sup (m : KAgentModel k Q Y Z) (q : Q)
    (ys : Fin k → Y) (cal : ReceiverCalibratedMAS m q ys)
    (hgap : 2 * cal.εcal < kInteractionGain m q ys) :
    Finset.univ.sup' (finUnivNonempty m) cal.agentCUE < cal.jointCUE := by
  rw [Finset.sup'_lt_iff (finUnivNonempty m)]
  intro i _
  exact Gk_implies_CUE_improvement m q ys cal hgap i

open Classical in
/--
Task 6: gating `G_k` by the shifted-CUE indicator is compatible with the
existing CUE bridge. Given the bridge implication `0 < G_k → 0 < cue`
(discharged by `Gk_implies_CUE_improvement` in applications), if the
interaction gain is positive and the raw CUE clears the threshold `f`, the
gated term `G_k · 𝟙[CUE_shifted > 0]` remains strictly positive: the gate
does not break the bridge.
-/
theorem Gk_gated_by_CUE_shifted
    {Zc : Type*} [Fintype Zc] [Nonempty Zc]
    (m : CUEModel Zc) (f : ℝ) (Gk_val : ℝ)
    (_hGk_implies : 0 < Gk_val → 0 < cue m)
    (hGkpos : 0 < Gk_val) (hflt : f < cue m) :
    0 < Gk_val * (if 0 < CUE_shifted m f then (1 : ℝ) else 0) := by
  have hshift : 0 < CUE_shifted m f := by
    unfold CUE_shifted; linarith
  simp only [if_pos hshift, mul_one]
  exact hGkpos

end Creativity.CUE.MASBridge
