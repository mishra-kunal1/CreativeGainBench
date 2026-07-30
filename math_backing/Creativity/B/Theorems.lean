import Creativity.B.Model
import Creativity.Probability.MaxEntropy
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Analysis.Convex.Jensen

namespace Creativity.B.Theorems

open Creativity.B

variable {Q Y Z : Type*} [Fintype Y] [Fintype Z]

omit [Fintype Y] in
theorem B1_dominance (m : BModel Q Y Z) (q : Q) (y₁ y₂ : Y)
    (hf1 : feasible m q y₁) (hf2 : feasible m q y₂)
    (hH : downstreamH m q y₂ < downstreamH m q y₁) :
    rewardB m q y₂ < rewardB m q y₁ := by
  classical
  simpa [rewardB, hf1, hf2] using hH

theorem B2_no_collapse (m : BModel Q Y Z) (q : Q) (y ystar : Y)
    (hf : feasible m q y) (hfstar : feasible m q ystar)
    (hH : downstreamH m q y < downstreamH m q ystar)
    (hS : (feasibleSet m q).Nonempty) :
    rewardB m q y < (feasibleSet m q).sup' hS (rewardB m q) := by
  classical
  exact (Finset.lt_sup'_iff hS).2
    ⟨ystar, by simpa [feasibleSet] using hfstar,
      B1_dominance m q ystar y hfstar hf hH⟩

theorem B3_weighted_high_lb (m : BModel Q Y Z) (q : Q) (π : PMF Y)
    (y_low y_high : Y)
    (_hf_low : feasible m q y_low) (_hf_high : feasible m q y_high)
    (_hH : downstreamH m q y_low < downstreamH m q y_high)
    (hmass :
      rewardB m q y_low ≤
        (π y_high).toReal * rewardB m q y_high) :
    rewardB m q y_low ≤
      ∑ y : Y, (π y).toReal * rewardB m q y := by
  classical
  have hterm_nonneg : ∀ y : Y, 0 ≤ (π y).toReal * rewardB m q y := by
    intro y
    exact mul_nonneg ENNReal.toReal_nonneg (rewardB_nonneg m q y)
  have hhigh_le_sum :
      (π y_high).toReal * rewardB m q y_high ≤
        ∑ y : Y, (π y).toReal * rewardB m q y := by
    rw [← Finset.univ.sum_erase_add _ (Finset.mem_univ y_high)]
    exact le_add_of_nonneg_left
      (Finset.sum_nonneg (fun y _ => hterm_nonneg y))
  exact le_trans hmass hhigh_le_sum

theorem B3_mixture_lb (m : BModel Q Y Z) (q : Q) (π : PMF Y)
    (_hsupp : ∀ y, 0 < π y → feasible m q y)
    (y_low y_high : Y)
    (hf_low : feasible m q y_low)
    (hf_high : feasible m q y_high)
    (hH : downstreamH m q y_low < downstreamH m q y_high)
    (_hmass : 0 < π y_high)
    (hlow_le :
      rewardB m q y_low ≤
        (π y_high).toReal * rewardB m q y_high) :
    rewardB m q y_low ≤
      ∑ y : Y, (π y).toReal * rewardB m q y := by
  classical
  exact B3_weighted_high_lb m q π y_low y_high hf_low hf_high hH hlow_le

theorem B3_expected_exceeds_dominated (m : BModel Q Y Z) (q : Q)
    (π : PMF Y) (y_low y_high : Y)
    (_hf_low : feasible m q y_low) (_hf_high : feasible m q y_high)
    (_hdom : downstreamH m q y_low < downstreamH m q y_high)
    (_hpos : 0 < (π y_high).toReal)
    (hlow_lt :
      rewardB m q y_low <
        (π y_high).toReal * rewardB m q y_high) :
    rewardB m q y_low <
      ∑ y : Y, (π y).toReal * rewardB m q y := by
  classical
  have hterm_nonneg : ∀ y : Y, 0 ≤ (π y).toReal * rewardB m q y := by
    intro y
    exact mul_nonneg ENNReal.toReal_nonneg (rewardB_nonneg m q y)
  have hhigh_le_sum :
      (π y_high).toReal * rewardB m q y_high ≤
        ∑ y : Y, (π y).toReal * rewardB m q y := by
    rw [← Finset.univ.sum_erase_add _ (Finset.mem_univ y_high)]
    exact le_add_of_nonneg_left
      (Finset.sum_nonneg (fun y _ => hterm_nonneg y))
  exact lt_of_lt_of_le hlow_lt hhigh_le_sum

omit [Fintype Z] in
/-- Pointwise evaluation of the mixture kernel: the mass at `z` is the
`π`-average of the component masses. -/
lemma mixtureKernel_apply_toReal (m : BModel Q Y Z) (q : Q) (π : PMF Y)
    (z : Z) :
    ((mixtureKernel m q π) z).toReal =
      ∑ y : Y, (π y).toReal * ((m.kernel q y) z).toReal := by
  unfold mixtureKernel
  rw [PMF.bind_apply, tsum_fintype,
    ENNReal.toReal_sum (fun y _ =>
      ENNReal.mul_ne_top (π.apply_ne_top y) ((m.kernel q y).apply_ne_top z))]
  exact Finset.sum_congr rfl fun y _ => ENNReal.toReal_mul

/--
**Entropy mixture concavity (PROOF-09, promoted from axiom).** Shannon entropy
is concave on finite probability mass functions: the entropy of a downstream
mixture kernel is at least the expected entropy of the component kernels.
Proved from primitives via concave Jensen (`ConcaveOn.le_map_sum`) applied to
`Real.concaveOn_negMulLog`, replacing the former
`entropy_mixture_concavity_axiom`.
-/
theorem entropy_mixture_concavity (m : BModel Q Y Z) (q : Q) (π : PMF Y) :
    ∑ y : Y, (π y).toReal * downstreamH m q y ≤
      Creativity.Probability.shannonEntropy (mixtureKernel m q π) := by
  classical
  rw [Creativity.Probability.shannonEntropy_eq_sum_negMulLog]
  have hswap :
      ∑ y : Y, (π y).toReal * downstreamH m q y =
        ∑ z : Z, ∑ y : Y,
          (π y).toReal * Real.negMulLog ((m.kernel q y) z).toReal := by
    calc
      ∑ y : Y, (π y).toReal * downstreamH m q y
          = ∑ y : Y, ∑ z : Z,
              (π y).toReal * Real.negMulLog ((m.kernel q y) z).toReal := by
            refine Finset.sum_congr rfl fun y _ => ?_
            rw [downstreamH,
              Creativity.Probability.shannonEntropy_eq_sum_negMulLog,
              Finset.mul_sum]
      _ = ∑ z : Z, ∑ y : Y,
            (π y).toReal * Real.negMulLog ((m.kernel q y) z).toReal :=
          Finset.sum_comm
  rw [hswap]
  refine Finset.sum_le_sum fun z _ => ?_
  rw [mixtureKernel_apply_toReal]
  have hjensen := Real.concaveOn_negMulLog.le_map_sum
    (t := (Finset.univ : Finset Y))
    (w := fun y => (π y).toReal)
    (p := fun y => ((m.kernel q y) z).toReal)
    (fun y _ => ENNReal.toReal_nonneg)
    (Creativity.Probability.sum_toReal_eq_one π)
    (fun y _ => Set.mem_Ici.mpr ENNReal.toReal_nonneg)
  simpa [smul_eq_mul] using hjensen

/-- Backward-compatible name for the promoted concavity theorem. The former
axiom `entropy_mixture_concavity_axiom` has been removed. -/
theorem B3b_jensen_anti_collapse (m : BModel Q Y Z) (q : Q) (π : PMF Y) :
    ∑ y : Y, (π y).toReal * downstreamH m q y ≤
      Creativity.Probability.shannonEntropy (mixtureKernel m q π) :=
  entropy_mixture_concavity m q π

end Creativity.B.Theorems
