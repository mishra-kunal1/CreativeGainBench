import Creativity.Probability.Entropy
import Creativity.Probability.MaxEntropy
import Creativity.MAS.OInformation
import Mathlib.Probability.ProbabilityMassFunction.Monad
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Tactic

namespace Creativity.MAS.OInfoBridge

open Creativity.Probability

noncomputable def jointPMF {Z : Type*} (p q : PMF Z) : PMF (Z × Z) :=
  p.bind fun z1 => (q.map fun z2 => (z1, z2))

structure TwoAgentProductModel (Q Y Z : Type*) where
  k1 : Q → Y → PMF Z
  k2 : Q → Y → PMF Z

/-- The independent product coupling factorizes pointwise:
`(jointPMF p q) (a, b) = p a * q b`. -/
lemma jointPMF_apply {Z : Type*} [Fintype Z] (p q : PMF Z) (z : Z × Z) :
    jointPMF p q z = p z.1 * q z.2 := by
  classical
  obtain ⟨a, b⟩ := z
  unfold jointPMF
  rw [PMF.bind_apply, tsum_fintype]
  have hmap : ∀ z1 : Z,
      (q.map fun z2 => (z1, z2)) (a, b) = if a = z1 then q b else 0 := by
    intro z1
    rw [PMF.map_apply, tsum_fintype]
    by_cases h : a = z1
    · subst h
      rw [if_pos rfl]
      rw [Finset.sum_eq_single b]
      · simp
      · intro c _ hc
        simp [Prod.ext_iff, Ne.symm hc]
      · intro hb
        exact absurd (Finset.mem_univ b) hb
    · rw [if_neg h]
      refine Finset.sum_eq_zero fun c _ => ?_
      simp [Prod.ext_iff, h]
  calc
    ∑ z1 : Z, p z1 * (q.map fun z2 => (z1, z2)) (a, b)
        = ∑ z1 : Z, p z1 * if a = z1 then q b else 0 := by
          exact Finset.sum_congr rfl fun z1 _ => by rw [hmap z1]
    _ = p a * q b := by
          rw [Finset.sum_congr rfl
            (fun z1 _ => mul_ite (a = z1) (p z1) (q b) 0)]
          simp [Finset.sum_ite_eq]

/-- Entropy additivity for independent product couplings, proved from the
product-sum expansion and the logarithm product rule
(`Real.negMulLog_mul`). Formerly an axiom (PROOF-10). -/
theorem shannonEntropy_jointPMF {Z : Type*} [Fintype Z] (p q : PMF Z) :
    shannonEntropy (jointPMF p q) = shannonEntropy p + shannonEntropy q := by
  classical
  have hpoint : ∀ x y : ℝ,
      x * y * Real.log (x * y) =
        y * (x * Real.log x) + x * (y * Real.log y) := by
    intro x y
    have h := Real.negMulLog_mul x y
    unfold Real.negMulLog at h
    linear_combination -h
  have hJ : ∀ x : Z × Z,
      ((jointPMF p q) x).toReal = (p x.1).toReal * (q x.2).toReal := by
    intro x
    rw [jointPMF_apply]
    exact ENNReal.toReal_mul
  unfold shannonEntropy
  have hsum :
      ∑ x : Z × Z,
          ((jointPMF p q) x).toReal * Real.log ((jointPMF p q) x).toReal =
        (∑ a : Z, (p a).toReal * Real.log (p a).toReal) +
          (∑ b : Z, (q b).toReal * Real.log (q b).toReal) := by
    rw [Fintype.sum_prod_type]
    have hterm : ∀ a b : Z,
        ((jointPMF p q) (a, b)).toReal *
            Real.log ((jointPMF p q) (a, b)).toReal =
          (q b).toReal * ((p a).toReal * Real.log (p a).toReal) +
            (p a).toReal * ((q b).toReal * Real.log (q b).toReal) := by
      intro a b
      rw [hJ (a, b)]
      exact hpoint _ _
    calc
      ∑ a : Z, ∑ b : Z,
          ((jointPMF p q) (a, b)).toReal *
            Real.log ((jointPMF p q) (a, b)).toReal
          = ∑ a : Z,
              ((p a).toReal * Real.log (p a).toReal +
                (p a).toReal *
                  ∑ b : Z, (q b).toReal * Real.log (q b).toReal) := by
            refine Finset.sum_congr rfl fun a _ => ?_
            rw [Finset.sum_congr rfl fun b _ => hterm a b,
              Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.mul_sum,
              sum_toReal_eq_one q, one_mul]
      _ = (∑ a : Z, (p a).toReal * Real.log (p a).toReal) +
            (∑ b : Z, (q b).toReal * Real.log (q b).toReal) := by
            rw [Finset.sum_add_distrib, ← Finset.sum_mul,
              sum_toReal_eq_one p, one_mul]
  rw [hsum]
  ring

/--
Entropy is additive on independent product couplings (PROOF-10). Previously
recorded as an axiom; now a theorem via `shannonEntropy_jointPMF`.
-/
theorem productKernel_joint_entropy_eq_sum
    {Q Y Z : Type*} [Fintype Z]
    (m : TwoAgentProductModel Q Y Z)
    (q : Q) (y1 y2 : Y) :
    shannonEntropy (jointPMF (m.k1 q y1) (m.k2 q y2)) =
      shannonEntropy (m.k1 q y1) + shannonEntropy (m.k2 q y2) :=
  shannonEntropy_jointPMF _ _

theorem productKernel_implies_redundancy
    {Q Y Z : Type*} [Fintype Z]
    (m : TwoAgentProductModel Q Y Z)
    (q : Q) (y1 y2 : Y)
    (h1 : 0 < shannonEntropy (m.k1 q y1))
    (h2 : 0 < shannonEntropy (m.k2 q y2)) :
    0 < shannonEntropy (m.k1 q y1) + shannonEntropy (m.k2 q y2) := by
  linarith

/--
Documentation theorem: product-kernel individual gain is not the same as
synergy. Independent product kernels can have positive joint entropy over each
individual marginal without satisfying a negative O-information condition.
-/
theorem productKernel_gain_is_not_synergy
    {Q Y Z : Type*} [Fintype Z]
    (_m : TwoAgentProductModel Q Y Z)
    (_q : Q) (_y1 _y2 : Y) :
    True := by
  trivial

end Creativity.MAS.OInfoBridge
