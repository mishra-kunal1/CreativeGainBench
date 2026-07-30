import Creativity.Probability.Entropy
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Tactic

namespace Creativity.MAS

/--
Directed communication graph on `k` agents. `hasEdge i j` means agent `j`
receives information from agent `i`. The current formal model records
irreflexivity; acyclicity and topology-specific construction are benchmark
protocol constraints rather than theorem assumptions in this finite entropy layer.
-/
structure CommGraph (k : ℕ) where
  hasEdge : Fin k → Fin k → Prop
  irrefl : ∀ i, ¬ hasEdge i i

def graphHasEdge {k : ℕ} (g : CommGraph k) : Prop :=
  ∃ i j : Fin k, g.hasEdge i j

structure KAgentModel (k : ℕ) (Q Y Z : Type*) where
  individualKernel : Fin k → Q → Y → PMF Z
  jointKernel : Q → (Fin k → Y) → PMF Z
  graph : CommGraph k
  /--
  Abstract pairwise information carried by the joint downstream process for
  agent pair `(i,j)` at `(q,ys)`. In a measure-theoretic refinement this is
  mutual information between the corresponding marginals of the joint kernel.
  -/
  pairMutualInfo : Fin k → Fin k → Q → (Fin k → Y) → ℝ
  /--
  Faithfulness assumption: every communication edge has a task/output witness
  with nonzero pairwise information. This rules out graph edges whose statistical
  effect cancels to zero in the observed downstream process.
  -/
  faithful : ∀ i j : Fin k, graph.hasEdge i j →
    ∃ q ys, 0 < pairMutualInfo i j q ys
  hk : 0 < k

variable {k : ℕ} {Q Y Z : Type*} [Fintype Z]

def isProductKernel (m : KAgentModel k Q Y Z) : Prop :=
  ∀ i j : Fin k, i ≠ j → ∀ q ys, m.pairMutualInfo i j q ys = 0

omit [Fintype Z] in
theorem edge_ne_of_hasEdge (m : KAgentModel k Q Y Z)
    {i j : Fin k} (hEdge : m.graph.hasEdge i j) : i ≠ j := by
  intro hij
  subst j
  exact m.graph.irrefl i hEdge

omit [Fintype Z] in
theorem productKernel_implies_pair_mi_zero (m : KAgentModel k Q Y Z)
    (hProd : isProductKernel m) (i j : Fin k) (hij : i ≠ j)
    (q : Q) (ys : Fin k → Y) :
    m.pairMutualInfo i j q ys = 0 :=
  hProd i j hij q ys

omit [Fintype Z] in
theorem nonProduct_base_case (m : KAgentModel 2 Q Y Z)
    (hEdge : m.graph.hasEdge 0 1) :
    ¬ isProductKernel m := by
  intro hProd
  obtain ⟨q, ys, hMI⟩ := m.faithful 0 1 hEdge
  have hij : (0 : Fin 2) ≠ 1 := by decide
  have hzero := productKernel_implies_pair_mi_zero m hProd 0 1 hij q ys
  linarith

omit [Fintype Z] in
theorem connectedMAS_nonProductKernel (m : KAgentModel k Q Y Z)
    (hEdge : graphHasEdge m.graph) :
    ¬ isProductKernel m := by
  intro hProd
  obtain ⟨i, j, hEdgeij⟩ := hEdge
  obtain ⟨q, ys, hMI⟩ := m.faithful i j hEdgeij
  have hij : i ≠ j := edge_ne_of_hasEdge m hEdgeij
  have hzero := productKernel_implies_pair_mi_zero m hProd i j hij q ys
  linarith

def finUnivNonempty (m : KAgentModel k Q Y Z) : (Finset.univ : Finset (Fin k)).Nonempty :=
  ⟨⟨0, m.hk⟩, Finset.mem_univ _⟩

noncomputable def agentEntropy (m : KAgentModel k Q Y Z)
    (i : Fin k) (q : Q) (y : Y) : ℝ :=
  Creativity.Probability.shannonEntropy (m.individualKernel i q y)

noncomputable def jointEntropy (m : KAgentModel k Q Y Z)
    (q : Q) (ys : Fin k → Y) : ℝ :=
  Creativity.Probability.shannonEntropy (m.jointKernel q ys)

noncomputable def maxAgentEntropy (m : KAgentModel k Q Y Z)
    (q : Q) (ys : Fin k → Y) : ℝ :=
  Finset.univ.sup' (finUnivNonempty m) (fun i => agentEntropy m i q (ys i))

noncomputable def kInteractionGain (m : KAgentModel k Q Y Z)
    (q : Q) (ys : Fin k → Y) : ℝ :=
  jointEntropy m q ys - maxAgentEntropy m q ys

/--
**Corrected zero-gain characterization (PROOF-11).** The interaction gain
vanishes iff the joint downstream entropy equals the best single-agent
entropy — the degenerate case in which the joint process adds nothing over the
best individual.

The previous axiom `gain_zero_iff_productKernel`
(`G = 0 ↔ isProductKernel`) was **incorrect**: an independent product kernel
with positive marginal entropies has `H_J = H_1 + H_2 > max(H_1, H_2)` by
entropy additivity (`productKernel_joint_entropy_eq_sum`, PROOF-10), hence
strictly *positive* gain. See `productKernel_additivity_gain_pos` for the
refuting direction. The axiom has been removed and replaced by this proved
characterization.
-/
theorem gain_zero_iff_joint_eq_max (m : KAgentModel k Q Y Z)
    (q : Q) (ys : Fin k → Y) :
    kInteractionGain m q ys = 0 ↔
      jointEntropy m q ys = maxAgentEntropy m q ys := by
  unfold kInteractionGain
  constructor
  · intro h; linarith
  · intro h; linarith

/--
Refutation of the old product-kernel biconditional: for a two-agent coupling
whose joint entropy is additive (the independent product case, supplied by
`productKernel_joint_entropy_eq_sum`) with strictly positive marginal
entropies, the interaction gain is strictly *positive*, not zero.
-/
theorem productKernel_additivity_gain_pos (m : KAgentModel 2 Q Y Z)
    (q : Q) (ys : Fin 2 → Y)
    (hadd : jointEntropy m q ys =
      agentEntropy m 0 q (ys 0) + agentEntropy m 1 q (ys 1))
    (h0 : 0 < agentEntropy m 0 q (ys 0))
    (h1 : 0 < agentEntropy m 1 q (ys 1)) :
    0 < kInteractionGain m q ys := by
  unfold kInteractionGain
  have hmax : maxAgentEntropy m q ys < jointEntropy m q ys := by
    unfold maxAgentEntropy
    rw [Finset.sup'_lt_iff (finUnivNonempty m)]
    intro i _
    fin_cases i <;> simp only [Fin.zero_eta, Fin.mk_one] <;> rw [hadd] <;>
      linarith
  linarith

/-- Corrected gain biconditional: zero gain iff joint entropy is degenerate at
the best single agent; positive gain iff it is not. -/
theorem gain_biconditional (m : KAgentModel k Q Y Z)
    (q : Q) (ys : Fin k → Y)
    (hNonNeg : 0 ≤ kInteractionGain m q ys) :
    (kInteractionGain m q ys = 0 ↔
      jointEntropy m q ys = maxAgentEntropy m q ys) ∧
      (0 < kInteractionGain m q ys ↔
        jointEntropy m q ys ≠ maxAgentEntropy m q ys) := by
  refine ⟨gain_zero_iff_joint_eq_max m q ys, ?_, ?_⟩
  · intro hpos heq
    have hzero := (gain_zero_iff_joint_eq_max m q ys).2 heq
    linarith
  · intro hne
    by_contra hnot
    have hle : kInteractionGain m q ys ≤ 0 := le_of_not_gt hnot
    have hzero : kInteractionGain m q ys = 0 := le_antisymm hle hNonNeg
    exact hne ((gain_zero_iff_joint_eq_max m q ys).1 hzero)

theorem kAgent_irreducibility (m : KAgentModel k Q Y Z)
    (q : Q) (ys : Fin k → Y)
    (hG : 0 < kInteractionGain m q ys) :
    ∀ i : Fin k, agentEntropy m i q (ys i) < jointEntropy m q ys := by
  intro i
  have hsup_lt : maxAgentEntropy m q ys < jointEntropy m q ys := by
    unfold kInteractionGain at hG
    linarith
  have hi_le :
      agentEntropy m i q (ys i) ≤ maxAgentEntropy m q ys := by
    unfold maxAgentEntropy
    exact Finset.le_sup' (fun i => agentEntropy m i q (ys i)) (Finset.mem_univ i)
  exact lt_of_le_of_lt hi_le hsup_lt

structure KAgentEstimator (m : KAgentModel k Q Y Z) (q : Q) (ys : Fin k → Y) where
  estIndividual : Fin k → ℝ
  estJoint : ℝ
  ε : ℝ
  bound_ind : ∀ i, |estIndividual i - agentEntropy m i q (ys i)| ≤ ε
  bound_joint : |estJoint - jointEntropy m q ys| ≤ ε

theorem kAgent_benchmark_detects
    (m : KAgentModel k Q Y Z) (q : Q) (ys : Fin k → Y)
    (est : KAgentEstimator m q ys)
    (hgap : 2 * est.ε < kInteractionGain m q ys) :
    Finset.univ.sup' (finUnivNonempty m) est.estIndividual < est.estJoint := by
  rw [Finset.sup'_lt_iff (finUnivNonempty m)]
  intro i _
  have hi_abs := abs_le.mp (est.bound_ind i)
  have hj_abs := abs_le.mp est.bound_joint
  have hi_true :
      agentEntropy m i q (ys i) ≤ maxAgentEntropy m q ys := by
    unfold maxAgentEntropy
    exact Finset.le_sup' (fun i => agentEntropy m i q (ys i)) (Finset.mem_univ i)
  unfold kInteractionGain at hgap
  linarith

end Creativity.MAS
