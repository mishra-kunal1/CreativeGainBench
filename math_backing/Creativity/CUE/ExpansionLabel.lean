import Creativity.CUE.Receiver
import Mathlib.Tactic

/-!
# Receiver expansion vs. clarification labels (PROOF-06)

The bidirectional receiver label `R_B^{→A}` records whether an output *expands*
the receiver's downstream generative space (high receiver entropy) or
*clarifies* it (low receiver entropy). The separation theorem
`receiver_expansion_clarification_separation` shows this label is **not** a
monotone function of CUE: two outputs can have identical CUE while carrying
opposite expansion/clarification labels.

The construction mirrors `BC_tension_example`: explicit finite witnesses with
numeric label values (an expansion magnitude `1` standing for maximal
`log|Z|`-scale expansion, and `0` standing for a fully-committed point-mass
receiver), while the CUE values are genuinely equal because both outputs share
the same underlying `CUEModel`.
-/

namespace Creativity.CUE

open Creativity.Probability

/-! ## Normalized receiver expansion (Task 4)

`R_B^{→A}` is defined as the normalized entropy ratio
`H(Gen(A | y, T)) / log|W|` of the receiver's downstream generative
distribution over its generation space `W`. Normalization makes the label
universally bounded in `[0, 1]`, which the Benchmark Score multiplier
`1 + α·R_B^{→A}` relies on (`Rcreativity_sign_matches_CUE_shifted`).
-/

/-- The normalized receiver-expansion label: entropy of the receiver's
generative distribution divided by the maximum entropy `log|W|`. -/
noncomputable def receiverExpansionNorm {W : Type*} [Fintype W] [Nonempty W]
    (gen : PMF W) : ℝ :=
  shannonEntropy gen / Real.log (Fintype.card W : ℝ)

/-- The normalized expansion label is nonnegative. -/
theorem receiverExpansionNorm_nonneg {W : Type*} [Fintype W] [Nonempty W]
    (gen : PMF W) :
    0 ≤ receiverExpansionNorm gen :=
  div_nonneg (shannonEntropy_nonneg gen)
    (Real.log_nonneg (by exact_mod_cast Fintype.card_pos))

/-- The normalized expansion label is at most one, by
`maxEntropy_finite_alphabet`. -/
theorem receiverExpansionNorm_le_one {W : Type*} [Fintype W] [Nonempty W]
    (gen : PMF W) :
    receiverExpansionNorm gen ≤ 1 := by
  unfold receiverExpansionNorm
  rcases eq_or_lt_of_le
      (Real.log_nonneg
        (show (1 : ℝ) ≤ (Fintype.card W : ℝ) by exact_mod_cast Fintype.card_pos))
    with hzero | hpos
  · -- degenerate one-element alphabet: entropy 0 over log 1 = 0
    rw [← hzero, div_zero]
    exact zero_le_one
  · exact (div_le_one hpos).mpr (maxEntropy_finite_alphabet gen)

/--
Task 4: `receiver_expansion_bounded`. The receiver-expansion label
`R_B^{→A}` is universally bounded in `[0, 1]` — not just at the two witness
points of the separation construction below.
-/
theorem receiver_expansion_bounded {W : Type*} [Fintype W] [Nonempty W]
    (gen : PMF W) :
    0 ≤ receiverExpansionNorm gen ∧ receiverExpansionNorm gen ≤ 1 :=
  ⟨receiverExpansionNorm_nonneg gen, receiverExpansionNorm_le_one gen⟩

/-- An output annotated with its receiver-expansion label
`R_B^{→A} = H(Gen(A | y))`. -/
structure LabeledOutput (Z : Type*) [Fintype Z] [Nonempty Z] where
  model : CUEModel Z
  receiverExpansion : ℝ
  expansion_nonneg : 0 ≤ receiverExpansion

/-- Build a labeled output from a receiver generative distribution, using the
normalized expansion label; nonnegativity is discharged by
`receiverExpansionNorm_nonneg`. -/
noncomputable def LabeledOutput.ofGen {Z W : Type*} [Fintype Z] [Nonempty Z]
    [Fintype W] [Nonempty W] (model : CUEModel Z) (gen : PMF W) :
    LabeledOutput Z :=
  ⟨model, receiverExpansionNorm gen, receiverExpansionNorm_nonneg gen⟩

variable {Z : Type*} [Fintype Z] [Nonempty Z]

/-- Expanding label: receiver entropy exceeds the expansion threshold `δE`. -/
def isExpanding (o : LabeledOutput Z) (δE : ℝ) : Prop :=
  δE < o.receiverExpansion

/-- Clarifying label: receiver entropy falls below `δE`. -/
def isClarifying (o : LabeledOutput Z) (δE : ℝ) : Prop :=
  o.receiverExpansion < δE

/-- A base CUE model on `Fin 2` with zero decision-quality gain and unit
bit length. Used only as the shared CUE carrier for the separation witnesses. -/
noncomputable def baseModelFin2 : CUEModel (Fin 2) where
  posterior := PMF.uniformOfFintype (Fin 2)
  brierDelta := 0
  voi := 0
  bitLength := 1
  brier_nonneg := le_refl 0
  bitLength_pos := one_pos
  brier_le_voi := le_refl 0
  voi_le_entropy := shannonEntropy_nonneg _

/--
**PROOF-06** `receiver_expansion_clarification_separation`. The receiver
expansion label is not a monotone function of CUE: there exist outputs with
equal CUE but opposite expansion/clarification labels — one expanding (high
receiver entropy) and one clarifying (low receiver entropy).

Scope note: the separate-categorical-reporting conclusion drawn from this
theorem applies to the Composability Score. The Benchmark Score
(`Creativity.CUE.benchmarkScore`) deliberately folds the label in
multiplicatively via `(1 + α·R_B^{→A})` as a documented scope exception:
the separation theorem itself (non-monotonicity in CUE) remains true and is
untouched by that choice.
-/
theorem receiver_expansion_clarification_separation :
    ∃ (o_expand o_clarify : LabeledOutput (Fin 2)) (δE : ℝ),
      cue o_expand.model = cue o_clarify.model ∧
        isExpanding o_expand δE ∧ isClarifying o_clarify δE := by
  refine ⟨⟨baseModelFin2, 1, zero_le_one⟩, ⟨baseModelFin2, 0, le_refl 0⟩,
    1 / 2, ?_, ?_, ?_⟩
  · rfl
  · show (1 : ℝ) / 2 < 1; norm_num
  · show (0 : ℝ) < 1 / 2; norm_num

end Creativity.CUE
