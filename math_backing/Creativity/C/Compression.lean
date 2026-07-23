import Mathlib.Data.List.Basic
import Mathlib.Data.Real.Basic

/-!
Infrastructure for probe-relative deformation (`ProbeCompressor` extends this).
`rewardC` is *not* part of \(R_{\mathrm{creativity}}\); it is retained only so
`novelty` / code-length axioms underpin `Creativity.D`. The
`AbstractCompressor` / `NormalizedCompressor` classes remain the formal
substrate of the C-side theorems (C1–C3, CBase) and of the D-side
`ProbeCompressor`, which the Benchmark Score still uses.
-/

namespace Creativity.C

class AbstractCompressor (α : Type*) where
  codeLength : List α → ℝ
  nonneg : ∀ s : List α, 0 ≤ codeLength s
  copy_minimal : ∀ (s corpus : List α),
    (codeLength (corpus ++ s)) - (codeLength corpus) ≤ codeLength s

namespace AbstractCompressor

noncomputable abbrev L {α : Type*} [AbstractCompressor α] : List α → ℝ :=
  AbstractCompressor.codeLength

end AbstractCompressor

noncomputable def novelty {α : Type*} [AbstractCompressor α]
    (s corpus : List α) : ℝ :=
  (AbstractCompressor.L (corpus ++ s)) - (AbstractCompressor.L corpus)

noncomputable def rewardC {α : Type*} [AbstractCompressor α]
    (s corpus : List α) (utility_val threshold : ℝ) : ℝ := by
  classical
  exact if threshold ≤ utility_val then novelty s corpus else 0

class NormalizedCompressor (α : Type*) extends AbstractCompressor α where
  append_padding_bound : ∀ (s pad corpus : List α),
    (AbstractCompressor.L (corpus ++ (s ++ pad))) ≤
      (AbstractCompressor.L (corpus ++ s)) + (AbstractCompressor.L pad)

end Creativity.C
