import Creativity.C.Compression
import Mathlib.Tactic

/-!
Composability Score only; not used in Benchmark Score after R_C removal.

The C1/C2/C3 theorems below remain valid and are retained for the
Composability Score, but the active benchmark pipeline no longer includes an
`R_C` term: its D-side content is carried by the R_C-free restatements
`D3_rewardD_bound_standalone` and `lambda_D_normalization_standalone`.
-/

namespace Creativity.C.Theorems

open Creativity.C

theorem C1_utility_gate {α : Type*} [AbstractCompressor α]
    (s corpus : List α) (u τ : ℝ) (h : u < τ) :
    rewardC s corpus u τ = 0 := by
  classical
  simp [rewardC, not_le.mpr h]

theorem C2_copy_penalty {α : Type*} [AbstractCompressor α]
    (corpus : List α) :
    novelty corpus corpus ≤ AbstractCompressor.L corpus := by
  simpa [novelty] using
    (AbstractCompressor.copy_minimal (s := corpus) (corpus := corpus))

theorem C3_no_padding_exploit {α : Type*} [NormalizedCompressor α]
    (s pad corpus : List α) :
    novelty (s ++ pad) corpus ≤ novelty s corpus + AbstractCompressor.L pad := by
  have h := NormalizedCompressor.append_padding_bound s pad corpus
  simp only [novelty, List.append_assoc]
  linarith

theorem C3_no_trivial_padding_gain {α : Type*} [NormalizedCompressor α]
    (s pad corpus : List α) (h_pad_trivial : AbstractCompressor.L pad = 0) :
    novelty (s ++ pad) corpus ≤ novelty s corpus := by
  have h := C3_no_padding_exploit (s := s) (pad := pad) (corpus := corpus)
  simpa [h_pad_trivial] using h

end Creativity.C.Theorems
