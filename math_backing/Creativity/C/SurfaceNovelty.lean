import Creativity.C.Compression
import Creativity.Probability.KLDivergence
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic

namespace Creativity.C

open Creativity.Probability

/--
Surface novelty via KL divergence between an output n-gram PMF and the corpus PMF.
This is Component N: tractable distributional divergence on a finite alphabet.
-/
noncomputable def surfaceNoveltyKL {α : Type*} [Fintype α]
    (p_output p_corpus : PMF α) : ℝ :=
  klDivergence p_output p_corpus

noncomputable def rewardN {α : Type*} [Fintype α]
    (p_output p_corpus : PMF α) (utility_val threshold : ℝ) : ℝ := by
  classical
  exact if threshold ≤ utility_val then surfaceNoveltyKL p_output p_corpus else 0

end Creativity.C
