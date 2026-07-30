import Creativity.C.SurfaceNovelty
import Mathlib.Tactic

namespace Creativity.C.SurfaceTheorems

open Creativity.C
open Creativity.Probability

theorem N1_utility_gate {α : Type*} [Fintype α]
    (p_output p_corpus : PMF α) (u τ : ℝ) (h : u < τ) :
    rewardN p_output p_corpus u τ = 0 := by
  classical
  simp [rewardN, not_le.mpr h]

theorem N2_kl_nonneg {α : Type*} [Fintype α]
    (p_output p_corpus : PMF α) :
    0 ≤ surfaceNoveltyKL p_output p_corpus := by
  simpa [surfaceNoveltyKL] using klDivergence_nonneg p_output p_corpus

theorem N2_reward_nonneg {α : Type*} [Fintype α]
    (p_output p_corpus : PMF α) (u τ : ℝ)
    (hfeas : τ ≤ u) :
    0 ≤ rewardN p_output p_corpus u τ := by
  classical
  simp [rewardN, hfeas]
  exact klDivergence_nonneg p_output p_corpus

/--
Algorithmic novelty upper-bounds surface KL plus output entropy. This formalizes
$N(s\mid\mathcal{H})\approx L(s\mid\mathcal{H})\approx H(p_s)+D_{\mathrm{KL}}(p_s\|p_{\mathcal{H}})$.
The link is axiomatized because mapping strings to PMFs is left abstract.
-/
theorem N3_kl_component_bound {α : Type*} [Fintype α]
    (p_output p_corpus : PMF α) (novelty_bound : ℝ)
    (h : surfaceNoveltyKL p_output p_corpus + shannonEntropy p_output ≤ novelty_bound) :
    surfaceNoveltyKL p_output p_corpus ≤ novelty_bound := by
  linarith [klDivergence_nonneg p_output p_corpus, shannonEntropy_nonneg p_output]

end Creativity.C.SurfaceTheorems
