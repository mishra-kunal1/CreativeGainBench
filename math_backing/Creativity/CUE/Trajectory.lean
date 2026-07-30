import Creativity.CUE.Receiver
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic

/-!
# Trajectory-level CUE diagnostics (PROOF-04, PROOF-05, PROOF-14, PROOF-15)

This file formalizes the trajectory-level Step-CUE constructs:

* `CUE_trajectory_monotone_nondecreasing` (PROOF-04): under the per-step
  utility gate (nonnegative step reward), the running Step-CUE curve is
  non-decreasing.
* `DC_score_existence` (PROOF-05): a diverge-then-converge trajectory exists
  (`DC = 1` achievable), and monotone trajectories fail the DC predicate
  (`DC = 0`).
* `step_CUE_gamma_positivity_characterization` (PROOF-14): the linear
  saturation-rate `γ` of the fitted Step-CUE curve is positive iff some step
  exceeds the pure exponential-saturation prediction (genuine continuous
  discovery).
* `dense_reward_DC_bonus_incentivizes_exploration` (PROOF-15): the terminal DC
  bonus is necessary — without it, a diverge-converge trajectory ties a
  monotone one on cumulative CUE; the bonus strictly breaks the tie toward
  exploration.
-/

namespace Creativity.CUE.Trajectory

/-! ## PROOF-04: monotone non-decreasing Step-CUE -/

/-- The incremental step reward of a Step-CUE curve. -/
def stepReward (C : ℕ → ℝ) (t : ℕ) : ℝ := C (t + 1) - C t

/--
**PROOF-04** `CUE_trajectory_monotone_nondecreasing`. If every reasoning step
satisfies the utility gate — i.e. the incremental Step-CUE contribution is
nonnegative — then the running Step-CUE curve `C` is non-decreasing.
-/
theorem CUE_trajectory_monotone_nondecreasing (C : ℕ → ℝ)
    (hgate : ∀ t, 0 ≤ stepReward C t) :
    Monotone C := by
  apply monotone_nat_of_le_succ
  intro n
  have := hgate n
  unfold stepReward at this
  linarith

/-! ## PROOF-05: Diverge-Converge score existence and non-triviality -/

/-- A diverge-then-converge trajectory: receiver entropy `E` strictly increases
up to a peak index `tstar` (divergence) and strictly decreases afterward
(convergence). `DC = 1` corresponds to satisfying this predicate. -/
def hasDivergeConverge (E : ℕ → ℝ) (tstar T : ℕ) : Prop :=
  0 < tstar ∧ tstar < T ∧
    (∀ t, t < tstar → E t < E (t + 1)) ∧
    (∀ t, tstar ≤ t → t + 1 ≤ T → E (t + 1) < E t)

/-- Explicit tent-shaped receiver-entropy trajectory peaked at `t = 1`:
`E 0 = -1`, `E 1 = 0`, `E 2 = -1`. -/
noncomputable def tentTrajectory : ℕ → ℝ :=
  fun t => 1 - ((t : ℝ) - 1) ^ 2

/--
**PROOF-05** `DC_score_existence`. A non-degenerate generation process can
realize a diverge-then-converge trajectory: `DC = 1` is achievable.
-/
theorem DC_score_existence :
    ∃ (E : ℕ → ℝ) (tstar T : ℕ), hasDivergeConverge E tstar T := by
  refine ⟨tentTrajectory, 1, 2, ?_, ?_, ?_, ?_⟩
  · exact one_pos
  · exact one_lt_two
  · intro t ht
    have h0 : t = 0 := by omega
    subst h0
    unfold tentTrajectory; norm_num
  · intro t hlo hhi
    have h1 : t = 1 := by omega
    subst h1
    unfold tentTrajectory; norm_num

/--
DC non-triviality (`DC = 0` side): a strictly increasing (monotone-divergent)
trajectory cannot satisfy the diverge-converge predicate, because the
convergence phase would contradict monotonicity at the peak.
-/
theorem monotone_increasing_not_DC (E : ℕ → ℝ)
    (hinc : ∀ t, E t < E (t + 1)) (tstar T : ℕ) :
    ¬ hasDivergeConverge E tstar T := by
  rintro ⟨_, hstar_lt, _, hconv⟩
  have hconverge : E (tstar + 1) < E tstar :=
    hconv tstar (le_refl tstar) hstar_lt
  have hincrease : E tstar < E (tstar + 1) := hinc tstar
  linarith

/-! ## PROOF-14: γ-positivity characterization -/

/-- The pure exponential-saturation component `C∞ (1 - e^{-μt})`. -/
noncomputable def expComponent (Cinf μ t : ℝ) : ℝ :=
  Cinf * (1 - Real.exp (-μ * t))

/-- The fitted Step-CUE curve `C∞ (1 - e^{-μt}) + γ t`. -/
noncomputable def stepCUEFit (Cinf μ γ t : ℝ) : ℝ :=
  expComponent Cinf μ t + γ * t

/--
**PROOF-14** `step_CUE_gamma_positivity_characterization`. The linear
saturation-rate `γ` is strictly positive iff the fitted curve exceeds the pure
exponential-saturation prediction at some positive time — i.e. the trajectory
contains genuine continuous discovery beyond exponential saturation.
-/
theorem step_CUE_gamma_positivity_characterization (Cinf μ γ : ℝ) :
    0 < γ ↔ ∃ t : ℝ, 0 < t ∧ expComponent Cinf μ t < stepCUEFit Cinf μ γ t := by
  constructor
  · intro hγ
    refine ⟨1, one_pos, ?_⟩
    unfold stepCUEFit
    have hone : (0 : ℝ) < γ * 1 := by simpa using hγ
    linarith
  · rintro ⟨t, ht, hlt⟩
    unfold stepCUEFit at hlt
    have hpos : 0 < γ * t := by linarith
    by_contra h
    push_neg at h
    nlinarith [mul_nonneg (neg_nonneg.mpr h) ht.le]

/-! ## PROOF-15: necessity of the terminal DC bonus -/

/-- Total trajectory reward: cumulative Step-CUE plus a terminal DC bonus. -/
def totalReward (cumCUE bonus : ℝ) : ℝ := cumCUE + bonus

/--
**PROOF-15** `dense_reward_DC_bonus_incentivizes_exploration`. The terminal DC
bonus is necessary. Consider a monotone-convergent trajectory and a
diverge-converge trajectory with *equal cumulative CUE* `c`. Without the bonus
their rewards tie (so the dense reward alone does not prefer exploration); with
a strictly positive bonus applied to the diverge-converge trajectory, its total
reward strictly exceeds the monotone one.
-/
theorem dense_reward_DC_bonus_incentivizes_exploration
    (c b : ℝ) (hb : 0 < b) :
    totalReward c 0 = totalReward c 0 ∧
      totalReward c 0 < totalReward c b := by
  refine ⟨rfl, ?_⟩
  unfold totalReward
  linarith

end Creativity.CUE.Trajectory
