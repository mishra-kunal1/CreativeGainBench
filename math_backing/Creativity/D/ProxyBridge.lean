import Creativity.D.ProxyReward
import Creativity.D.Theorems
import Mathlib.Tactic

/-!
# Proxy faithfulness: bridging R̃_D (training) to R_D (benchmark)

R̃_D is trained on resampled probe prompts; R_D is measured on the frozen
hold-out probe set `P`. This file proves the conditional bridge between them:

- `D_tilde_proxy_rank_agreement`: under a `ProxyFaithfulness` witness — the KL
  shift on resampled `q_probe` and R_D on frozen `P` are monotonically related
  up to error bound δ — a proxy gap greater than `2ε` forces the same rank
  ordering in the benchmark quantity. The proof structure mirrors
  `B_rank_preserved` (Thm 23) and `C_rank_preserved` (Thm 24).

- `ProxyFaithfulness_decay_detectable`: the **formal Goodhart early-warning
  condition**. If at a training checkpoint the empirical correlation
  ρ(R̃_D, R_D) falls below the threshold `ρ_min`, then no `ProxyFaithfulness`
  witness is available at that checkpoint, the rank-agreement guarantee of
  `D_tilde_proxy_rank_agreement` no longer holds, and training must halt or
  the proxy must be recalibrated. This converts the proxy-faithfulness
  assumption into a monitorable training health invariant rather than a
  one-time post-hoc validation.
-/

namespace Creativity.D.ProxyBridge

open Creativity.D.ProxyReward
open Creativity.D.ProbeCompressor

variable {Q S Z α : Type*} [Fintype Z] [ProbeCompressor α]

/--
Proxy faithfulness for R̃_D: after monotone calibration, the proxy reward on
resampled probe prompts tracks the benchmark reward R_D on the frozen probe
set `P` (`probes`) uniformly up to error bound `epsilon` (the δ of the prose
statement). This is an *empirical* assumption — unlike the purely algebraic
`BEstimator`/`CEstimator` bounds, it must be estimated by the benchmark
correlation study before training begins.
-/
structure ProxyFaithfulness (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : S → List α) (corpus : List α) (probes : List (List α))
    (u τ : ℝ) where
  epsilon : ℝ
  hbound : ∀ s : S,
    |rewardDTilde m qprobe s u τ -
      rewardD (artifact s) corpus probes u τ| ≤ epsilon

/--
**Proxy rank agreement** (mirrors `B_rank_preserved` (Thm 23) and
`C_rank_preserved` (Thm 24)): under `ProxyFaithfulness`, if
`R̃_D(s₁) - R̃_D(s₂) > 2ε` then `R_D(s₁, H, P) > R_D(s₂, H, P)`.
-/
theorem D_tilde_proxy_rank_agreement
    (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : S → List α) (corpus : List α) (probes : List (List α))
    (u τ : ℝ)
    (pf : ProxyFaithfulness m qprobe artifact corpus probes u τ)
    (s₁ s₂ : S)
    (hgap : rewardDTilde m qprobe s₂ u τ + 2 * pf.epsilon <
      rewardDTilde m qprobe s₁ u τ) :
    rewardD (artifact s₂) corpus probes u τ <
      rewardD (artifact s₁) corpus probes u τ := by
  have h1 := abs_le.mp (pf.hbound s₁)
  have h2 := abs_le.mp (pf.hbound s₂)
  linarith

/--
A proxy-faithfulness monitor over training checkpoints:

* `rho t` — empirical correlation ρ(R̃_D, R_D) measured on a small held-aside
  validation set at checkpoint `t`;
* `rho_min` — the correlation threshold below which faithfulness is
  considered broken. **Calibration requirement**: `rho_min` must be estimated
  by the benchmark correlation study *before training begins*;
* `faithfulAt t` — a `ProxyFaithfulness` witness at checkpoint `t` exists;
* `faithful_implies_rho` — the calibration link: whenever faithfulness
  actually holds, the measured correlation stays at or above `rho_min`.
-/
structure ProxyMonitor (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : S → List α) (corpus : List α) (probes : List (List α))
    (u τ : ℝ) where
  rho : ℕ → ℝ
  rho_min : ℝ
  rho_min_precalibrated : Prop
  faithfulAt : ℕ → Prop
  faithful_hasWitness : ∀ t : ℕ, faithfulAt t →
    Nonempty (ProxyFaithfulness m qprobe artifact corpus probes u τ)
  faithful_implies_rho : ∀ t : ℕ, faithfulAt t → rho_min ≤ rho t

/--
**Goodhart early-warning condition**: if at training checkpoint `t` the
empirical correlation ρ(R̃_D, R_D) falls below `ρ_min`, then no proxy
faithfulness holds at `t` — so the hypothesis of
`D_tilde_proxy_rank_agreement` is unavailable and its rank-agreement
guarantee no longer applies. Training must halt or the proxy must be
recalibrated before any further comparison against benchmark metrics.
-/
theorem ProxyFaithfulness_decay_detectable
    (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : S → List α) (corpus : List α) (probes : List (List α))
    (u τ : ℝ)
    (M : ProxyMonitor m qprobe artifact corpus probes u τ)
    (t : ℕ) (hdecay : M.rho t < M.rho_min) :
    ¬ M.faithfulAt t := by
  intro hfaithful
  exact absurd (M.faithful_implies_rho t hfaithful) (not_le.mpr hdecay)

/--
Decay detectability as a checkpoint-indexed invariant: at every checkpoint,
either the measured correlation is at least `ρ_min`, or faithfulness has
demonstrably failed and training must halt. This is the form consumed by
`Goodhart_resistance_summary`.
-/
def DecayDetectable (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : S → List α) (corpus : List α) (probes : List (List α))
    (u τ : ℝ)
    (M : ProxyMonitor m qprobe artifact corpus probes u τ) : Prop :=
  ∀ t : ℕ, M.rho t < M.rho_min → ¬ M.faithfulAt t

/-- Every monitor satisfies the decay-detectability invariant. -/
theorem proxyMonitor_decayDetectable
    (m : DTildeModel Q S Z) (qprobe : Q)
    (artifact : S → List α) (corpus : List α) (probes : List (List α))
    (u τ : ℝ)
    (M : ProxyMonitor m qprobe artifact corpus probes u τ) :
    DecayDetectable m qprobe artifact corpus probes u τ M :=
  fun t hdecay =>
    ProxyFaithfulness_decay_detectable m qprobe artifact corpus probes u τ
      M t hdecay

end Creativity.D.ProxyBridge
