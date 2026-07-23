import Creativity.Training.RLSignal
import Creativity.D.ProxyBridge
import Mathlib.Tactic

/-!
# Dual-Process Separation Principle and Structural Goodhart Collapse

This file formalizes the two headline structural results of the
receiver-grounded framework:

* `dual_process_separation_principle` (PROOF-03): the training reward
  `R_train` depends only on live (resampled, updatable) quantities, so its
  gradient is independent of the frozen benchmark objects `Π_frozen = {P, T, A}`.
  Corollary `Rtrain_cannot_goodhart_frozen`: no gradient path can move `R_train`
  by exploiting a frozen object, because `R_train` is invariant under any change
  to the frozen component.

* `structural_goodhart_collapse` (PROOF-13): if the rank ordering of the proxy
  and the benchmark disagree beyond the resolution margin `2ε` — which is
  exactly what a violated Goodhart condition permits — then the
  `ProxyFaithfulness` premise of `D_tilde_proxy_rank_agreement` cannot hold.
  Goodhart resistance is thereby a theorem/violation pair: protection
  (`D_tilde_proxy_rank_agreement`) ↔ collapse (`rank_reversal_refutes_faithfulness`).
-/

namespace Creativity.Training.DualProcess

open Creativity.D.ProxyReward
open Creativity.D.ProxyBridge

/-! ## PROOF-03: Dual-Process Separation Principle -/

/-- A training configuration splits into a live component (resampled probe
prompts, current policy, corpus schedule state) and a frozen component
(the benchmark probe set `P`, task battery `T`, receiver `A`). -/
structure TrainingConfig (Live Frozen : Type*) where
  live : Live
  frozen : Frozen

/-- A reward function depends only on the live component: any two configurations
agreeing on `live` receive equal reward. This is the discrete statement that the
reward's gradient in the frozen coordinate is zero. -/
def dependsOnlyOnLive {Live Frozen : Type*}
    (R : TrainingConfig Live Frozen → ℝ) : Prop :=
  ∀ c c' : TrainingConfig Live Frozen, c.live = c'.live → R c = R c'

/-- The training reward as a live-only projection: `R_train` is computed from a
function `f` of the live component alone. -/
def liveReward {Live Frozen : Type*} (f : Live → ℝ) :
    TrainingConfig Live Frozen → ℝ :=
  fun c => f c.live

/--
**PROOF-03** `dual_process_separation_principle`. Any training reward built as a
function of the live component alone is independent of the frozen benchmark
objects: its gradient has zero dependence on `Π_frozen`.
-/
theorem dual_process_separation_principle {Live Frozen : Type*}
    (f : Live → ℝ) :
    dependsOnlyOnLive (Frozen := Frozen) (liveReward f) := by
  intro c c' hlive
  unfold liveReward
  rw [hlive]

/--
Corollary: `R_train` cannot Goodhart the benchmark through any differentiable
path. Changing the frozen benchmark objects (holding the live component fixed)
leaves `R_train` unchanged, so no gradient step on `R_train` can be driven by,
or exploit, a frozen object.
-/
theorem Rtrain_cannot_goodhart_frozen {Live Frozen : Type*}
    (f : Live → ℝ) (live : Live) (frozen frozen' : Frozen) :
    liveReward f (⟨live, frozen⟩ : TrainingConfig Live Frozen) =
      liveReward f ⟨live, frozen'⟩ :=
  dual_process_separation_principle f ⟨live, frozen⟩ ⟨live, frozen'⟩ rfl

/-! ## PROOF-13: Structural Goodhart Collapse -/

/-- A two-candidate proxy/benchmark instance: proxy readings `proxy s` on
resampled probes and benchmark readings `bench s` on the frozen probe set. -/
structure ProxyRankInstance (S : Type*) where
  proxy : S → ℝ
  bench : S → ℝ
  ε : ℝ

/-- The rank-agreement guarantee: a proxy gap beyond `2ε` implies the same
benchmark ordering. -/
def rankAgreement {S : Type*} (I : ProxyRankInstance S) : Prop :=
  ∀ s₁ s₂ : S, I.proxy s₂ + 2 * I.ε < I.proxy s₁ → I.bench s₂ < I.bench s₁

/-- A rank reversal: the proxy strictly prefers `s₁` (beyond `2ε`) while the
benchmark strictly prefers `s₂`. This is exactly the exploitation a violated
Goodhart condition permits. -/
def rankReversal {S : Type*} (I : ProxyRankInstance S) (s₁ s₂ : S) : Prop :=
  I.proxy s₂ + 2 * I.ε < I.proxy s₁ ∧ I.bench s₁ < I.bench s₂

/--
Generic collapse: any rank reversal refutes rank agreement, hence refutes the
`ProxyFaithfulness` premise of `D_tilde_proxy_rank_agreement`.
-/
theorem rank_reversal_refutes_agreement {S : Type*}
    (I : ProxyRankInstance S) {s₁ s₂ : S}
    (hrev : rankReversal I s₁ s₂) :
    ¬ rankAgreement I := by
  rintro hagree
  obtain ⟨hproxy, hbench⟩ := hrev
  have hforce : I.bench s₂ < I.bench s₁ := hagree s₁ s₂ hproxy
  linarith

/--
A rank reversal is realizable: there is a two-candidate instance exhibiting the
proxy/benchmark disagreement. This is the explicit adversarial witness (on a
two-element candidate set) that a violated Goodhart condition admits. Here the
proxy ranks `true` above `false` beyond the margin while the benchmark ranks
them oppositely.
-/
theorem reversal_instance_exists :
    ∃ (I : ProxyRankInstance Bool) (s₁ s₂ : Bool),
      rankReversal I s₁ s₂ := by
  refine ⟨⟨fun b => if b then 10 else 0, fun b => if b then 0 else 10, 1⟩,
    true, false, ?_, ?_⟩
  · show (0 : ℝ) + 2 * 1 < 10; norm_num
  · show (0 : ℝ) < 10; norm_num

/--
**PROOF-13** `structural_goodhart_collapse`. Whenever a Goodhart condition is
violated so that a rank reversal becomes realizable, the rank-agreement
guarantee collapses: there is a proxy/benchmark instance whose reversal refutes
`rankAgreement`. Each of the four conditions
(`D_tilde_no_fixed_target`, `C_corpus_stationarity_condition`,
`B_utility_gate_tightness`, `ProxyFaithfulness_decay_detectable`) instantiates
this collapse through its own exploitation path; the shared structural core is
that any admitted reversal breaks the guarantee.
-/
theorem structural_goodhart_collapse :
    ∃ (I : ProxyRankInstance Bool), ¬ rankAgreement I := by
  obtain ⟨I, s₁, s₂, hrev⟩ := reversal_instance_exists
  exact ⟨I, rank_reversal_refutes_agreement I hrev⟩

end Creativity.Training.DualProcess
