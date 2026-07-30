import Creativity.B.Model
import Creativity.B.CoherenceBound
import Creativity.D.Deformation
import Creativity.D.ProxyReward
import Creativity.D.ProxyBridge
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

namespace Creativity.Training

open Creativity.B
open Creativity.B.CoherenceBound
open Creativity.D
open Creativity.D.ProxyReward
open Creativity.D.ProxyBridge
open ProbeCompressor

/-!
Training and additive B–D diagnostics for the \(R_C\)-free framework.

The reported Benchmark Score is `Creativity.CUE.Rcreativity`
(`𝟙[CUE>0]·𝟙[R_D>δ_D]·(CUE·(1+α·R_B^{→A})+λ_G·G_k)`). This file supplies:

* `benchmarkReward` — auxiliary additive `R_B + λ_D·R_D` diagnostic (never
  the reported score);
* `trainReward` — live training signal `R_B + λ̃_D·R̃_D` (no \(R_C\) channel);
* group-relative advantage lemmas and the Goodhart-resistance summary.
-/

/-- Auxiliary additive B–D diagnostic `R_B(q,y) + λ_D · R_D(q,s)`.

Not the reported Benchmark Score (`Rcreativity`). Retained for B/D
rank-preservation correspondence. -/
noncomputable def benchmarkReward
    {Q Y Z α : Type*} [Fintype Z] [ProbeCompressor α]
    (m : BModel Q Y Z) (lambdaD : ℝ)
    (q : Q) (y : Y) (s corpus : List α) (probes : List (List α)) (u τ : ℝ) : ℝ :=
  rewardB m q y + lambdaD * rewardD s corpus probes u τ

theorem benchmarkReward_nonneg
    {Q Y Z α : Type*} [Fintype Z] [ProbeCompressor α]
    (m : BModel Q Y Z) (lambdaD : ℝ)
    (q : Q) (y : Y) (s corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (hB : 0 ≤ rewardB m q y)
    (hD : 0 ≤ rewardD s corpus probes u τ)
    (hlambdaD : 0 ≤ lambdaD) :
    0 ≤ benchmarkReward m lambdaD q y s corpus probes u τ := by
  unfold benchmarkReward
  exact add_nonneg hB (mul_nonneg hlambdaD hD)

noncomputable def groupMean {G : ℕ} (rewards : Fin G → ℝ) : ℝ :=
  (∑ j : Fin G, rewards j) / (G : ℝ)

noncomputable def groupDenom {G : ℕ} (rewards : Fin G → ℝ) (ε : ℝ) : ℝ :=
  Real.sqrt ((∑ j : Fin G, (rewards j - groupMean rewards)^2) / (G : ℝ)) + ε

noncomputable def groupRelativeAdvantage {G : ℕ}
    (rewards : Fin G → ℝ) (i : Fin G) (ε : ℝ) : ℝ :=
  (rewards i - groupMean rewards) / groupDenom rewards ε

theorem groupDenom_pos {G : ℕ} (rewards : Fin G → ℝ) (ε : ℝ) (hε : 0 < ε) :
    0 < groupDenom rewards ε := by
  unfold groupDenom
  exact add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) hε

theorem advantage_sign_correct {G : ℕ} (rewards : Fin G → ℝ)
    (i j : Fin G) (ε : ℝ)
    (hj : rewards j < groupMean rewards)
    (hi : groupMean rewards < rewards i)
    (hε : 0 < ε) :
    groupRelativeAdvantage rewards j ε < 0 ∧
      0 < groupRelativeAdvantage rewards i ε := by
  have hden : 0 < groupDenom rewards ε := groupDenom_pos rewards ε hε
  constructor
  · unfold groupRelativeAdvantage
    exact div_neg_of_neg_of_pos (sub_neg.mpr hj) hden
  · unfold groupRelativeAdvantage
    exact div_pos (sub_pos.mpr hi) hden

theorem B_winner_advantage_pos {G : ℕ} (rewards : Fin G → ℝ)
    (iw : Fin G) (ε : ℝ)
    (_h_winner : ∀ j ≠ iw, rewards j < rewards iw)
    (h_mean : groupMean rewards < rewards iw)
    (hε : 0 < ε) :
    0 < groupRelativeAdvantage rewards iw ε := by
  have hden : 0 < groupDenom rewards ε := groupDenom_pos rewards ε hε
  unfold groupRelativeAdvantage
  exact div_pos (sub_pos.mpr h_mean) hden

/-!
## Training reward (no \(R_C\))

  R_train(q,y,s) := R_B(q,y) + λ̃_D · R̃_D(q,s)

`R_D` is frozen-benchmark only; training uses the proxy `R̃_D` on resampled
probe prompts. The reported measurement remains `Rcreativity`, not this
additive diagnostic.
-/

/-- Benchmark-aligned training reward: `R_B + λ̃_D · R̃_D` (no \(R_C\)). -/
noncomputable def trainReward
    {Q Y Z Qp S Zp : Type*} [Fintype Z] [Fintype Zp]
    (m : BModel Q Y Z) (dm : DTildeModel Qp S Zp)
    (lambdaDTilde : ℝ)
    (q : Q) (y : Y) (s : S) (qprobe : Qp) (u τ : ℝ) : ℝ :=
  rewardB m q y + lambdaDTilde * rewardDTilde dm qprobe s u τ

theorem trainReward_nonneg
    {Q Y Z Qp S Zp : Type*} [Fintype Z] [Fintype Zp]
    (m : BModel Q Y Z) (dm : DTildeModel Qp S Zp)
    (lambdaDTilde : ℝ)
    (q : Q) (y : Y) (s : S) (qprobe : Qp) (u τ : ℝ)
    (hB : 0 ≤ rewardB m q y)
    (hlambdaDTilde : 0 ≤ lambdaDTilde) :
    0 ≤ trainReward m dm lambdaDTilde q y s qprobe u τ := by
  unfold trainReward
  exact add_nonneg hB
    (mul_nonneg hlambdaDTilde (D_tilde_nonneg dm qprobe s u τ))

noncomputable def trainRewards
    {Q Y Z Qp S Zp : Type*} [Fintype Z] [Fintype Zp] {G : ℕ}
    (m : BModel Q Y Z) (dm : DTildeModel Qp S Zp)
    (lambdaDTilde : ℝ)
    (q : Q) (outputs : Fin G → Y) (artifacts : Fin G → S)
    (qprobe : Qp) (utilities : Fin G → ℝ) (τ : ℝ) :
    Fin G → ℝ :=
  fun k =>
    trainReward m dm lambdaDTilde q (outputs k) (artifacts k)
      qprobe (utilities k) τ

theorem trainReward_advantage_sign_correct
    {Q Y Z Qp S Zp : Type*} [Fintype Z] [Fintype Zp] {G : ℕ}
    (m : BModel Q Y Z) (dm : DTildeModel Qp S Zp)
    (lambdaDTilde : ℝ)
    (q : Q) (outputs : Fin G → Y) (artifacts : Fin G → S)
    (qprobe : Qp) (utilities : Fin G → ℝ) (τ : ℝ)
    (i j : Fin G) (ε : ℝ)
    (hj : trainRewards m dm lambdaDTilde q outputs artifacts
        qprobe utilities τ j <
      groupMean (trainRewards m dm lambdaDTilde q outputs artifacts
        qprobe utilities τ))
    (hi : groupMean (trainRewards m dm lambdaDTilde q outputs artifacts
        qprobe utilities τ) <
      trainRewards m dm lambdaDTilde q outputs artifacts
        qprobe utilities τ i)
    (hε : 0 < ε) :
    groupRelativeAdvantage (trainRewards m dm lambdaDTilde q outputs
      artifacts qprobe utilities τ) j ε < 0 ∧
    0 < groupRelativeAdvantage (trainRewards m dm lambdaDTilde q
      outputs artifacts qprobe utilities τ) i ε :=
  advantage_sign_correct _ i j ε hj hi hε

/-- Train/benchmark split: `R_train` (live proxy) ≠ `benchmarkReward` (frozen D)
whenever the weighted D̃ and D terms differ. -/
theorem trainReward_neq_benchmarkReward
    {Q Y Z Qp Zp α : Type*} [Fintype Z] [Fintype Zp] [ProbeCompressor α]
    (m : BModel Q Y Z) (dm : DTildeModel Qp (List α) Zp)
    (lambdaD lambdaDTilde : ℝ)
    (q : Q) (y : Y) (s corpus : List α) (probes : List (List α))
    (qprobe : Qp) (u τ : ℝ)
    (hD : lambdaDTilde * rewardDTilde dm qprobe s u τ ≠
      lambdaD * rewardD s corpus probes u τ) :
    trainReward m dm lambdaDTilde q y s qprobe u τ ≠
      benchmarkReward m lambdaD q y s corpus probes u τ := by
  unfold trainReward benchmarkReward
  intro heq
  apply hD
  linarith

/--
**Goodhart resistance summary** (three certificates; no \(R_C\) channel):

1. `D_tilde_no_fixed_target` — probe prompts resampled every batch;
2. `B_utility_gate_tightness` — τ above incoherent high-entropy baselines;
3. `ProxyFaithfulness_decay_detectable` — proxy/benchmark rank decay halt.
-/
theorem Goodhart_resistance_summary
    {Q Y Z Qp S Zp α : Type*}
    [Fintype Z] [Fintype Zp] [ProbeCompressor α]
    (Pd : DTildeTrainingProtocol Qp)
    (m : BModel Q Y Z) (q : Q) (cal : CoherenceBound.TauCalibration m q)
    (dm : DTildeModel Qp S Zp) (qprobe : Qp) (artifact : S → List α)
    (corpus : List α) (probes : List (List α)) (u τ : ℝ)
    (M : ProxyMonitor dm qprobe artifact corpus probes u τ) :
    NoFixedTarget Pd.probeBatch ∧
    (∀ base : CoherenceBound.IncoherentBaseline m q,
      CoherenceBound.BUtilityGateTight m q base) ∧
    DecayDetectable dm qprobe artifact corpus probes u τ M :=
  ⟨D_tilde_no_fixed_target Pd,
    CoherenceBound.B_utility_gate_tightness m q cal,
    proxyMonitor_decayDetectable dm qprobe artifact corpus probes u τ M⟩

end Creativity.Training
