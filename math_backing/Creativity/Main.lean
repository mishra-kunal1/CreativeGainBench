import Creativity.B.Theorems
import Creativity.B.Baseline
import Creativity.B.Estimator
import Creativity.B.DPO
import Creativity.B.CoherenceBound
import Creativity.CUE.Receiver
import Creativity.CUE.ExpansionLabel
import Creativity.CUE.Trajectory
import Creativity.CUE.MASBridge
import Creativity.CUE.BenchmarkScore
import Creativity.D.ProxyReward
import Creativity.D.DPO_tilde
import Creativity.D.ProxyBridge
import Creativity.Probability.MaxEntropy
import Creativity.Training.DPOBridge
import Creativity.Training.RLSignal
import Creativity.Training.Commensurability
import Creativity.Training.Composability
import Creativity.Training.DualProcess
import Creativity.Training.TwoGRPO
import Creativity.D.Theorems
import Creativity.MAS.TwoAgent
import Creativity.MAS.KAgentModel
import Creativity.MAS.OInformation
import Creativity.MAS.OInfoBridge
import Creativity.Benchmark.EstimatorBounds
import Creativity.Benchmark.Spec

/-!
# Formal Creativity Framework — Bridged Validation

## Layer 0: Receiver-Grounded Foundation (CUE)

- `cue` / `CUEModel`: Creative Update Efficiency — receiver decision-quality
  gain (Brier-delta) per output bit.
- `CUE_voi_equivalence` (PROOF-01): CUE lower-bounds normalized Shannon VOI.
- `shannon_upper_bound_normalization` (PROOF-02): CUE ≤ log|Z| / |y|_bits.
- `CUE_primary_gate`: nonpositive CUE disqualifies downstream metrics.
- `receiver_expansion_clarification_separation` (PROOF-06): expansion label
  is not monotone in CUE (diagnostic; Benchmark Score uses it as a bonus).
- Trajectory / DC / G_k bridge theorems (PROOF-04, 05, 07, 14, 15).

## Layer 0d: Canonical Benchmark Score \(R_{\mathrm{creativity}}\)

\[
  R_{\mathrm{creativity}}(q,y,s)
  = 𝟙[\mathrm{CUE}>0]\cdot𝟙[R_D>δ_D]
    ·(\mathrm{CUE}\cdot(1+α·R_B^{→A}) + λ_G·G_k)
\]

Lean: `Rcreativity` / `benchmarkScore`. No \(R_C\) term. Both gates are
multiplicative safety floors: either failure zeroes the *entire* score,
including \(λ_G G_k\).

- `cueGate` / `dGate`: CUE and \(R_D\) enter as binary gates on the whole score.
- `Rcreativity_cue_gate_closed` / `Rcreativity_d_gate_closed`: score = 0.
- `Rcreativity_nonneg` / `Rcreativity_bounded` / `Rcreativity_monotone_in_CUE`.
- `Rcreativity_pos_iff_CUE_pos`: positivity matches CUE when D-gate open and
  `λ_G·G_k = 0`.
- `unit_conversion_sign_and_order_invariant`: bits/nats reconciliation; `α`
  absorbs residual scale.
- Bridge proofs replacing former \(R_C\) dependencies:
  `D3_rewardD_bound_standalone`, `lambda_D_normalization_standalone`,
  `paretoFrontier_nonempty_BD`.
- Optional `Rcreativity_shifted` (threshold `f`) is calibration-only (KG-5).

## Layer 0b: Dual-Process Separation and Structural Goodhart

- `dual_process_separation_principle` (PROOF-03).
- `structural_goodhart_collapse` (PROOF-13).
- `twoGRPO_advantage_quantization_unbiased` (PROOF-16).
- Training signal: `trainReward = R_B + λ̃_D·R̃_D` (no \(R_C\)).
- `Goodhart_resistance_summary`: D̃ no-fixed-target, B gate tightness,
  proxy-decay detectability.

## Layer 1: Abstract Objective Properties (B / D)

- B1–B3, D1, D2 (diagnostic for the binary D-gate), D3 standalone, D4.

## Layer 2–5: Baselines, estimators, DPO (B / D̃), MAS

As before, without C-channel theorems.
-/

#check Creativity.B.Theorems.B1_dominance
#check Creativity.B.Theorems.B2_no_collapse
#check Creativity.B.Theorems.B3_expected_exceeds_dominated

#check Creativity.D.Theorems.D1_utility_gate
#check Creativity.D.Theorems.D2_dominance
#check Creativity.D.Theorems.D3_rewardD_bound_standalone
#check Creativity.D.Theorems.D4_zero_for_corpus_copy
#check Creativity.D.Theorems.D4_zero_reward_for_corpus_copy

#check Creativity.B.Baseline.BBase_indistinguishable
#check Creativity.B.Baseline.BContrast_distinguishes

#check Creativity.B.Estimator.B_rank_preserved

#check Creativity.B.DPO.DPO_B_correct_direction
#check Creativity.Training.DPO.dpo_logit_direction
#check Creativity.Training.DPO.dpo_full_gradient_out_of_scope
#check Creativity.Training.DPO.dpo_gradient_increases_logodds
#check Creativity.Training.DPO.softmax_monotone_in_logit_diff
#check Creativity.Training.DPO.dpo_step_increases_expected_RB
#check Creativity.Training.DPO.DPO_B_gradient_correct

#check Creativity.B.Theorems.B3b_jensen_anti_collapse
#check Creativity.B.Theorems.entropy_mixture_concavity
#check Creativity.B.Theorems.mixtureKernel_apply_toReal

#check Creativity.Probability.maxEntropy_finite_alphabet
#check Creativity.Probability.klDivergence_uniform_eq
#check Creativity.Probability.sum_toReal_eq_one
#check Creativity.Probability.shannonEntropy_eq_sum_negMulLog

#check Creativity.CUE.CUEModel
#check Creativity.CUE.cue
#check Creativity.CUE.cue_nonneg
#check Creativity.CUE.CUE_voi_equivalence
#check Creativity.CUE.CUE_voi_equality
#check Creativity.CUE.shannon_upper_bound_normalization
#check Creativity.CUE.CUE_primary_gate
#check Creativity.CUE.LabeledOutput
#check Creativity.CUE.receiver_expansion_clarification_separation
#check Creativity.CUE.Trajectory.CUE_trajectory_monotone_nondecreasing
#check Creativity.CUE.Trajectory.DC_score_existence
#check Creativity.CUE.Trajectory.monotone_increasing_not_DC
#check Creativity.CUE.Trajectory.step_CUE_gamma_positivity_characterization
#check Creativity.CUE.Trajectory.dense_reward_DC_bonus_incentivizes_exploration
#check Creativity.CUE.MASBridge.ReceiverCalibratedMAS
#check Creativity.CUE.MASBridge.Gk_implies_CUE_improvement
#check Creativity.CUE.MASBridge.Gk_implies_CUE_improvement_sup

#check Creativity.CUE.CUE_shifted
#check Creativity.CUE.CUEThresholdModel
#check Creativity.CUE.CUE_shifted_bound
#check Creativity.CUE.CUE_shifted_sign_iff
#check Creativity.CUE.receiverExpansionNorm
#check Creativity.CUE.receiver_expansion_bounded
#check Creativity.CUE.Rcreativity_sign_matches_CUE_shifted
#check Creativity.CUE.MASBridge.Gk_gated_by_CUE_shifted

#check Creativity.CUE.cueGate
#check Creativity.CUE.dGate
#check Creativity.CUE.Rcreativity
#check Creativity.CUE.benchmarkScore
#check Creativity.CUE.Rcreativity_cue_gate_closed
#check Creativity.CUE.Rcreativity_d_gate_closed
#check Creativity.CUE.benchmarkScore_gate_closed
#check Creativity.CUE.Rcreativity_nonneg
#check Creativity.CUE.benchmarkScore_nonneg
#check Creativity.CUE.Rcreativity_bounded
#check Creativity.CUE.benchmarkScore_bounded
#check Creativity.CUE.Rcreativity_monotone_in_CUE
#check Creativity.CUE.benchmarkScore_monotone_in_CUE
#check Creativity.CUE.Rcreativity_pos_iff_CUE_pos
#check Creativity.CUE.unit_conversion_sign_and_order_invariant
#check Creativity.Training.Commensurability.lambda_D_normalization_standalone
#check Creativity.Training.Composability.paretoFrontier_nonempty_BD

#check Creativity.Training.DualProcess.dual_process_separation_principle
#check Creativity.Training.DualProcess.Rtrain_cannot_goodhart_frozen
#check Creativity.Training.DualProcess.rank_reversal_refutes_agreement
#check Creativity.Training.DualProcess.reversal_instance_exists
#check Creativity.Training.DualProcess.structural_goodhart_collapse

#check Creativity.Training.TwoGRPO.groupScale_eq
#check Creativity.Training.TwoGRPO.advantage_antisymm
#check Creativity.Training.TwoGRPO.twoGRPO_advantage_quantization_unbiased

#check Creativity.MAS.MAS_irreducibility
#check Creativity.MAS.MAS_benchmark_detects_irreducibility

#check Creativity.Training.Commensurability.lambdaD
#check Creativity.Training.Commensurability.normalizedRewardB_eq_one
#check Creativity.Training.Commensurability.normalizedRewardD_eq_one
#check Creativity.Training.Commensurability.rewardB_le_maxEntropy

#check Creativity.Training.benchmarkReward
#check Creativity.Training.benchmarkReward_nonneg
#check Creativity.Training.advantage_sign_correct
#check Creativity.Training.B_winner_advantage_pos

#check Creativity.D.ProxyReward.DTildeModel
#check Creativity.D.ProxyReward.rewardDTilde
#check Creativity.D.ProxyReward.D_tilde_utility_gate
#check Creativity.D.ProxyReward.D_tilde_nonneg
#check Creativity.D.ProxyReward.D_tilde_zero_for_identical
#check Creativity.D.ProxyReward.D_tilde_no_fixed_target
#check Creativity.D.ProxyReward.D_tilde_reward_dominance
#check Creativity.D.ProxyReward.D_tilde_dominance

#check Creativity.D.DPOTilde.DPO_D_tilde_gradient_correct

#check Creativity.D.ProxyBridge.ProxyFaithfulness
#check Creativity.D.ProxyBridge.D_tilde_proxy_rank_agreement
#check Creativity.D.ProxyBridge.ProxyFaithfulness_decay_detectable
#check Creativity.D.ProxyBridge.proxyMonitor_decayDetectable

#check Creativity.B.CoherenceBound.B_utility_gate_tightness
#check Creativity.B.CoherenceBound.B_incoherence_reward_below_threshold
#check Creativity.B.CoherenceBound.B_coherent_dominates_incoherent

#check Creativity.Training.trainReward
#check Creativity.Training.trainReward_nonneg
#check Creativity.Training.trainReward_advantage_sign_correct
#check Creativity.Training.trainReward_neq_benchmarkReward
#check Creativity.Training.Goodhart_resistance_summary

#check Creativity.MAS.kAgent_irreducibility
#check Creativity.MAS.kAgent_benchmark_detects
#check Creativity.MAS.CommGraph
#check Creativity.MAS.isProductKernel
#check Creativity.MAS.nonProduct_base_case
#check Creativity.MAS.connectedMAS_nonProductKernel
#check Creativity.MAS.gain_zero_iff_joint_eq_max
#check Creativity.MAS.productKernel_additivity_gain_pos
#check Creativity.MAS.gain_biconditional

#check Creativity.MAS.OInfo.oInformation3_negative_of_pair_sum_lt
#check Creativity.MAS.OInfo.equalMarginals_gain_implies_synergy
#check Creativity.MAS.OInfo.negativeGain_implies_redundancy

#check Creativity.MAS.OInfoBridge.jointPMF_apply
#check Creativity.MAS.OInfoBridge.shannonEntropy_jointPMF
#check Creativity.MAS.OInfoBridge.productKernel_joint_entropy_eq_sum
#check Creativity.MAS.OInfoBridge.productKernel_implies_redundancy
#check Creativity.MAS.OInfoBridge.productKernel_gain_is_not_synergy

#check Creativity.Training.Composability.Bmax_not_jointly_dominated_BD
#check Creativity.Training.Composability.BD_tension_example

#check Creativity.Benchmark.plugIn_bias_axiom
#check Creativity.Benchmark.entropy_ranking_sample_requirement

#check Creativity.Benchmark.predictedS2DominatesS1
#check Creativity.Benchmark.predictedGroupthinkSignature
#check Creativity.Benchmark.benchmark_can_detect_S2_dominance
