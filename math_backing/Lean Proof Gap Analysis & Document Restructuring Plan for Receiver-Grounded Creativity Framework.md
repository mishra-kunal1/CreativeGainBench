# Lean Proof Gap Analysis & Document Restructuring Plan for Receiver-Grounded Creativity Framework

## Executive Summary

The `bridged_validation.pdf` formalizes a substantial portion of the creativity evaluation framework, covering entropy monotonicity, compression novelty, probe-relative deformation, multi-agent irreducibility, DPO directionality, and Goodhart-resistance conditions across 73 named Lean theorems. However, the research proposal introduces five constructs that are either absent or only axiomatically asserted in the current Lean document: (1) Creative Update Efficiency (CUE) as a Shannon-theoretic Value-of-Information normalized quantity, (2) the formal Dual-Process Separation Principle tying the frozen-benchmark / live-proxy split to a Goodhart-resistance invariant, (3) trajectory-level Step-CUE curve fitting and Diverge-Converge Score, (4) the information-normalization commensurability chain connecting CUE to a Shannon upper bound, and (5) the multi-agent interaction gain \(G_k\) in its *proposal-specific* form that certifies not just joint entropy but decision-quality improvement. Several existing axioms (`entropy_mixture_concavity_axiom`, `gain_zero_iff_productKernel`, `productKernel_joint_entropy_eq_sum`, `plugIn_bias_axiom`) are flagged as needing full proofs from primitives. The section below maps every gap to a concrete Lean proof task and proof strategy, followed by a document restructuring task list.[^1]

***

## Part I: Literature Landscape — What Is Directly Relevant

### Existing Creativity Evaluation Work

Current creativity metrics for LLMs reduce a multi-dimensional property to a scalar. The four representative metrics (perplexity, Creativity Index, syntactic templates, LLM-as-a-Judge) disagree across domains and reach only ~40% self-consistency in high-creativity regimes. CreativityPrism consolidates eight tasks across divergent thinking, creative writing, and logical reasoning into quality/novelty/diversity dimensions, and finds that "high performance in one creative dimension or domain rarely generalizes to others; specifically, novelty metrics often show weak or negative correlations with other metrics". This fragmentation directly motivates the multi-dimensional CUE-gated approach.[^2][^3]

The "creativity coverage" framework from early 2026 defines creativity as a *boundary* rather than a scalar by identifying which regions of human creative space LLMs can reach — a geometric distributional framing. Critically, it still measures properties of the output's position in distributional space rather than its effect on a receiver, precisely the gap this proposal addresses. The Artificial Hivemind paper demonstrates that even aggressive temperature scaling fails to expand the semantic diversity within a single model family and that standard multi-agent interaction contracts rather than expands the solution space, empirically validating the need for interaction-outcome metrics rather than interaction-structure metrics.[^4][^5][^6]

### ComplLLM — The Closest Prior Work

ComplLLM (Guo et al., 2026) is the most directly relevant prior work. It post-trains a decision-assistant LLM using *complementary information as reward* — the output must complement existing agent decisions to score, validated on synthetic and real-world expert tasks. This confirms that receiver decision quality improvement is a valid training signal in verifiable decision-support domains. The proposal's CUE generalizes ComplLLM in three ways not present in that work: (a) it applies to open-ended creative tasks without binary payoff decompositions, (b) it normalizes by output length in bits connecting to a Shannon upper bound, and (c) it is embedded in a full dual-process Goodhart-resistance protocol separating frozen benchmark objects from live training proxies.[^7][^8]

### 2-GRPO

The "It Takes Two: Your GRPO Is Secretly DPO" paper provides the theoretical basis for 2-GRPO used in Phase 2. It proves that GRPO is a form of contrastive learning fundamentally connected to DPO, that the minimal two-rollout case (2-GRPO) achieves performance on par with 16-GRPO, and reduces training time by over 70% while using only 1/8 of rollouts. The advantages in 2-GRPO quantize to \(\pm \frac{1}{\sqrt{2}}\), which is unbiased and proportional to the probability of generating a correct answer — this quantization property is relevant to the dense trajectory reward \(r_{step}\) proofs needed.[^9][^10]

### Information-Theoretic Lean Formalization Context

Statistical Learning Theory in Lean 4 (2026) provides the most current reference point for empirical process theory in Lean 4. Lean 4 formalizations of Shannon entropy exist (e.g., the `shannon-entropy` Reservoir package), but formal proofs of Shannon concavity for *mixture kernels* (as needed for `entropy_mixture_concavity_axiom`) require the product-sum expansion with the logarithm product rule — an open item the current document notes explicitly. The FormalMATH benchmark uses Lean 4 for formal mathematical reasoning over 5,560 problems, confirming Lean 4 as the right target, but also revealing that even strong models achieve only ~16% success rate under practical sampling budgets — relevant to the proof-writing difficulty assessment.[^11][^12][^13][^1]

***

## Part II: Novelty Analysis — What Is New and Where It Can Be Strengthened

### Core Novel Contributions

| Contribution | Status in bridged_validation.pdf | Gap |
|---|---|---|
| CUE as Brier-delta/bits VOI metric | **Absent** — B measures entropy, not Brier-calibration shift | Full new formalization needed |
| Shannon upper bound normalization for CUE | **Absent** — bit-normalization not connected to any Shannon bound | Proof of \(\text{CUE} \le H(\text{optimal}) / |y|_{\text{bits}}\) needed |
| Dual-Process Separation Principle (formal) | **Partial** — `trainReward_neq_fullReward` exists but is definitional, not a structural theorem | Full DPS theorem with Goodhart-resistance implications needed |
| Trajectory Step-CUE curve \(\gamma\) | **Absent** — no trajectory-level formalism in document | Monotone non-decreasing CUE trajectory theorem needed |
| Diverge-Converge Score DC formal definition | **Absent** | DC existence / non-triviality proof needed |
| \(R_B^{\to A}\) as expansion vs. clarification label | **Partial** — entropy ordering exists, but the bidirectional label semantics not formalized | Formal separation theorem needed |
| Multi-Agent \(G_k\) tied to decision quality (not just entropy) | **Partial** — \(G_k\) certifies joint entropy irreducibility but not CUE improvement | \(G_k > 0 \Rightarrow \Delta\text{CUE} > 0\) under faithfulness conditions needed |
| \(\lambda\)-normalization commensurability | **Partial** — commensurability prose exists in §2.5 but the canonical normalization is asserted, not proved | Proof that \(\lambda_C = \log|Z| / (E[|s|] \cdot \log|\Sigma|)\) equalizes maximal contributions needed |

### Where Novelty Can Be Improved

Three areas would substantially strengthen the proposal's claims against referee scrutiny:

1. **CUE-VOI Bridge**: The proposal frames CUE as grounded in classical Value of Information theory, but VOI in the Bayesian decision theory literature is defined as \(\text{VOI} = E[U(\text{decision after observing } y)] - E[U(\text{best decision before})])\). The current CUE definition uses Brier score delta rather than expected utility delta. A formal theorem proving that Brier-delta is a proper scoring rule lower-bounded by VOI (or equivalent up to calibration) would close this claim rigorously.[^14][^15]

2. **Goodhart-resistance structural proof**: The four Goodhart conditions are currently collected as a protocol conjunction (`Goodhart_resistance_summary`) assembled from axioms. A structural theorem proving that violating *any single condition* causes the proxy-rank-agreement guarantee to collapse would make Goodhart-resistance a *theorem* about the architecture rather than a checklist.[^1]

3. **DC Score non-triviality**: The Diverge-Converge Score needs a proof that under a non-degenerate generation process, both monotone convergence and monotone divergence have strictly positive probability of occurring — otherwise DC is trivially satisfied or trivially failed and carries no information.

***

## Part III: Missing Lean Proofs — Complete List with Proof Strategies

### Tier 1 — Critical Gaps (Proposal Claims Not Formalized)

***

#### PROOF-01: `CUE_voi_equivalence`
**Statement**: For a receiver agent \(A\) with Brier-calibrated beliefs, the normalized Brier-delta CUE is a proper lower bound on Shannon Value of Information: \(\text{CUE}(y, A, T) \le \text{VOI}(y, A, T) / |y|_{\text{bits}}\), with equality when the receiver's prior is calibrated and the decision space is binary.

**Proof strategy**: (a) Define a `VOIModel` structure extending `BModel` with a loss function `L: Z → Decision → ℝ` and prior `prior: PMF(Z)`. (b) Define `VOI(y) = E[minLoss(posterior(y))] - minLoss(prior)` where `posterior` is Bayes-updated by `y`. (c) Prove the Brier score is a proper scoring rule satisfying `BrierDelta(y) ≤ KL(posterior ‖ prior)` via Gibbs inequality. (d) Use `shannonEntropy_lt_iff` and the data-processing inequality to chain: `BrierDelta ≤ KL ≤ I(Y;Z)`. (e) Normalize both sides by `|y|_bits` using the existing commensurability λ-normalization scaffold. Key dependencies: `N2_kl_nonneg`, `kl_cross_entropy_decomposition`, `C1_utility_gate`.

***

#### PROOF-02: `shannon_upper_bound_normalization`
**Statement**: For any output \(y\) and task battery \(T\), \(\text{CUE}(y, A, T) \le H_{\max}(Z) / |y|_{\text{bits}}\) where \(H_{\max}(Z) = \log|Z|\) is the Shannon entropy upper bound over the downstream state space.

**Proof strategy**: (a) Since `CUE ≤ VOI/|y|_bits` from PROOF-01, and `VOI ≤ H(Z|prior)` by the non-negativity of residual entropy, and `H(Z|prior) ≤ H_max(Z) = log|Z|` by the maximum entropy theorem for finite alphabets, the chain is: `CUE ≤ log|Z| / |y|_bits`. (b) Formalize `maxEntropy_finite_alphabet : ∀ (p : PMF Z), H(p) ≤ Real.log |Z|` — this requires the standard finite-PMF maximum-entropy proof by Lagrange multipliers or by log-sum inequality. (c) Prove the normalization factor `λ_CUE = 1 / (log|Z| / E[|y|_bits])` makes the normalized CUE lie in `[0, 1]`. Key tactic: `Real.log_le_log` + `Finset.card_pos` + `linarith`.

***

#### PROOF-03: `dual_process_separation_principle`
**Statement (DPS Theorem)**: Let \(\Pi_{\text{frozen}} = \{P, T, A\}\) be the frozen benchmark objects and \(\Pi_{\text{live}} = \{\tilde{R}_D, T_1, R_B^{\to A, \text{train}}\}\) be the live training proxies. If the protocol satisfies all four Goodhart conditions, then for any gradient update step \(\nabla_\theta\), \(\nabla_\theta R_{\text{train}} \perp \Pi_{\text{frozen}}\) (the gradient has zero dependence on frozen objects). Corollary: \(R_{\text{train}}\) cannot Goodhart \(R_{\text{benchmark}}\) through any differentiable path.

**Proof strategy**: This is a *structural independence* theorem, not an algebraic one. (a) Define a `FrozenSet` predicate over the training protocol: `isFrozen(x) ↔ ¬∃ t, updatedAt(x, t)`. (b) Prove `trainReward_gradient_independence`: for any admissible `DTildeTrainingProtocol`, the gradient `∂R_train/∂θ` depends only on live-resampled quantities by unfolding `rewardDTilde` and `T1` definitions. (c) Use `D_tilde_no_fixed_target` (Assumption 4 in document) and `C_corpus_stationarity_condition` (Assumption 5) as structural preconditions. (d) The corollary follows from `trainReward_neq_fullReward` (Theorem 48) by showing the gap is not just definitional but *monotonically diverges* under any gradient update that exploits the frozen P. Key new structure: `GradientIndependence` typeclass with field `no_frozen_dependence`.

***

#### PROOF-04: `CUE_trajectory_monotone_nondecreasing`
**Statement**: For the Step-CUE curve \(C(t) = \text{CUE}(y_{1:t}, A, T)\), if each reasoning step satisfies the utility gate (i.e., partial output at step \(t\) is feasible), then \(C(t)\) is non-decreasing in expectation: \(E[C(t+1)] \ge E[C(t)]\).

**Proof strategy**: (a) Define `CUETrajectory : Fin(T) → ℝ` as a sequence of running CUE values. (b) Formalize the step reward `r_t = CUE(y_{1:t}) - CUE(y_{1:t-1})` as the incremental Brier-delta contribution. (c) Prove `step_reward_nonneg_under_gate`: under the utility gate (feasibility of partial prefix), each additional token cannot reduce the receiver's expected Brier improvement by the law of total expectation — this follows from the tower property: `E[CUE(y_{1:t+1})] = E[CUE(y_{1:t})] + E[r_{t+1}]`, and `E[r_{t+1}] ≥ 0` iff the incremental KL satisfies the gate condition. (d) Key lemma: `brier_tower_property` — Brier-score delta decomposes additively along a Markov chain of receiver belief updates. This is the creative-trajectory analogue of `B3_mixture_lb` but for sequential rather than mixture updates.

***

#### PROOF-05: `DC_score_existence`
**Statement**: For any non-degenerate generation process, there exists an output trajectory \(y_{1:t^*+k}\) such that \(R_B^{\to A}(t)\) is strictly increasing for \(t \le t^*\) (divergence phase) and strictly decreasing for \(t > t^*\) (convergence phase), i.e., DC = 1 is achievable.

**Proof strategy**: (a) Define `DCTrajectory` as a predicate on trajectories: `∃ t* : Fin(T), ∀ t ≤ t*, receiverEntropy(t) < receiverEntropy(t+1) ∧ ∀ t > t*, receiverEntropy(t) > receiverEntropy(t+1)`. (b) Prove existence by *explicit construction*: on a 3-step trajectory over alphabet Z with `|Z| ≥ 2`, construct step 1 as maximum-entropy output (any uniform distribution over Z), step 2 as a partial commitment (bimodal over Z), step 3 as full commitment (point mass). (c) Prove the constructed trajectory satisfies the DC predicate using `shannonEntropy_point_mass` (Theorem 2) and `B1_dominance` (Theorem 8). (d) Prove the complementary: for monotone convergence (only step 3 type) and monotone divergence (only step 1 type), DC = 0. Use `B2_no_collapse` to show that pure convergent trajectories are dominated.

***

#### PROOF-06: `receiver_expansion_clarification_separation`
**Statement**: The receiver expansion label \(R_B^{\to A}\) is *not* a monotone function of CUE. Specifically, there exist outputs \(y_1, y_2\) with equal CUE but opposite \(R_B^{\to A}\) labels: one expanding (high entropy, \(R_B^{\to A}(y_1) > \delta_E\)) and one clarifying (low entropy, \(R_B^{\to A}(y_2) < \delta_E\)).

**Proof strategy**: (a) This is a *separation theorem* proved by explicit finite construction, mirroring `BC_tension_example` (Theorem 69) and `BCD_tension_example` (Theorem 70). (b) On `Fin(2)` for the decision space, construct `y_expand` such that `DecQual(after) - DecQual(before) = ε` (equal small CUE) and `H(Gen(A|y_expand)) = log 2` (maximum expansion). Construct `y_clarify` such that the same CUE delta holds but `H(Gen(A|y_clarify)) = 0` (point mass — receiver fully committed). (c) Prove: `CUE(y_expand) = CUE(y_clarify)` (equal utility improvement) but `receiverEntropy(y_expand) > δ_E > receiverEntropy(y_clarify)`. (d) Key dependency: `BContrast_distinguishes` pattern (Theorem 16) applied to the entropy dimension vs. the expansion label dimension.

***

#### PROOF-07: `Gk_implies_CUE_improvement`
**Statement**: Under the faithfulness assumption and product-kernel bridge, positive interaction gain implies that the joint output achieves strictly higher CUE than any single-agent output: \(G_k(q, y) > 0 \Rightarrow \text{CUE}(y_{\text{joint}}, A, T) > \max_i \text{CUE}(y_i, A, T)\).

**Proof strategy**: (a) This is the *decision-quality bridge* connecting the existing `MAS_irreducibility` theorem (Theorem 26) to the CUE layer. (b) Define `CUEModel` extending `KAgentModel` with a Brier-delta layer: each agent's downstream entropy \(H_i\) induces a Brier score via the receiver calibration map. (c) Formally, `H_i(q, y_i) < H_J(q, y)` from `kAgent_irreducibility` (Theorem 31). (d) Prove that under a receiver with bounded calibration error `ε_cal`, `BrierDelta_J ≥ BrierDelta_i + (H_J - H_i - 2ε_cal)`. This connects entropy irreducibility to CUE improvement under a `BrierCalibrationAssumption`. (e) Key dependency: `MAS_irreducibility` + new `brier_entropy_calibration_link` lemma.

***

#### PROOF-08: `lambda_normalization_equalizes_maxima`
**Statement**: The canonical normalization \(\lambda_C = \log|Z| / (E[|s|] \cdot \log|\Sigma|)\) ensures that \(\max_{y} R_B(q,y) = \max_{y} \lambda_C R_C(y)\) at their respective Shannon maxima.

**Proof strategy**: (a) The maximum of \(R_B\) over feasible \(y\) is `H_max(Z) = log|Z|` (from PROOF-02). (b) The maximum of \(R_C\) is bounded by `L(s) ≤ |s| · log|Σ|` (maximum compression cost for a string of length |s| over alphabet Σ, achieved by a string with maximum Kolmogorov complexity). (c) Setting `λ_C · (E[|s|] · log|Σ|) = log|Z|` yields the canonical formula. (d) Prove this as `normalization_eq_max_contribution`: `λ_C * max_novelty = max_entropy` where max_novelty is computed from the `AbstractCompressor` axioms and max_entropy from `maxEntropy_finite_alphabet`. Key tactic: `ring` + `Real.log_pow` + `Finset.sup_le`.

***

### Tier 2 — Existing Axioms That Should Become Theorems

***

#### PROOF-09: `entropy_mixture_concavity` (promote from axiom)
**Current status**: `entropy_mixture_concavity_axiom` — explicitly noted as "an axiom in the current Lean development, not a theorem derived from primitives".[^1]

**Statement**: For finite \(Y, Z\), model \(m\), prompt \(q\), and mixture \(\pi : \text{PMF}(Y)\), \(\sum_y \pi(y) H_q(y) \le H(\mu_q(\pi))\).

**Proof strategy**: (a) Use the log-sum inequality: for non-negative \(a_i, b_i\), \(\sum a_i \log(a_i/b_i) \ge (\sum a_i)\log(\sum a_i / \sum b_i)\). (b) Apply to PMF sums over the mixture: \(H(\mu_q(\pi)) = -\sum_z [\sum_y \pi(y) T(z|q,y)] \log[\sum_y \pi(y) T(z|q,y)]\). (c) Jensen's inequality for the concave log function gives the desired bound. (d) In Lean 4: use `Real.inner_le_iff` (for log concavity) + `Finset.sum_comm` (for double-sum interchange) + `PMF.bind_apply` (for mixture kernel definition) + `linarith`. This follows the standard proof in Cover & Thomas §2.1[^16] and can be built on the existing `shannonEntropy_nonneg` infrastructure.

***

#### PROOF-10: `productKernel_joint_entropy_eq_sum` (promote from axiom)
**Current status**: Assumption 3 in the document — "Lean records this as an axiom because the full proof requires a product-sum expansion together with the logarithm product rule".[^1]

**Statement**: For independent product coupling, \(H(Z_1, Z_2) = H(Z_1) + H(Z_2)\).

**Proof strategy**: (a) Unfold the joint entropy definition: \(H(Z_1, Z_2) = -\sum_{z_1, z_2} p(z_1) p(z_2) \log[p(z_1) p(z_2)]\). (b) Use logarithm product rule: \(\log[p(z_1) p(z_2)] = \log p(z_1) + \log p(z_2)\). (c) Split the double sum: \(\sum_{z_1,z_2} p(z_1)p(z_2)[\log p(z_1) + \log p(z_2)] = \sum_{z_1} p(z_1)\log p(z_1) \cdot \sum_{z_2} p(z_2) + \sum_{z_2} p(z_2)\log p(z_2) \cdot \sum_{z_1} p(z_1)\). (d) Since PMF sums are 1, this reduces to \(H(Z_1) + H(Z_2)\). In Lean 4: `Finset.sum_product'` + `PMF.sum_eq_one` + `Real.log_mul` + `ring`.

***

#### PROOF-11: `gain_zero_iff_productKernel` (promote from axiom)
**Current status**: Assumption 2 — "this assumption is a structural bridge that should be proved from marginal independence and entropy additivity in a future measure-theoretic refinement".[^1]

**Statement**: \(G_k(q, y) = 0 \iff \text{isProductKernel}(m)\).

**Proof strategy**: (a) (\(\Leftarrow\)): If `isProductKernel(m)`, then `H_J = H_1 + H_2` by PROOF-10. Therefore `G = H_J - max(H_1, H_2) = H_1 + H_2 - max(H_1, H_2) = min(H_1, H_2) ≥ 0`, but this does not give \(G = 0\) in general — this exposes a gap: product kernel implies \(G \ge \min(H_1, H_2) \ge 0\), not \(G = 0\). **Critical issue**: The biconditional as stated is false for the current definition of \(G\) when both agents have non-zero individual entropy. (b) Resolution: \(G = 0\) requires the stronger condition \(H_J = \max_i H_i\), i.e., the joint process adds nothing over the best individual. For product kernels, \(H_J = H_1 + H_2 > \max(H_1, H_2)\) whenever both entropies are positive. **The current axiom as stated is incorrect for \(G = H_J - \max_i H_i\) under product kernels with positive marginals.** (c) Corrected formulation: \(G_k = 0 \iff H_J = \max_i H_i\), which holds iff one agent's output fully determines the joint process (degenerate case). The product-kernel bridge must be restated. This is a **substantive gap** requiring both a corrected definition and a new proof strategy involving conditional independence.

***

#### PROOF-12: `plugIn_entropy_bias_bound` (promote from axiom)
**Current status**: `plugIn_bias_axiom` — imported Miller-Madow scale bound, `|Ĥ_plug - H| ≤ |Y|/(2n)`[^1].

**Statement**: For alphabet Y, counts `c : Y → ℕ`, sample size `n > 0`, the plug-in entropy estimator satisfies `|Ĥ_plug(c,n) - H| ≤ |Y|/(2n)`.

**Proof strategy**: (a) The Miller-Madow bias correction gives `E[Ĥ_plug] ≈ H + (|Y|-1)/(2n)`. The bound `|Y|/(2n)` is the dominant term. (b) In Lean, formalize this via a concentration inequality for bounded random variables (each `-p log p` term is bounded in `[0, 1/e]`). (c) Use Hoeffding's inequality applied to the empirical frequency estimator, combined with the Lipschitz continuity of `-x log x`. (d) This requires importing statistical concentration bounds — the `StatisticalLearningTheory` Lean 4 library[^13] may provide `Hoeffding_inequality` as a dependency. (e) Alternatively, accept this as a "trusted import" axiom with explicit reference to Miller & Madow (1955) and note the dependency in the formal document. Key structure: `BoundedLipschitz` + `empiricalMeasure_concentration`.

***

### Tier 3 — New Proofs for Proposal-Specific Reward Structure

***

#### PROOF-13: `R_train_proxy_faithfulness_collapse`
**Statement (Structural Goodhart Theorem)**: If any *single* Goodhart condition is violated, the proxy-rank-agreement guarantee `D_tilde_proxy_rank_agreement` (Theorem 63) collapses — i.e., the violation implies the existence of a training path where \(\tilde{R}_D(s_1) > \tilde{R}_D(s_2)\) but \(R_D(s_1, H, P) < R_D(s_2, H, P)\).

**Proof strategy**: Prove four sub-theorems, one per Goodhart condition:
- `D_tilde_fixed_target_exploits_rank`: If `Q_probe` contains a fixed string across steps, the policy can memorize it to maximize \(\tilde{R}_D\) without improving \(R_D\) on the frozen \(P\). Proof by explicit adversarial construction on `Fin(2)` probe space.
- `C_nonstationarity_breaks_antipadding`: If corpus \(H\) includes policy outputs, `C3_no_padding_exploit` breaks because the padding baseline shifts. Proof by showing `C3` requires `corpus_stationary` precondition.
- `B_loose_gate_admits_incoherence`: If `τ ≤ H_q(y_random)`, then incoherent high-entropy outputs pass the gate and `R_B` is maximized by noise. Proof using `CBase_infeasible_can_maximize` pattern (Theorem 17).
- `proxy_decay_undetected_invalidates_ranking`: If `ρ(R̃_D, R_D) < ρ_min` without detection, proxy rank-agreement fails. Proof by constructing `s_1, s_2` where proxy and benchmark diverge beyond `2ε`.

***

#### PROOF-14: `step_CUE_gamma_positivity_characterization`
**Statement**: The saturation-rate parameter \(\gamma > 0\) in the fitted Step-CUE curve \(C(t) = C_\infty(1 - e^{-\mu t}) + \gamma t\) is strictly positive iff the trajectory has at least one step where marginal CUE improvement exceeds the exponential saturation prediction, i.e., the output contains genuine continuous discovery.

**Proof strategy**: (a) Define `StepCUEFit : Fin(T) → ℝ` as the fitted exponential-plus-linear curve. (b) Prove `gamma_pos_iff_superexponential_step`: \(\gamma > 0 \iff \exists t, C(t) > C_\infty(1 - e^{-\mu t})\). (c) This is a real-analysis statement on finite sequences; in Lean 4, use `Real.exp_pos`, `Real.sub_pos`, and `Finset.exists_mem` with the sequence ordering. (d) Prove the converse: if every step CUE increment is below the exponential curve, then the linear component is zero or negative.

***

#### PROOF-15: `dense_reward_DC_bonus_incentivizes_exploration`
**Statement**: The terminal DC bonus \(r_{DC}\) in the dense trajectory reward is necessary — without it, the optimal policy under \(r_t\) alone produces monotone-convergent trajectories (DC = 0) despite maximizing cumulative CUE.

**Proof strategy**: (a) Define a 3-step MDP with states being receiver entropy levels \(\{H_L, H_M, H_H\}\) and show the optimal Bellman solution without \(r_{DC}\) terminates at \(H_H\) via direct path (DC = 0 monotone divergence) or via \(H_L\) (DC = 0 monotone convergence). (b) With \(r_{DC}\) added, prove the path through \(H_M \to H_H \to H_L\) (diverge then converge) achieves strictly higher cumulative reward. (c) Key Lean structure: `Bellman_optimality_finite_MDP` (can be built from `Finset.sup` with the recursive value function).

***

#### PROOF-16: `2GRPO_advantage_quantization_unbiased`
**Statement**: In 2-GRPO with two rollouts \(\{y_1, y_2\}\), the advantages \(A_1 = +1/\sqrt{2}, A_2 = -1/\sqrt{2}\) (assuming \(r_1 > r_2\)) are unbiased estimators of the signed rank of the policy's output quality: \(E[A_i | \text{rank}(y_i)] = \pm \frac{1}{\sqrt{2}}\).

**Proof strategy**: (a) This formalizes the 2-GRPO theory from Wu et al.[^9] at the Lean level. (b) Prove `advantage_sign_correct` (already Theorem 58) generalizes to the 2-rollout case with quantized values. (c) Key step: prove that the group denominator `\sqrt{(r_1 - \mu)^2 + (r_2 - \mu)^2 + \epsilon}` equals `|r_1 - r_2|/\sqrt{2}` when `\mu = (r_1 + r_2)/2`. (d) Prove this using `Real.sqrt_sq` + `ring` in Lean 4.

***

## Part IV: Document Restructuring — Task List

The current document is organized as: Problem → Framework → B/C/D objectives → Estimators → MAS → Training Signal → Goodhart → Proxy Validation → Algorithm. This linear order **does not match the proposal's logical priority** where CUE is the primary gate and B/C/D are subordinate components. The restructuring should make the receiver-grounded CUE the definitional anchor from which everything else derives.

### Phase 0 — Pre-Restructure Cleanup

- [ ] **P0.1** Remove all "Axiom status" remarks on `entropy_mixture_concavity_axiom`, `gain_zero_iff_productKernel`, `productKernel_joint_entropy_eq_sum`, and `plugIn_bias_axiom`. Replace each with a placeholder section header "Proof Target: [PROOF-09/10/11/12]" and the proof strategy from Part III above, to make the proof debt visible.
- [ ] **P0.2** Correct the statement of `gain_zero_iff_productKernel` (PROOF-11 identifies this as incorrectly stated for positive-marginal product kernels). Add a "Known Gap" note in the current Assumption 2 section.
- [ ] **P0.3** Audit all 73 theorems in Table 2 against the proposal's R_creativity formula. Tag each as: `[CORE]` (directly referenced by R_creativity), `[SUPPORT]` (supports a CORE theorem), or `[INFRASTRUCTURE]` (general information-theoretic scaffolding). This tagging drives the new section ordering.

### Phase 1 — New Introductory Part: CUE Formalization

- [ ] **P1.1** Add **Part 0: Receiver-Grounded Creativity Foundation** as the new Part I, before the existing "Creativity Definition" section. This part should contain:
  - Definition of `CUEModel` extending `BModel` with `brierScore`, `decisionQuality`, and `taskBattery` fields
  - `CUE_definition`: formal statement of CUE as Brier-delta per bit
  - PROOF-01 and PROOF-02 (VOI equivalence and Shannon upper bound)
  - `CUE_primary_gate_axiom`: formal statement that zero/negative CUE disqualifies all downstream metrics

- [ ] **P1.2** Move the current "Framework Preview" (§2.1–2.6) to follow Part 0, now framing B, C, D as *components of the CUE-gated score* rather than equal co-equal dimensions. Update equation numbering.

- [ ] **P1.3** Add a **Formal Correspondence Table** at the end of Part 0 mapping proposal constructs to Lean theorem names:

| Proposal Construct | Lean Theorem | Status |
|---|---|---|
| CUE(y, A, T) | `CUE_definition` | NEW (PROOF-01) |
| R_D gate | `D1_utility_gate`, `D4_zero_for_corpus_copy` | Existing |
| R_B^{→A} label | `receiver_expansion_clarification_separation` | NEW (PROOF-06) |
| G_k | `kAgent_irreducibility` + `Gk_implies_CUE_improvement` | Partial (PROOF-07) |
| Step-CUE γ | `CUE_trajectory_monotone_nondecreasing` | NEW (PROOF-04) |
| DC Score | `DC_score_existence` | NEW (PROOF-05) |
| DPS Principle | `dual_process_separation_principle` | NEW (PROOF-03) |

### Phase 2 — Reorganize Main Parts Around Logical Priority

- [ ] **P2.1** Restructure Part I ("Creativity Definition") into three tiers:
  - **Tier A: Primary Gate** — CUE (new Part 0 content)
  - **Tier B: Structural Filters** — R_D gate (existing §6), R_B^{→A} label (new PROOF-06)
  - **Tier C: Interaction Certification** — G_k (existing §8 + PROOF-07)
  
- [ ] **P2.2** Move the Commensurability section (§2.5) to *after* PROOF-02 and PROOF-08, since it now has a proof rather than an assertion. Rename to "Commensurability: Proved Normalization" and include the full PROOF-08 derivation.

- [ ] **P2.3** Merge the current **Part II Benchmark Objectives** with new trajectory-level diagnostics (PROOF-04, PROOF-05) into a single **Part II: Benchmark Metrics (Frozen)**. Subsections:
  - §A: CUE as primary validity gate (from Part 0)
  - §B: Structural Novelty Gate (existing R_D theorems)
  - §C: Receiver Expansion Label (PROOF-06)
  - §D: Multi-Agent Interaction Gain (existing + PROOF-07)
  - §E: Step-CUE Curve γ (PROOF-04)
  - §F: Diverge-Converge Score (PROOF-05)

- [ ] **P2.4** Create a new **Part III: Dual-Process Separation Principle** containing PROOF-03 and PROOF-13 (Structural Goodhart Theorem). This should sit *between* the benchmark metrics and the training signal, making explicit that the separation is what makes the benchmark non-Goodhartable. Currently this logical link is implied but not formalized as a standalone part.

- [ ] **P2.5** Rename current **Part III (2-CGRPO Training Signal)** to **Part IV** and add PROOF-14, PROOF-15, PROOF-16 as new subsections:
  - §14a: Step-CUE Dense Reward Formalization (PROOF-04 applied to r_step)
  - §14b: DC Bonus Necessity Theorem (PROOF-15)
  - §14c: 2-GRPO Advantage Quantization (PROOF-16)

### Phase 3 — Goodhart Resistance Enhancement

- [ ] **P3.1** Replace the current "Goodhart Resistance Summary" (Theorem 62) with a **two-level Goodhart structure**:
  - **Level 1: Architectural Goodhart Resistance** — `dual_process_separation_principle` (PROOF-03): gradient independence from frozen objects
  - **Level 2: Protocol Goodhart Resistance** — existing `Goodhart_resistance_summary` (Theorem 62): four protocol conditions

- [ ] **P3.2** Add the **Structural Goodhart Collapse Theorem** (PROOF-13) as a falsifiability anchor: explicitly proves that each condition's violation admits a specific exploitation path. This converts the Goodhart section from a checklist into a theorem-theorem pair (protection ↔ violation).

- [ ] **P3.3** Update Table 2 (Lean-checked claims) to include all 16 new proof targets. Add a `Status` column: `Machine-Checked`, `Proof-in-Document`, `Proof-Strategy`, or `Known-Gap`.

### Phase 4 — Empirical-Formal Bridge Appendix

- [ ] **P4.1** Add **Appendix E: Experimental Falsifiability Map** translating each of E1–E5 from the proposal into formal statements derivable from the Lean theorems:
  - E1 (Judge Calibration) ↔ `B_rank_preserved`, `C_rank_preserved` at `gap > 2ε`
  - E2 (Calibration Failure Replication) ↔ `BContrast_distinguishes` (Theorem 16): CUE maintains rank where reward models fail
  - E3 (Repetition vs. Score) ↔ `receiver_expansion_clarification_separation` (PROOF-06): R_B^{→A} anti-correlated with cosine similarity by construction
  - E4 (Discriminant Validity) ↔ `BCD_tension_example` (Theorem 70): Pareto-frontier non-empty with distinct maximizers
  - E5 (Receiver Stability) ↔ PROOF-05 + `D_tilde_proxy_rank_agreement` (Theorem 63)

- [ ] **P4.2** Add **Appendix F: 2-GRPO Training Protocol Specification** containing the formal `DTildeTrainingProtocol` instantiation for the Qwen 27B training run, including: concrete `ρ_min` calibration procedure, `τ` calibration against incoherent baselines, corpus stationarity schedule, and checkpoint monitoring frequency. This makes the Lean-certified guarantees operationally executable.

- [ ] **P4.3** Add a **Known Gaps Registry** at the end of the document: a numbered list of every axiom not yet promoted to a theorem, every assumption not yet proved from primitives, and the corresponding proof strategy. Currently these are scattered across the document; centralizing them makes the proof debt auditable and aids reproducibility.

### Phase 5 — Abstract and Introduction Rewrite

- [ ] **P5.1** Rewrite the abstract to lead with CUE as the primary contribution (currently the abstract leads with "information dynamics" and B/C/D without mentioning CUE or VOI grounding). New abstract should state: "We introduce Creative Update Efficiency (CUE), a receiver-grounded creativity metric defined as Brier-score delta per output bit, proved to be a lower bound on Shannon Value of Information. We formalize [N] Lean 4 theorems certifying..."

- [ ] **P5.2** Add a **Contributions Hierarchy** box in the introduction distinguishing:
  1. *Primary*: CUE as VOI-grounded creativity gate (new, no prior work)
  2. *Secondary*: Dual-Process Separation Principle (Goodhart-resistance as architectural theorem)
  3. *Tertiary*: Trajectory diagnostics (Step-CUE γ, DC Score)
  4. *Supporting*: B/C/D information-theoretic scaffolding (from current document, strengthened)

- [ ] **P5.3** Add a **Relationship to ComplLLM** subsection in the introduction explicitly stating what the proposal generalizes (open-ended tasks, no binary payoff decomposition, bit-normalized efficiency) and what it inherits (decision quality as training signal, receiver-outcome focus).[^7]

***

## Part V: Priority Order for Implementation

| Priority | Proof ID | Rationale |
|---|---|---|
| P0 | PROOF-09 | Unblocks B3b_jensen_anti_collapse — most Lean dependencies blocked on this axiom |
| P0 | PROOF-10 | Unblocks gain_zero_iff_productKernel correctness (PROOF-11 depends on this) |
| P0 | PROOF-11 (corrected) | Must fix incorrect axiom before any MAS irreducibility claims hold for product kernels |
| P1 | PROOF-01 | Core novelty claim; required for rewritten abstract |
| P1 | PROOF-02 | Required for commensurability proof (PROOF-08) |
| P1 | PROOF-03 | DPS Principle — headline structural result |
| P2 | PROOF-07 | CUE-decision bridge for MAS claims |
| P2 | PROOF-06 | R_B^{→A} separation — required for E3 falsifiability |
| P3 | PROOF-04, PROOF-05 | Trajectory diagnostics — required for Phase 2 training protocol |
| P3 | PROOF-13 | Structural Goodhart — required for revised Part IV |
| P4 | PROOF-08, PROOF-12 | Commensurability + bias bound — important but not blocking |
| P4 | PROOF-14, PROOF-15, PROOF-16 | 2-GRPO specifics — self-contained, implement last |

---

## References

1. [bridged_validation.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/26864273/a8586179-ac88-4c11-9ad5-3bd6db6594da/bridged_validation.pdf?AWSAccessKeyId=ASIA2F3EMEYESMHN6QQZ&Signature=Y1guvai6I96CUllgRfYllUL9Vck%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEGoaCXVzLWVhc3QtMSJGMEQCICFdw%2Fj52v2vZFBdbTr87nxP3gP3TScn74hXrSLlMiNlAiAOOQ9BfEfkG%2FymobkRJ0xvJwZFYsx0cGKlBlplGKAgqSrzBAgzEAEaDDY5OTc1MzMwOTcwNSIMrdc53njzHwPktYypKtAEFNVT9QoLqvcVZ81Q6H2%2FzLvtUiLhsH4p8xQv8MY36t62wHq3o%2Bb7JxfoH5GCh%2FTFbyHuMRtqD59AOFZDvZ84nKwufKKTzv0iUe0DPQ0nFoVaJaa7z8gcLTnvWrf9%2Bhq6j1wd9HDwvy%2F6doaPxk4t%2FdgJHgT9MAHgW%2FssNFbWPBPEAVVUex3JPC8NjfrOgdnfmcsqvoHpvD9gCtuJHRimKW6ip3DXyw%2FEGpiRrFLKzXfa2UrZmTQs1yRy7wQuiJZazXwdfVn1WIPwXgILgjkBIifJZzuaqReZAkiOOAf2Nk%2FHEVFgOmu%2BMZpyCJrZSgCuHPb7D5v%2BNA4gKWTIDhfoVYvnPaebRIyYyimsyTelcNpl36Ot3QwVkVITWo5lap%2F1dJfmJFEOoej6mlFL9qPWZd9Lh2nrrC7Qm1bIBMZAPWUtVTMjDvareox%2BIOw4MsVZvqW%2F%2FTRNV5XfWX8DYr5k6N%2BIDBNcIB2Jw8hL%2BdhAH3%2F%2FmJieoC%2FG%2FFoutTgddG%2BKDT5%2Fk%2BUSBQAX3sRTY53NdTPNqVu%2FDOElnesMIFHglU5%2Br8UlcanLaPrkLtQEbSzjB5Cx4ysVJZm4VQ1jI0k3t%2BJY3G%2BXWoWuKQSkLS9KTID%2Fu1DjlWGfGAOeGPFfzQ5sM3KyGQIxflQJuLo9H2w9%2BXG2eyB%2B5BdPJe3taZ7UXQhfIAsd8A6nSbnzBjqyOKJxscG6d0fy69cr9VxjvvBHv00%2BeANlkvpPaY4JGCtJgSdTPtKTiI%2BA8SWx1FhSUnRYmbVEyK7Va9qFo%2Bpso657yDCShd%2FSBjqZAabg%2FaTVIgqm3tHLnA0di87uRoOgtb1IfvGPaW34iNgjT%2BVRvxiEG23Uf3FKsYfexzuH0xxfWmyhZWj5LkIf1lhMn9girE050VdMP1BsvDqFC3Yk2Lqi%2FFtR4bPekuWXmSZXsBZLktT%2F46FSWeZaepxzx%2F23rprFeUSthRSPvKcWD8S0bWu8v%2Bk5frru7dZ9uVW9ElWwwHk98A%3D%3D&Expires=1784139877) - **page-1**
FormalCreativityforMAS-LLMs:ALean-CheckedBridgefromInformationObjectivestoBenchmarkClaims...

2. [CreativityPrism: A Holistic Evaluation Framework for Large Language Model Creativity](https://arxiv.org/abs/2510.20091) - Creativity is often seen as a hallmark of human intelligence. While large language models (LLMs) are...

3. [A Critical Analysis of Existing Creativity Evaluations](https://aclanthology.org/2026.eacl-long.297.pdf) - by LC Lu · 2026 · Cited by 7 — We examine, analyze, and compare four rep- resentative creativity mea...

4. [Human-Grounded Boundaries for Evaluating LLM Creativity](https://openreview.net/forum?id=Hpp265n83f) - We introduce creativity coverage, a novel framework for evaluating large language model (LLM) creati...

5. [Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling ...](https://www.wispaper.ai/en/blog/diversity-collapse-multi-agent-llm-systems-20260421/eng) - This paper investigates "Diversity Collapse" in Multi-Agent LLM Systems (MAS) during open-ended idea...

6. [Tanishq5262/Artificial-Hivemind-Audit - Hugging Face](https://huggingface.co/Tanishq5262/Artificial-Hivemind-Audit) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

7. [ComplLLM: Fine-tuning LLMs to Discover Complementary Signals for Decision-making](https://www.arxiv.org/abs/2602.19458) - Multi-agent decision pipelines can outperform single agent workflows when complementarity holds, i.e...

8. [ComplLLM: Fine-tuning LLMs to Discover Complementary Signals for Decision-making](https://www.newx.sg/paper/detail/8d60805b-11e7-11f1-b84c-00163e10baa7)

9. [It Takes Two: Your GRPO Is Secretly DPO](https://openreview.net/forum?id=3axBqFqDgk) - by Y Wu · Cited by 20 — The work theoretically and empirically showed that 2-GRPO can be compatible ...

10. [It Takes Two: Your GRPO Is Secretly DPO - alphaXiv](https://www.alphaxiv.org/overview/2510.00977v1) - View recent discussion. Abstract: Group Relative Policy Optimization (GRPO) is a prominent reinforce...

11. [FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models](http://arxiv.org/abs/2505.02735) - Formal mathematical reasoning remains a critical challenge for artificial intelligence, hindered by ...

12. [shannon-entropy | Reservoir](https://reservoir.lean-lang.org/@SamuelSchlesinger/shannon-entropy) - A formalization of Shannon's seminal 1948 paper defining entropy.

13. [Statistical Learning Theory in Lean 4: Empirical Processes ...](https://arxiv.org/html/2602.02285v1) - We present the first comprehensive Lean 4 formalization of statistical learning theory (SLT) grounde...

14. [Value of Information Analysis in Models to Inform Health Policy](https://pmc.ncbi.nlm.nih.gov/articles/PMC7612603/) - by CH Jackson · 2022 · Cited by 53 — Value of information (VoI) is a decision-theoretic approach to ...

15. [Microsoft Word - Manuscript Value of Information final.docx](https://mediatum.ub.tum.de/doc/1214671/1214671.pdf)

16. [Proof: Concavity of the Shannon entropy](https://statproofbook.github.io/P/ent-conc.html) - Concavity Theorem: The entropy is concave in the probability mass function p p , are probability mas...

