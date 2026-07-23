# Plan: Core Argument Under 7 Pages with Indexed Appendix

Target: `bridged_validation.tex` (currently 43 pages) → a **≤ 7-page core** that
carries the complete logical argument, plus an **indexed appendix** of results
(`[A.1]`, `[B.3]`, …) that the core cites for every supporting detail. Nothing
is deleted; everything not on the critical path moves to an appendix with a
stable index the core can reference.

---

## 1. The core logical chain (what stays in the body)

The argument the 7 pages must carry, as a single dependency chain. Each node
lists the *only* formal statements that appear in the body (statement + one- or
two-sentence proof sketch; full proofs go to the appendix).

### §1 — Thesis and the CUE gate (~1.5 pages)

The claim: creativity is receiver-grounded — an output is creative only insofar
as it improves a receiver's decisions, efficiently per bit, and this gate is
formally certified.

| Body statement | Lean name | Full detail moves to |
|---|---|---|
| CUE definition (Brier-delta per bit) | `cue`, `CUEModel` | [A.1] |
| CUE ≤ VOI / bits, equality when calibrated | `CUE_voi_equivalence` (PROOF-01) | [A.2] |
| CUE ≤ log\|Z\| / bits | `shannon_upper_bound_normalization` (PROOF-02) | [A.3] |
| Zero/negative CUE disqualifies everything | `CUE_primary_gate` | [A.4] |

One callout box: *CUE gates everything; B/C/D are components of the gated
score.* The max-entropy lemma (`maxEntropy_finite_alphabet`) is cited as [A.5],
not stated.

### §2 — The gated components, by invariant only (~1 page)

One paragraph + one compact table. Each component is characterized by the
single invariant that distinguishes it from a rubric, with the theorem cited by
index. No definitions of BModel/compressors in the body.

| Component | The one invariant stated in the body | Cite |
|---|---|---|
| B (expansiveness) | gate + strict entropy dominance; mixture anti-collapse now axiom-free | [B.1]–[B.4] |
| C (novelty) | gate + no credit for copies or padding | [B.5]–[B.7] |
| D (structural influence) | gate + zero for corpus copies, frozen probe set | [B.8]–[B.10] |
| Baseline contrast | utility-only and novelty-only rewards provably miss what B/C enforce | [B.11]–[B.12] |
| Commensurability | λ_C provably equalizes the B and C Shannon maxima | [B.13] (PROOF-08) |

### §3 — Dual-Process Separation and Structural Goodhart (~1 page)

This is the headline structural result and must stay in the body in full
statement form:

- **DPS theorem** (`dual_process_separation_principle`, PROOF-03): the training
  gradient is structurally independent of every frozen benchmark object.
  Corollary `Rtrain_cannot_goodhart_frozen`. Proof sketch: two sentences.
- **Structural Goodhart Collapse** (`structural_goodhart_collapse`, PROOF-13):
  any admitted rank reversal refutes the proxy-rank-agreement guarantee; each
  of the four protocol violations admits one. The four exploitation paths
  become a 4-line list; the constructions go to [C.1]–[C.4].
- Two-level Goodhart framing (architectural vs. protocol) in one short
  paragraph.

### §4 — Training signal and its certificates (~1.5 pages)

- Definitions of `R_train` vs `R_full` (two display equations) and the
  machine-checked split `trainReward_neq_fullReward` [D.1].
- DPO directionality: one theorem statement covering B/C/D̃
  (`DPO_B_gradient_correct`, `DPO_C_gradient_correct`,
  `DPO_D_tilde_gradient_correct`), scaffolding to [D.2]–[D.6].
- The four Goodhart-resistance conditions as a 4-row checklist with the
  conjunction theorem `Goodhart_resistance_summary` [D.7]; per-condition
  lemmas to [D.8]–[D.11].
- 2-GRPO quantization (`twoGRPO_advantage_quantization_unbiased`, PROOF-16):
  statement only — advantages quantize to ±1/√2, so the update is a signed-rank
  step [D.12].

### §5 — Proxy validation and resolution (~1 page)

- `D_tilde_proxy_rank_agreement` (rank agreement under faithfulness) and
  `ProxyFaithfulness_decay_detectable` (the ρ_min early-warning invariant),
  statements only [E.1]–[E.3].
- The Training-to-Benchmark correspondence table stays in the body (it *is*
  the argument), compressed to 4 rows; the asymmetry remark (ρ_min is the only
  empirical link) stays as one sentence.

### §6 — Multi-agent bridge and the algorithm (~1 page)

- `kAgent_irreducibility` + corrected `gain_zero_iff_joint_eq_max`
  (statements; the correction note is one sentence citing [F.2]).
- `Gk_implies_CUE_improvement` (PROOF-07): gain beyond 2ε_cal forces joint CUE
  above every agent — this closes the loop back to §1's receiver grounding.
- 2-CGRPO two-phase protocol as an 8-line numbered list; full protocol spec
  stays in appendix [G.2].
- Closing paragraph: the chain CUE-gate → certified components → structural
  separation → certified training → monitored proxy → receiver-grounded MAS
  gain, each link machine-checked.

**Page budget: 1.5 + 1 + 1 + 1.5 + 1 + 1 = 7.0 pages** (with the abstract
counted inside §1's budget; trim §2's table padding if overflow).

---

## 2. Indexed appendix structure (where everything else goes)

Every appendix item gets a stable index the core cites. Proposed scheme —
`\newtheorem{appres}{Result}[section]` inside each appendix so items render as
**Result A.1**, **Result B.3**, etc., plus a one-page **Appendix Index** table
at the front of the appendices mapping index → Lean name → one-line statement.

| Appendix | Contents (moved from current body) |
|---|---|
| **A. CUE foundation** | CUEModel fields, PROOF-01/02 full proofs, `maxEntropy_finite_alphabet` + `klDivergence_uniform_eq` derivation, expansion/clarification separation (PROOF-06) construction |
| **B. Component theorems** | All B/C/D/N theorem statements + proofs (B1–B3b incl. promoted `entropy_mixture_concavity`, C1–C3, D1–D4, N-surface), baseline contrasts, commensurability derivations, entropy layer, helper lemmas |
| **C. Structural Goodhart** | Rank-reversal machinery, the explicit reversal instance, the four exploitation-path constructions |
| **D. Training scaffolding** | DPO logit/softmax/transfer lemmas, RL-signal lemmas (advantage sign, group denominators), per-condition Goodhart lemmas, 2-GRPO `groupScale_eq`/antisymmetry, DC-bonus necessity (PROOF-15) |
| **E. Proxy bridge** | ProxyFaithfulness/ProxyMonitor definitions, rank-agreement proof, decay-detection proof |
| **F. Multi-agent** | KAgentModel, faithfulness discussion, corrected-axiom refutation (`productKernel_additivity_gain_pos`), O-information sign bridges, `shannonEntropy_jointPMF` proof (PROOF-10), estimator detection theorems |
| **G. Protocols & experiments** | Hold-out probe protocol, 2-GRPO training protocol spec, falsifiability map (E1–E5), benchmark spec, estimator calibration |
| **H. Trajectory diagnostics** | Step-CUE monotonicity (PROOF-04), γ characterization (PROOF-14), DC existence/non-triviality (PROOF-05) — cited from §1 and §4 in one sentence each |
| **I. Ledgers & registry** | All validation-ledger tables, the CUE-layer status tables, Known Gaps Registry (KG-1..KG-4), composability examples |

Rule: the body never states a lemma whose only role is to support another
statement — it cites the index. The body keeps exactly one display equation per
concept.

---

## 3. Mechanical steps

1. Create `docs/core_argument.tex` (new 7-page master) and
   `docs/appendix_results.tex`; both `\input` shared macro preamble extracted
   to `docs/creativity_macros.tex`.
2. Move content: current Parts I–V bodies migrate to appendices A–I per the
   table above, wrapped in `appres` environments with `\label{res:A1}`-style
   labels; the Appendix Index table is generated from these labels.
3. Write the 7-page core per §1–§6 above, citing `[\ref{res:...}]` throughout.
4. Keep `bridged_validation.tex` as the full technical report (unchanged);
   the core + appendix become the paper-facing artifact. Both cite the same
   Lean names, so the audit trail (Lean name → ledger row → appendix result →
   core citation) stays intact.
5. Compile check: enforce the page budget with `\usepackage{layouts}` or a CI
   `pdfinfo` check that pages(core before appendices) ≤ 7.

## 4. What is deliberately *not* in the core

- No proofs longer than two sentences (all in appendices).
- No Lean tactic detail (`linarith`, `Finset.sum_comm`, …) — appendix only.
- No estimator calibration constants, no O-information taxonomy, no
  composability/Pareto examples — cited as [F.x]/[I.x].
- Trajectory diagnostics appear only as two one-sentence citations (they
  support the dense reward and the DC bonus; they are not on the critical
  chain).
- The Known Gaps Registry is cited once in §3 (one sentence: "all remaining
  assumptions are centralized in [I.3]").
