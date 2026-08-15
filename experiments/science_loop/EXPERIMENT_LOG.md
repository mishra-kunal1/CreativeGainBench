# Science-in-the-loop — experiment log

**Goal:** Refine CreativeGainBench metrics so human vs LLM (gemma2:2b) outputs show meaningful creativity distinction.

**Started:** 2026-08-13 (local)

**Constraints:** Never fit δ_D to z*; R_B is ungated (BBase); default G_k is G_k_surface; prefer existing E*/V* harnesses.

---

## Run index

| ID | When (UTC) | Kind | Status | Summary |
|----|------------|------|--------|---------|
| E1-Gemma-Rerun | 2026-08-13 | experiment1 gemma2:2b | **done** (R_D+agg) | ΔR_D=+2.27e-05 CI includes 0; gens 1489 |
| EMB-PCA-01 | 2026-08-13 | embedding PCA separability | **done** | CV acc=0.986; silhouette=0.394 (STRONG) |
| DIAG-E4-Baseline | 2026-08-13T17:39Z | E4 check | done | pooled_r=-0.0164 |
| DIAG-E4-Variants | 2026-08-13T17:45Z | H-20260813-01 P1 | done | gemma Spearman rd↔cue≈0.068 |
| H-20260813-01 | 2026-08-13 | hypothesis | **approved** (iter2) | CUE/R_D length-protocol alignment; Stage C after E1 R_D |

---

## E1-Gemma-Rerun — Experiment 1 with `gemma2:2b`

### Intent
Reproduce the poetry_v2 eval-split human-vs-model measurement for the local Ollama model `gemma2:2b`, with current CountNgram ProbeCompressor R_D, length-clipped scoring, and F0 feasibility gate.

### Protocol
- Split: `eval` (author-disjoint)
- Artifacts: `src/creativegainbench/artifacts/poetry_v2`
- Metric version: `poetry_v2`
- Generation: temperature=1.0, top_p=0.9, max_tokens=1024, workers=2, Ollama `http://127.0.0.1:11434/v1`
- Steps: `00_ensure_delta_d` → `01_schema` → `02_generate --model gemma2:2b` → `03_score_rd --model gemma2:2b` → `04_score_cue --model gemma2:2b` → `05_aggregate`
- Skipped for this pass: `03b_score_edge_cue` (MAS; not required for single-agent human vs gemma R_D/CUE)

### Preflight
- Postgres: `poems-pg` started
- Ollama tags include `gemma2:2b`
- Cleared prior rows: `generations` where `model=gemma2:2b` (was 4673 → 0); `scores` where `side=gemma2:2b` (was 1489 → 0)
- Eval poems with prompt: 1489 (expected generation count)

### Commands
```bash
podman start poems-pg
# NOTE: 01_schema remigrates poems.llm_output → generations[gemma2:2b].
# For a true Ollama regen we delete those rows AFTER schema, then run generate.
.venv/bin/python -c "..."  # DELETE generations/scores for gemma2:2b
.venv/bin/python experiments/experiment1/components/02_generate.py --model gemma2:2b \
  2>&1 | tee experiments/experiment1/logs/02_generate_gemma_rerun.log
.venv/bin/python experiments/experiment1/components/03_score_rd.py --model gemma2:2b --device cpu \
  2>&1 | tee experiments/experiment1/logs/03_score_rd_gemma_rerun.log
.venv/bin/python experiments/experiment1/components/04_score_cue.py --model gemma2:2b \
  2>&1 | tee experiments/experiment1/logs/04_score_cue_gemma_rerun.log
.venv/bin/python experiments/experiment1/components/05_aggregate.py \
  2>&1 | tee experiments/experiment1/logs/05_aggregate_gemma_rerun.log
```

### Incident: schema remigration
First `run_all` attempt hit `01_schema`, which re-inserted 4673 `gemma2:2b` rows from `poems.llm_output`, so `02_generate` reported `need 0`. Pipeline killed; generations cleared again; fresh Ollama generate started for 1489 eval prompts.

### Results
_(filled as steps complete)_

### Artifacts
- Log: `experiments/experiment1/logs/run_all_gemma_rerun.log`
- R_D JSONL: `experiments/experiment1/results/rd_gemma2_2b.jsonl`
- CUE JSONL: `experiments/experiment1/results/cue_gemma2_2b.jsonl`
- Aggregate: `experiments/experiment1/results/REPORT.md`, `ladder_summary.json`

---

## Science-in-the-loop crew

### Objective
Propose and test metric refinements that increase meaningful human–LLM creativity separation on gemma2:2b (and stay Lean-honest).

### Workers
- `hypothesis` (scientist)
- `critic` (methodologist)
- `experimenter` (operator)

### Session
- **session_id:** `53fcd43ffa154b42b1282934932a9dc6`
- **Started:** after E1 Ollama regen kickoff (generate in progress)
- Workers: hypothesis / critic / experimenter

### Hypotheses and follow-on runs

---

## H-20260813-01 — CUE/R_D length-protocol alignment (close E4) + paired ΔR_D reporting

Posted by `hypothesis`, iteration 1.

```yaml
id: H-20260813-01
goal: Close E4 (corr(R_D, CUE) into 0.1-0.5) and sharpen human-vs-gemma2:2b R_D delta reporting, with no delta_D refit and no fitting to z*.
observation: >-
  Snapshot 2026-08-13T17:25Z: the single open FAIL is E4 pooled corr(R_D, CUE) = -0.01643 (band 0.1-0.5).
  experiments/experiment1/results/REPORT.md: gemma2:2b mean dR_D = +3.712e-05, 95% CI [-4.08e-05, +1.18e-04] includes 0; CUE delta +4.2e-06 (n=200, no CI).
  Code audit: experiments/experiment1/components/03_score_rd.py applies the F10 length protocol (clip to per-domain eval-median chars, floor 200) before scoring R_D, but components/04_score_cue.py clips only to max_chars — CUE and R_D for the same poem/side are computed on different texts.
  CUE = brier_delta / bit_length_utf8(text) (src/creativegainbench/metrics/cue.py), so CUE variance is dominated by 1/length, which is orthogonal to lambda_D-normalized r_d_norm.
hypothesis: >-
  The E4 failure is a length-protocol artifact, not a construct failure: corr(r_d_norm, cue) ~ 0 because (a) CUE text is unclipped relative to R_D's F10 clip and (b) the 1/bit_length denominator injects length variance orthogonal to R_D. Applying the same per-domain clip in 04_score_cue restores a moderate positive association.
prediction: >-
  P1 (cheap, frozen data): joining existing cue_*.jsonl with rd_*.jsonl, pooled |corr(r_d_norm, brier_delta)| > |corr(r_d_norm, cue)| and lands in [0.05, 0.5], with the same sign under Spearman.
  P2 (gated rescore): after clip alignment, rescored gemma2:2b CUE (frozen cue_ids.json, n=200) gives by_model |Pearson(r_d_norm, cue)| in [0.1, 0.5] in the E4 recompute.
  P3 (reporting): paired per-poem bootstrap of dR_D on feasibility-gated pairs (both sides r_d_gate=1) plus gate-rate delta with CI; reported as observed, even if null.
experiment_plan:
  - command: APPROVAL_REQUIRED (patch) - add read-only script experiments/science_loop/runs/H-20260813-01/analyze_e4_variants.py, then run it with .venv/bin/python
    expect: pooled + per-model Pearson/Spearman for (r_d_norm, cue) and (r_d_norm, brier_delta); P1 pass criterion above
    max_runtime_min: 2
  - command: APPROVAL_REQUIRED (patch + expensive) - surgical edit to 04_score_cue.py applying 03_score_rd's per-domain median clip (record length_clip_chars in JSONL); rescore gemma2:2b only; rerun e4_convergent_discriminant.py
    expect: gemma2:2b by_model |r| in [0.1, 0.5]; gated on E1-Gemma-Rerun 03_score_rd completion
    max_runtime_min: 90
  - command: APPROVAL_REQUIRED (patch) - add experiments/science_loop/runs/H-20260813-01/paired_rd_stats.py (paired bootstrap + Wilcoxon on rd_gemma2_2b.jsonl, all pairs and gated stratum), run it
    expect: dR_D 95% bootstrap CI per stratum + gate-rate delta, written to runs/H-20260813-01/paired_rd_stats.json
    max_runtime_min: 5
risks:
  - Forking paths across correlation variants. Pre-registered decision rule: only P2 (length-aligned CUE, by_model Pearson on gemma2:2b) decides; the P1 brier_delta variant is diagnostic only and is NOT substituted into the E4 check.
  - Pooled E4 mixes stale cue files from the other 3 models; full pooled closure needs their rescore too (separate approval after P2 passes on gemma2:2b).
  - E1-Gemma-Rerun in flight; rescoring against half-regenerated generations would mix cohorts. Step 2 waits for 03_score_rd_gemma_rerun completion.
  - CUE receiver (Ollama belief model) is stochastic; keep frozen cue_ids.json, report n and Fisher-z CI on r.
stop_rule: >-
  Accept if P2 holds (gemma2:2b by_model |r| in [0.1, 0.5] after alignment). Reject if |r| < 0.05 after alignment (length artifact ruled out; next suspect is z* annotation quality, not delta_D). Iterate (max 3) if 0.05 <= |r| < 0.1 or if P1 and P2 disagree in sign. dR_D CIs from P3 are reported as observed, never tuned.
iteration: 1
critic_status: pending
```

---

### Critic review — H-20260813-01, iteration 1 — VERDICT: revise

**Reviewer:** critic (methodologist). **Date:** 2026-08-13. **Ruling:** `revise` — issues 1–4 blocking, 5–7 advisory. Card is close; all three mechanism claims were fact-checked against code and hold.

**Fact-check (performed before ruling, read-only):**
- Stale join confirmed: `experiments/construct_validity/experiments/e4_convergent_discriminant.py` reads `r_d_norm` embedded in `cue_*.jsonl` and computes Pearson only (band 0.1–0.5).
- Clip mismatch confirmed: `04_score_cue.py` scores `text.strip()[:max_chars]` (flat) while `03_score_rd.py` applies the F10 per-domain-median clip.
- Zero-inflation is structural: `brier_delta = max(0, ·)` and `cue = brier_delta / bit_length`, so `cue == 0` iff the posterior is not strictly better than the prior; REPORT.md gate rates (42–47% nonzero) imply >50% zeros.
- REPORT.md gemma2:2b ΔR_D = +3.712e-05, 95% CI [-4.08e-05, +1.18e-04] — matches card.

**Blocking issues:**

1. **P2 primary endpoint under-specified.** Stage C rescores CUE for gemma2:2b only, so P2 is evaluable only on gemma pairs (≤200 ids × 2 sides ≈ 400 pairs). Pre-register exactly: population (all protocol-matched pairs vs the `cue > 0` subset), estimator (Spearman), expected n, and whether "in [0.1, 0.5]" means point estimate or bootstrap CI. With n ≤ 400 and >50% zeros, |rho| = 0.1 carries a wide CI — fix the decision rule now or the accept/reject call will be arguable either way.
2. **P1 is a forking path.** "|rho| ≥ 0.05 pooled OR within-side" across 2 estimators × 3 populations (× per-model blocks) nearly guarantees some pass. Pick one primary quantity (suggest: pooled fresh-join Spearman, all models) and demote everything else to descriptive diagnostics.
3. **Stage C must not rerun recorded E4 as-is.** `e4_convergent_discriminant.py` globs `cue_*.jsonl` in `experiment1/results/`; a rescored gemma file there would pool one protocol-matched file with three stale ones and record a contaminated `experiment_results` row. Write Stage C output under `runs/H-20260813-01/` (outside the glob) and compute the pre-registered correlation with a dedicated script. Any change to the recorded E4 estimator or band is a separate, versioned, critic-approved protocol change made only after results are in.
4. **Length confound uncontrolled — cuts both ways.** `cue = brier_delta / bit_length` and `r_d_norm` is length-normalized, so shared 1/length variance can manufacture (or mask) corr(R_D, CUE) with no construct link. Stage A must additionally report corr(bit_length, cue), corr(length, r_d_norm), and a length-controlled association (rank partial or within-length-band) on the `cue > 0` subset (`bit_length` is recoverable there as `brier_delta / cue`). Amend the stop rule: a band pass that disappears under length control is NOT an H1a accept.

**Advisory:**

5. Also report the gate-level association (rank / point-biserial corr of `r_d_norm` vs `1[cue > 0]`): with >50% zeros most usable signal may live in the gate, and the gate is how CUE actually enters `r_creativity`.
6. `04_score_cue.py` also upserts CUE payloads into `scores` under `metric_version = <metric_version>_cue`; Stage C's new metric_version suffix must apply to those DB rows too, not just the jsonl filename.
7. H1b semantics: if the fresh-generation ΔR_D CI still includes 0, that may be a metric-honest null (gemma2:2b matches human R_D here). Rejecting H1b must never trigger metric changes aimed at forcing separation — that is fitting to the human/model label, the same sin as fitting δ_D to z*. The card's "report honestly" wording is correct; keep it through iterations.

**Confirmed sound:** the Stage A sequencing constraint (run before the rerun's `03_score_rd` overwrites `rd_gemma2_2b.jsonl`, else drop the gemma llm side from the fresh join) is real and correctly handled; Spearman pre-registration from structural inspection (not computed values), with both estimators co-reported, is acceptable; costs are honest (Stages A/B cheap, Stage C correctly APPROVAL_REQUIRED at ~400 receiver calls, ~45 min).
**Lean honesty:** pass — no ungated-R_B claims, `G_k_surface` untouched, `cue_gate = 1[CUE>0]` unchanged, no δ_D fitting anywhere in the plan.

**Requested next step:** hypothesis worker revises the card to `iteration: 2` addressing issues 1–4 (the issue-4 Stage A script amendment is cheap and can ride along); critic will re-review.

---

## DIAG-E4-Baseline — 2026-08-13T17:39:01Z

### Intent
Cheap re-run of `e4_convergent_discriminant.py` on existing frozen CUE JSONLs / scores to lock a baseline before metric changes.

### Command
```bash
.venv/bin/python experiments/construct_validity/experiments/e4_convergent_discriminant.py
```

### Result
- **Status:** FAIL (as expected; confirms snapshot)
- **pooled_r:** -0.016430094945788464
- **n:** 1600
- **by_model:**
  - gemma2:2b: **0.02074**
  - llama3.1:8b: -0.03893
  - mistral:latest: -0.01825
  - phi4:14b: -0.02375
- **band:** [0.1, 0.5]
- **exit:** 0
- **log:** `experiments/science_loop/runs/diag-e4-baseline/e4_baseline.log`

### Interpretation
Even gemma’s within-model corr is ~0.02 — far below 0.1. Supports H-20260813-01 claim that E4 failure is not a fluke of pooling alone.

### Run index update
| DIAG-E4-Baseline | 2026-08-13T17:39:01Z | construct_validity E4 | done | pooled_r=-0.0164 |

---

### Critic audit — runs/H-20260813-01/analyze_e4_variants.py (pre-iteration-2) — 2026-08-13

**Reviewer:** critic (methodologist). Iteration-2 card not yet posted; this is an interim audit of the planner-written diagnostic script.

- **Safe to run now, any time:** read-only over `experiments/experiment1/results/cue_*.jsonl`; joins on the `r_d_norm` values frozen inside those files, so it does not read `rd_*.jsonl` and is immune to the E1-Gemma-Rerun `03_score_rd` overwrite race. Output goes to `runs/H-20260813-01/e4_variants.json` only.
- **Descriptive only — NOT the P1 endpoint:** the script implements the iteration-1 variants (Pearson/Spearman of `r_d_norm` vs `cue` and vs `brier_delta`, pooled + per-model) but omits the issue-4 length controls (corr(`bit_length`, `cue`), corr(`bit_length`, `r_d_norm`), length-controlled association on the `cue > 0` subset, where `bit_length = brier_delta / cue` is recoverable) and the issue-5 gate association (`r_d_norm` vs `1[cue > 0]`). Its numbers must not be cited as a pre-registered P1 pass/fail.
- **Requirement for iteration 2:** extend this script (or add a companion in the same run dir) with the issue-4 columns, and pre-register exactly one primary P1 quantity (issue 2); everything else is demoted to diagnostics.

**Verdict on H-20260813-01 iteration 2:** pending — card not yet in the log.

---

## H-20260813-01 — iteration 2 (revision addressing critic blocking issues 1–4)

Posted by `hypothesis`, 2026-08-13. Supersedes the iteration 1 card above. Critic advisories 5–7 also incorporated. E1-Gemma-Rerun generate is still in flight; no fresh-R_D claims are made here.

```yaml
id: H-20260813-01
goal: Close E4 (corr(R_D, CUE) into 0.1-0.5) and sharpen human-vs-gemma2:2b R_D delta reporting, with no delta_D refit and no fitting to z* or human/model labels.
observation: >-
  Snapshot 2026-08-13T17:25Z: single open FAIL is E4 pooled corr(R_D, CUE) = -0.01643 (band 0.1-0.5).
  DIAG-E4-Baseline: per-model Pearson also ~0 (gemma2:2b +0.02074, other three negative, pooled n=1600), so pooling is not the sole cause.
  Code facts (critic fact-checked): 04_score_cue.py scores text.strip()[:max_chars] (flat clip) while 03_score_rd.py applies the F10 per-domain-median clip, so CUE and R_D score different texts for the same poem/side; cue = brier_delta / bit_length_utf8(text) injects 1/length variance; brier_delta = max(0, .) makes cue zero-inflated (>50% zeros given REPORT.md gate rates of 42-47% nonzero).
  experiments/experiment1/results/REPORT.md (pre-rerun): gemma2:2b mean dR_D = +3.712e-05, 95% CI [-4.08e-05, +1.18e-04] includes 0.
hypothesis: >-
  H1a (E4 is a length-protocol artifact): corr(r_d_norm, cue) ~ 0 mainly because (a) CUE text is clipped differently from R_D's F10 clip and (b) the 1/bit_length denominator injects length variance orthogonal to r_d_norm. Scoring CUE on the identical F10-clipped text restores a moderate positive rank association that survives length control.
  H1b (reporting): fresh-generation paired dR_D with per-poem bootstrap CI is the honest human-vs-gemma2:2b separation statement; a CI including 0 is a metric-honest null and is reported as such.
prediction: >-
  P1 (cheap, frozen data; ONE primary): pooled fresh-join Spearman rho(r_d_norm, brier_delta) across all four models' cue_*.jsonl (both sides, zeros included) is >= +0.05 AND exceeds pooled Spearman rho(r_d_norm, cue). Every other quantity the script emits (Pearson, per-model blocks, cue>0 subset, length diagnostics, gate point-biserial) is descriptive only and decides nothing.
  P2 (gated rescore; PRIMARY ENDPOINT, pre-registered in full):
    population = all gemma2:2b protocol-matched sides from frozen cue_ids.json (200 poems x 2 sides; expected n ~= 400 minus sides lacking a fresh generation or R_D score), zeros INCLUDED;
    estimator = Spearman rho(r_d_norm, cue_len_aligned) computed by a dedicated script (no globbing);
    decision rule = point estimate in [0.1, 0.5] AND 10k-resample nonparametric bootstrap 95% CI excludes 0 AND length control passes: rank partial rho(r_d_norm, cue | bit_length) on the cue>0 subset >= +0.05 with the same sign (bit_length recorded directly in the fresh JSONL, not recovered).
  P3 (reporting): paired per-poem bootstrap (10k resamples) of dR_D = R_D(human) - R_D(gemma2:2b) on fresh rd_gemma2_2b.jsonl; primary population = all pairs with both sides scored; Wilcoxon signed-rank co-reported; feasibility-gated stratum (both sides r_d_gate=1) and gate-rate delta reported descriptively. Reported as observed, even if null.
experiment_plan:
  - command: "Stage A (read-only, may run pre-approval per planner): APPROVAL_REQUIRED patch amending runs/H-20260813-01/analyze_e4_variants.py to add, on the cue>0 subset with bit_length recovered as brier_delta/cue: spearman(bit_length, cue), spearman(bit_length, r_d_norm), rank partial rho(r_d_norm, cue | bit_length), plus pooled point-biserial corr(r_d_norm, 1[cue>0]); then run .venv/bin/python experiments/science_loop/runs/H-20260813-01/analyze_e4_variants.py"
    expect: P1 primary criterion; length diagnostics logged to inform Stage C interpretation
    max_runtime_min: 2
  - command: "Stage B (after E1-Gemma-Rerun 03_score_rd completes): APPROVAL_REQUIRED patch adding runs/H-20260813-01/paired_rd_stats.py implementing P3; run with .venv/bin/python; outputs runs/H-20260813-01/paired_rd_stats.json"
    expect: dR_D bootstrap CI (all-pairs primary; gated stratum descriptive) + gate-rate delta
    max_runtime_min: 5
  - command: "Stage C (gated, expensive, ~400 receiver calls ~45 min; after Stage B prerequisites): APPROVAL_REQUIRED patch adding runs/H-20260813-01/score_cue_lenaligned.py - standalone CUE scorer reusing 04_score_cue's receiver but applying 03_score_rd's F10 per-domain-median clip, recording bit_length and length_clip_chars per side, writing ONLY runs/H-20260813-01/cue_gemma2_2b_lenaligned.jsonl (no writes under experiments/experiment1/results/, NO scores-table upserts); then runs/H-20260813-01/compute_p2.py evaluates the pre-registered P2 decision rule into p2_result.json"
    expect: P2 decision rule evaluated; recorded E4 check and experiment_results rows untouched
    max_runtime_min: 90
risks:
  - Forking paths: exactly one primary per prediction as pre-registered above; all other outputs descriptive. No variant (e.g. brier_delta) is ever substituted into the recorded E4 check.
  - Contamination: Stage C never writes under experiments/experiment1/results/ and never upserts to the scores table, so e4_convergent_discriminant.py's cue_*.jsonl glob and recorded experiment_results stay clean. Any change to the recorded E4 estimator/band is a separate, versioned, critic-approved protocol change made only after results are in.
  - Shared 1/length variance can manufacture or mask the association; hence the length-control clause is INSIDE the P2 decision rule (a band pass that dies under length control is not an accept).
  - E1-Gemma-Rerun in flight: Stages B/C wait for 03_score_rd_gemma_rerun completion; Stage A joins only fields embedded in frozen cue_*.jsonl, so it is immune to rd_gemma2_2b.jsonl being overwritten.
  - CUE receiver stochasticity: frozen cue_ids.json; n reported; bootstrap CI used (not Fisher z, given zero-inflation).
stop_rule: >-
  Accept H1a iff the full P2 decision rule passes (point Spearman in [0.1, 0.5] AND bootstrap 95% CI excludes 0 AND length-controlled partial rho >= +0.05 with same sign).
  Reject H1a if point Spearman < 0.05, or negative, or a band pass disappears under length control (partial rho < +0.02 or sign flip); the next suspect is then z* annotation quality, not delta_D.
  Iterate (max 3 total) if 0.05 <= point < 0.1, or point is in band but the bootstrap CI includes 0.
  H1b/P3: the dR_D CI is reported as observed; a CI including 0 is accepted as an honest null and never triggers metric changes aimed at forcing human-model separation.
iteration: 2
critic_status: pending
```

### Changelog vs iteration 1 (mapped to critic issues)

- Issue 1 (P2 under-specified): population, estimator, expected n, and a three-clause decision rule (point-in-band + bootstrap CI excluding 0 + length control) are now pre-registered inside P2.
- Issue 2 (P1 forking paths): P1 reduced to a single primary comparison on the pooled fresh join (Spearman, zeros included); all other estimator/population combinations demoted to descriptive diagnostics.
- Issue 3 (Stage C contamination): all Stage C outputs live under runs/H-20260813-01/ (outside the E4 glob), scored by a standalone script with a dedicated P2 evaluator; no scores-table upserts at all, which also resolves advisory 6; the recorded E4 check is never rerun on mixed files.
- Issue 4 (length confound): bit_length is recorded directly in the fresh JSONL; Stage A gains length diagnostics (corr(bit_length, cue), corr(bit_length, r_d_norm), rank partial); the length-control clause is part of the P2 accept rule, and the stop rule explicitly refuses a band pass that dies under length control.
- Advisory 5: pooled point-biserial corr(r_d_norm, 1[cue>0]) added to Stage A diagnostics.
- Advisory 7: honest-null wording for H1b retained and strengthened in the stop rule.

---

## DIAG-E4-Variants — 2026-08-13T17:45Z

### Intent
H-20260813-01 Stage A / P1 diagnostic: Pearson & Spearman of `r_d_norm` vs `cue` and vs length-free `brier_delta` on frozen `cue_*.jsonl` (embedded `r_d_norm`). Read-only.

### Command
```bash
.venv/bin/python experiments/science_loop/runs/H-20260813-01/analyze_e4_variants.py
```

### Results (`e4_variants.json`)

| Slice | n | Pearson rd↔cue | Pearson rd↔brier_delta | Spearman rd↔cue | Spearman rd↔brier_delta |
|-------|---|----------------|------------------------|-----------------|-------------------------|
| pooled | 1600 | -0.0164 | -0.0016 | 0.0115 | 0.0202 |
| gemma2_2b | 400 | 0.0207 | 0.0224 | 0.0684 | 0.0677 |
| llama3.1_8b | 400 | -0.0389 | -0.0191 | -0.0043 | 0.0121 |
| mistral_latest | 400 | -0.0183 | 0.0030 | -0.0372 | -0.0350 |
| phi4_14b | 400 | -0.0237 | -0.0084 | 0.0186 | 0.0344 |

### Interpretation
- Confirms E4 baseline Pearson pool ≈ -0.016.
- Switching to `brier_delta` (drop `/ bit_length`) barely moves gemma Spearman (~0.068) — still below 0.1. So **length-in-denominator alone does not explain** the E4 miss; clip mismatch / stale join / construct gap remain live.
- Artifact: `experiments/science_loop/runs/H-20260813-01/e4_variants.json`

---

### Critic review — H-20260813-01, iteration 2 — VERDICT: approved

**Reviewer:** critic (methodologist). **Date:** 2026-08-13. **Ruling:** `approved` — blocking issues 1–4 from the iteration 1 review are resolved. Advisories A–C below are non-blocking and do not require an iteration 3.

**Blocking-issue resolution check:**

1. **Resolved.** P2 is fully pre-registered: population = frozen `cue_ids.json` gemma2:2b sides (200 poems x 2 sides, zeros included, expected n ~= 400); estimator = Spearman computed by a dedicated script (no globbing); decision rule = point estimate in [0.1, 0.5] AND 10k-resample bootstrap 95% CI excludes 0 AND rank partial rho(r_d_norm, cue | bit_length) >= +0.05 with the same sign. No arguable calls remain on the accept branch.
2. **Resolved.** P1 collapsed to a single primary pooled Spearman comparison; every other estimator/population combination is explicitly descriptive and decides nothing.
3. **Resolved.** Stage C writes only under `runs/H-20260813-01/` (outside the E4 `cue_*.jsonl` glob), performs no `scores`-table upserts (which also closes advisory 6), and P2 is evaluated by a dedicated `compute_p2.py`. The recorded E4 check and `experiment_results` rows stay untouched; any estimator/band change remains a separate versioned protocol change.
4. **Resolved.** `bit_length` is recorded directly in the fresh JSONL (not recovered); Stage A gains the requested length diagnostics (corr(bit_length, cue), corr(bit_length, r_d_norm), rank partial on the cue>0 subset); the length-control clause sits inside the P2 accept rule, and the stop rule explicitly refuses a band pass that dies under length control. Advisory 5 (gate point-biserial) and advisory 7 (honest-null wording) are incorporated.

**New advisories (non-blocking):**

- **A. P1 is already numerically refuted by DIAG-E4-Variants.** On the identical population (frozen `cue_*.jsonl`, pooled, zeros included) pooled Spearman rho(r_d_norm, brier_delta) = 0.0202 < +0.05. This is honest, not tuned — but it means mechanism (b) (the 1/bit_length denominator) is already weakened, and H1a's weight now rests on clip alignment (a), which only Stage C / P2 tests. The card should state explicitly whether a P1 fail gates Stage C spend; recommendation: it does not auto-abort (Stage C is the only test of mechanism (a)); the planner gate decides.
- **B. Stop-rule gap.** Point-in-band + bootstrap CI excluding 0 + partial rho in [+0.02, +0.05) matches neither the accept clause (needs >= +0.05) nor the reject clause (needs < +0.02 or sign flip). Pre-assign this outcome to the iterate branch now so it cannot be argued later.
- **C. "Fresh-join" wording.** P1 says "pooled fresh-join" but the mechanics (per the card's own risks section and the audited Stage A script) join on `r_d_norm` embedded in frozen `cue_*.jsonl`. Call it a frozen-embedded join; a genuinely fresh join against `rd_*.jsonl` would race the E1-Gemma-Rerun `03_score_rd` overwrite.

**Lean honesty:** pass — no delta_D refit, no fitting to z* or to human/model labels, R_B stays ungated (BBase), `G_k_surface` default untouched, `cue_gate = 1[CUE>0]` semantics unchanged.

**Cost:** honest — Stages A/B are minutes; Stage C ~400 receiver calls (~45–90 min), correctly APPROVAL_REQUIRED and gated on `03_score_rd_gemma_rerun` completion.

**Go/no-go:** Stage A amendment may proceed now (read-only, immune to the rd overwrite race). Stage B after E1-Gemma-Rerun `03_score_rd` completes. Stage C gated as planned. `critic_status: approved` for iteration 2; advisories A–C to be noted in run artifacts or the next card edit.

---
## E1-Gemma-Rerun — generate complete
### Finished generate: 2026-08-13T18:50:59Z
- Result: ok=1489 fail=0
- Log: experiments/experiment1/logs/02_generate_gemma_rerun.log
- Starting 03_score_rd (human+gemma) and EMB-PCA-01 in parallel.


---
## E1-Gemma-Rerun — generate complete
### Finished generate: 2026-08-13T18:51:01Z
Starting 03_score_rd (human+gemma). Stage B CUE clip patch + 04_score_cue deferred until after R_D (H-20260813-01).

### 03_score_rd finished
end 2026-08-13T19:58:48Z
Artifacts: experiments/experiment1/results/rd_gemma2_2b.jsonl
Next: Stage B patch 04_score_cue length-clip → isolated Stage C under runs/H-20260813-01/
### 03_score_rd finished
end 2026-08-13T20:00:30Z
- human + gemma2:2b scored n=1489 each
- Artifacts: experiments/experiment1/results/rd_gemma2_2b.jsonl, rd_human.json


---

## E1-Gemma-Rerun — R_D + aggregate results (2026-08-13T20:00Z)

### Generate
- **ok=1489 fail=0** via Ollama `gemma2:2b` (fresh regen; not `poems.llm_output` migration)
- Log: `experiments/experiment1/logs/02_generate_gemma_rerun.log`

### R_D (`03_score_rd --model gemma2:2b`)
- Human + gemma scored with F10 domain median clips; feasibility gate on
- JSONL: `experiments/experiment1/results/rd_gemma2_2b.jsonl`
- Log: `experiments/experiment1/logs/03_score_rd_gemma_rerun.log`

### Aggregate (`05_aggregate`) — gemma2:2b primary
| Metric | Human | gemma2:2b | Δ (H−M) | 95% CI |
|--------|------:|----------:|--------:|--------|
| mean R_D | 2.868e-04 | 2.641e-04 | **+2.268e-05** | [-3.80e-05, +8.68e-05] **includes 0** |
| wins | 690 | 799 | — | model wins more pairs |
| tail ≥q90 | 10.7% | 9.3% | — | |

CUE columns in REPORT.md still reflect **pre-rerun** cue JSONLs (Stage B/C not run yet).

### Interpretation for crew
Fresh gemma generations still **do not** yield a significant mean R_D gap (CI includes 0). Distinction problem is not fixed by regen alone.

---

## EMB-PCA-01 — Idea-embedding PCA / class separability

### Intent
Test whether the **measurement-stack embedding** (MiniLM + poetry boundary → mean-pooled idea vectors), under the same F10 length clips as R_D, encodes enough information to separate human vs `gemma2:2b` texts. If yes but R_D/CUE fail, the gap is **metric design**, not representation collapse.

### Protocol
- Population: eval split, paired `(body, generations[gemma2:2b])`
- Sample: **stratified n=400** by domain (seed=42); full 1489 available for follow-up
- Clip: per-domain median chars (F10), same formula as `03_score_rd`
- Embedding: `mean_pool_idea_embeddings` + `poetry_line_splitter` + frozen `idea_boundary.pt`
- Analysis: SVD PCA (3 comps); 5-fold stratified logistic regression CV; silhouette on PC1–2; centroid distance / separation ratio; paired cosine and ‖Δ‖₂
- **No δ_D fit; diagnostic only**

### Command
```bash
.venv/bin/python experiments/science_loop/runs/EMB-PCA-01/analyze_embedding_pca.py \
  --model gemma2:2b --device cpu --stratified-limit 400
```

### Results (2026-08-13T20:01Z)
| Metric | Value |
|--------|------:|
| Linear probe CV acc (full 384-d) | **0.986 ± 0.009** |
| Linear probe CV acc (PCA-3) | **0.962 ± 0.009** |
| Chance | 0.50 |
| Silhouette (PCA-2) | 0.394 |
| Centroid separation ratio (full) | 0.674 |
| Centroid separation ratio (PCA-2) | 2.04 |
| Mean paired cosine(human, llm) | 0.681 |
| PC1–3 variance | 13.5% + 6.8% + 5.2% = 25.5% |

**Verdict: STRONG** — embeddings carry substantial human-vs-LLM class information.

### Artifacts (crew should loop here)
| Path | Contents |
|------|----------|
| `experiments/science_loop/embedding_pca_latest.md` | **Stable pointer** — latest PCA report |
| `experiments/science_loop/runs/EMB-PCA-01/REPORT.md` | Human-readable report |
| `experiments/science_loop/runs/EMB-PCA-01/pca_report.json` | Machine-readable metrics |
| `experiments/science_loop/runs/EMB-PCA-01/pca_coords.jsonl` | PC1–3 per example |
| `experiments/science_loop/runs/EMB-PCA-01/pca_scatter.png` | PC1/PC2 scatter |
| `experiments/science_loop/runs/EMB-PCA-01/analyze_embedding_pca.py` | Repro script |
| `experiments/science_loop/runs/EMB-PCA-01/run.log` | stdout |

### Implication for H-20260813-01 / metric refinement
Class signal **exists** in the idea-embedding space (CV≈99%), while R_D Δ CI includes 0 and E4 corr≈0. So creativity metrics are **not extracting** the available human/LLM geometry. Prefer hypotheses that:
1. project or gate using embedding-space class direction (without fitting δ_D to z*), or
2. change the R_D/CUE functional (clip alignment Stage B/C still on deck),  
rather than swapping MiniLM for a larger encoder as the first move.

