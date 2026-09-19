# Falsifiability suite — E7 causal contribution controls & E5 receiver/encoder stability

These experiment IDs are **not** construct-validity E4/E5. Do not extend or
rerun `experiments/construct_validity/experiments/e4_*.py` / `e5_*.py`.

**Isolation:** never write (or read as this suite's input glob)
`experiments/experiment1/results/cue_*.jsonl`. All outputs live under
`experiments/falsifiability/`.

This protocol is frozen **before** live Ollama scoring.

## Shared design

| Knob | Frozen value |
|------|----------------|
| Seed | **42** |
| Phase A n | **≈ 80** (or `--limit` for smoke). Stratify by `domain_cluster` when poems exist. |
| Receivers | weak `gemma2:2b`, mid `llama3.1:8b`, strong `phi4:14b` (Ollama OpenAI-compat) |
| CUE temperature | **0** |
| Prior cache | one prior per `(receiver, prompt)` — prior does not depend on y |
| `--base-url` | local `http://127.0.0.1:11434/v1` or Ollama Cloud `https://ollama.com/v1` |
| Length-match | all control y's to `|y_matched|` chars **±20%** (F10 `length_match`) |
| Workers | 2 |

### Critical z* rule

Freeze z* from **y_matched** only:

- Phase A: MiniLM `OutcomeAnnotator` (poetry `EXEMPLARS`)
- Phase B: domain-specific exemplars via `OutcomeAnnotator(domain=...)` — poetry centroids are **not** overwritten

Apply the **same** z* to all four y arms:

```text
compute_cue_for_output(..., external_outcome_index=z_star_matched, prior=cached_prior)
```

If z* were computed per-y, random junk labeled `low_quality` with a confident
posterior would inflate CUE.

**Primary DV:** CUE. **Co-primary diagnostic:** raw `brier_delta` (length-free)
with the same paired tests.

## E7 — Causal contribution controls

Item-yoked y arms:

1. `matched` — true contribution for prompt q
2. `cross` — y from a different prompt, **same domain**, fixed permutation, no replacement
3. `random` — length-matched token-shuffled / filler with no task structure
4. `irrelevant` — length-matched, domain-plausible but wrong
   - Phase A: fluent verse from a different topic
   - Phase B math: e.g. “apply the triangle inequality” on an induction item

Primary measure: **PairedMeanDiff** (not unpaired Hedges g).

- pairing = PAIRED, geometry = SIGNED_DIFFERENCE
- statistic = mean(model − human)
- human = CUE_control, model = CUE_matched → positive = matched higher
- default_margin = 0.0 (CI excluding 0 ⇒ DIFFERENT)
- Three planned contrasts {cross, random, irrelevant}; multiplicity =
  existing `benjamini_yekutieli`

Hedges g is an **optional unpaired diagnostic only**.

Official E7 judge: mid receiver `llama3.1:8b` (other receivers reported as
sensitivity). Smoke panels with a single receiver use that receiver.

### E7 pass

For each control in {cross, random, irrelevant}:

- PairedMeanDiff on (CUE_matched − CUE_control) is **DIFFERENT** with **positive**
  estimate after BY adjustment (`p_adj_by` < α)
- Same paired tests on raw `brier_delta` are reported (co-primary diagnostic;
  not a substitute for the CUE gate)

Length: mean `|bits_arm / bits_matched − 1| < 0.20` over the three controls.

## E5 — Receiver + encoder stability

### E5a Receiver

CUE (matched arm) on the three frozen receivers. For each receiver pair:

- **SpearmanRho** (tie-aware average ranks, AGREEMENT, γ = 0.80) **and**
- **ConcordanceCC** (co-primary)

Pass = BCa CI **lower** bound > 0.80 for **both** measures (verdict EQUIVALENT),
not a point estimate.

Optional: Krippendorff α on the 3-receiver matrix with γ = 0.80.

### E5b Encoder

Keep H, codebook, MiniLM, δ_D **frozen**. Rebuild **probe encodings only**
(original / plain / technical). Per item

```text
CV = sd({R_D_norm variants}) / |mean|
```

Pass if BCa CI **upper** of mean CV (item-resampled via `Resampler.bca_ci_rows`)
is **< 0.15**.

Negative control: scrambled probes must **not** pass that bar. If scrambled
also has CI upper < 0.15, E5b fails (control did not discriminate).

Plain/technical banks are **deterministic register-shift templates** (no LLM).

## Phase B (scaffolding; not a live run in this change)

- Panel from `data/subset/eval_all_domains.jsonl` when present
  (`scientific_proposal`, `creative_writing`, `mathematical_proof`)
- Domain z* exemplars in `outcome_annotator.py`
- Math validity **secondary** path: `OllamaReceiverAgent.condition(q, context=y)`
  plus a verifier-style validity bit. **LLM proxy, no Lean 4 oracle.**

## Stats engine

All CIs: `Resampler` BCa. Do not invent a parallel bootstrap.

## Smoke vs live

Constructors and scorers accept `--limit`. Unit tests use `--synthetic` JSONL
and must pass **without** GPU, Ollama, or Postgres. Live Phase A Ollama scoring
is a follow-up, not part of landing this suite.
