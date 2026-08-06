# Creativity Stats — Statistical Contract

This document is the contract for `creativity_stats.py`: how samples, measures,
resampling, multiplicity, and verdicts compose. Implementation is numpy-only.

## Data: `Sample`

Canonical input for one `(metric, framework, model-vs-human)` comparison:

| Field | Shape / type | Required | Role |
|-------|----------------|----------|------|
| `human` | `(n_h,)` float | yes | Human-arm scores (also calibration observations) |
| `model` | `(n_m,)` float | yes | Model-arm scores |
| `item_ids` | `(n,)` | for paired judge analyses | Shared item keys; equal `len(human)==len(model)` |

`Sample.require_paired()` enforces equal `human`/`model` lengths, non-`None`
`item_ids`, and `len(item_ids) == len(human)`. Direct `StatisticalMeasure.evaluate()`
calls for `PAIRED` measures raise if that contract is violated; the pipeline also
skips paired measures when `item_ids` is absent.
| `rater_matrix` | `(n_items, n_raters)` | optional | Human multi-rater matrix; **NaNs allowed** |
| `predictive` | `(n_items, n_draws)` | optional | Predictive draws **per item** (row-aligned with `human`) |

**Predictive convention:** row `i` is an iid sample from the model's predictive
distribution for item `i`. Observations for CRPS / PIT are `human[i]`.

Optional fields default to `None` and do not break the existing API.

## Pairing

| Value | Meaning | Resampling |
|-------|---------|------------|
| `UNPAIRED` | Generator: two independent populations of artifacts | Resample each arm independently |
| `PAIRED` | Judge: `human[i]` and `model[i]` score the same item | Resample shared item indices |

Reliability and calibration measures resample **items** (rows of
`rater_matrix` or packed `[human | predictive]`), not the two score arms.

## Geometry

Geometry selects the equivalence / adequacy rule on the measure's own scale:

| Geometry | Scale | Match / adequate when |
|----------|-------|------------------------|
| `SIGNED_DIFFERENCE` | signed, null at 0 | CI ⊂ `(-d, +d)` |
| `NONNEG_DISTANCE` | ≥ 0, lower better | CI upper bound `< d` |
| `AGREEMENT` | high good | CI lower bound `> γ` (`margin` = γ) |

## Verdict

Four-state, equivalence-first outcome from CI + (when applicable) a permutation p:

| Verdict | Meaning |
|---------|---------|
| `EQUIVALENT` | Negligible difference / adequate agreement; not “practically different” |
| `DIFFERENT` | Provably non-negligible / inadequate |
| `TRIVIALLY_DIFFERENT` | Detectable (p < α) but still inside the equivalence band |
| `INDETERMINATE` | Underpowered: cannot conclude either way |

Agreement measures omit the difference-test p-value; adequacy is CI-only.

## Resampler

Single engine for every measure:

1. **BCa bootstrap CI** (`n_boot`, `ci_level`) — bias-corrected and accelerated,
   with jackknife acceleration. Paired / unpaired / row-bootstrap variants share
   the same BCa transform. Non-finite bootstrap/jackknife replicates are
   dropped before `z0` / acceleration / quantiles (avoids all-NaN CIs when a
   few resamples are invalid, e.g. sparse reliability matrices).
2. **Permutation null** (`n_perm`) — two-sided for signed differences; one-sided
   (upper) for non-negative distances. Calibration uses shuffle-observations
   against fixed predictive rows.

Scalar energy distance and CRPS pairwise absolute terms use **O(n log n)**
sorted closed forms (no full pairwise matrices).

Defaults live in `config.yaml` (`n_boot`, `n_perm`, `ci_level`, `alpha`).
`ComparisonPipeline(..., alpha=)` propagates α to every measure it owns.

## Multiplicity

Across difference-tests for one metric run, raw p-values are adjusted with
**Benjamini–Yekutieli** FDR (`correction: by`), valid under arbitrary dependence
among measures. Adjusted values appear as `p_adj_by` in `MeasureResult.extra`
/ report rows. Agreement-only rows keep `p_adj_by: null`.

## Margin philosophy

- Every measure exposes `default_margin(sample)` on **its own scale**.
- Defaults are conventions (e.g. Cohen-small, 0.80 agreement) — **override per
  benchmark** with substantive justification (`config.yaml` / constructor
  `margin=`).
- Do not treat defaults as scientific claims about creativity.

## Reliability gate

Before model comparison on human-judged metrics, check inter-rater reliability
when `rater_matrix` is present:

- **Measure:** Krippendorff's α (`krippendorff_alpha`), levels
  `nominal` | `ordinal` | `interval` (default interval for continuous scores).
- **Geometry:** `AGREEMENT` with threshold γ (default 0.80).
- **Arm usage:** `rater_matrix` only; `human` / `model` ignored.
- **Pipeline:** runs iff `rater_matrix is not None`.
- **Note:** ICC(2,k) is a common continuous alternative; not implemented
  (numpy-only α covers missing ratings and mixed levels).

## Calibration extensions

When `predictive` is present and `predictive.shape[0] == len(human)`:

| Measure | Estimate | Geometry |
|---------|----------|----------|
| `crps` | Mean CRPS of draws vs `human` | `NONNEG_DISTANCE` |
| `pit_uniformity` | KS distance of PIT values to Uniform[0,1] | `NONNEG_DISTANCE` |

PIT extras: `pit_histogram`, `pit_bin_edges`, `pit_mad` (mean |ECDF − sorted PIT|).

## Pipeline applicability (summary)

1. `requires_rater_matrix` → run only if `rater_matrix` present.
2. `requires_predictive` → run only if `predictive` present and row-aligned.
3. Else: existing rule — skip `PAIRED` measures unless `item_ids` present and
   lengths match.
4. Unpaired / paired human–model behavior is unchanged when extensions are absent.

## Report shape

`ComparisonReport.rows()` / `report_to_dict()` emit per-(metric × measure)
records validated by `report.schema.json`. Optional `model` labels the
model arm in multi-model dumps.
