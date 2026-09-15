# OLLAMA-CLOUD-02 — CUE meter + panel fix (free models)

Live re-score after making CUE auditable and replacing the synthetic same-template panel. **PROTOCOL.md was not edited.** γ=0.80 and the E7 cross control stay frozen.

This is still a **quota-bounded n=24** look, not the pre-registered PROTOCOL n=80.

## Why this run exists

CLOUD-01 did **not** show that free models cannot measure creativity. It stacked three bugs:

1. **Meter:** parse misses returned silent uniform → Brier Δ = 0 (gpt-oss 96/96 CUE exactly 0; priors/posteriors dropped from JSONL).
2. **Panel:** synthetic volta lyrics made MiniLM z\* constant (`novel_structure` on all 288 rows) and made `cross` another poem of the same template.
3. **E5b:** `--synthetic` 8-probe bank + identical y’s → one CV for every item.

## Run metadata

| Knob | CLOUD-01 | CLOUD-02 |
|------|----------|----------|
| Run ID | `OLLAMA-CLOUD-01` | `OLLAMA-CLOUD-02` |
| Date (UTC) | 2026-08-27 | 2026-08-27 |
| Panel | Phase A `--synthetic` lyrics | held-out writing/science/math (`heldout_prompts_v1.json`) |
| Matched y | same-template poems | mixed-quality OutcomeAnnotator banks (6 each of novel_structure / fluent_paraphrase / clear_utility / low_quality) |
| Cross | within-cluster lyric | **across-domain** derangement (math y on a writing prompt, etc.) |
| n | 24, seed 42 | 24, seed 42 |
| z\* | MiniLM; **all 288** `novel_structure` | MiniLM; **4 labels × 72 rows** (0,1,2,3) |
| Probes | `SYNTHETIC_PROBES` (n=8) | `poetry_v2_ctx` (n=205) |
| CUE triples | 288/288, 0 errors | **288/288**, 0 HTTP errors, **1 parse_fail** (counted, not CUE=0) |
| Isolation | no `experiment1/results/cue_*.jsonl` | same |

Models (unchanged free-tier band): `gpt-oss:20b`, `gemma4:31b` (E7 judge), `nemotron-3-nano:30b`. Base URL `https://ollama.com/v1`. Encoder MiniLM CPU. Workers 2.

Meter smoke (`meter_smoke.json`, n=2): Gemma and Nemotron parsed JSON with non-uniform posteriors; gpt-oss also parsed and updated (not inert).

## Meter audit

| Check | Result |
|-------|--------|
| Silent uniform CUE zeros | **none** (0 rows with parse_ok and prior==posterior at CUE=0) |
| Parse failures | **1/288** — `nemotron-3-nano:30b` / `heldout-scientific_proposal-6` / `irrelevant` (`missing_novel_structure`); `cue=null` |
| gpt-oss inert? | **no** — matched-arm CUE variance > 0; 8 unique values |
| Vectors persisted | `prior`, `posterior`, `parse_ok_*`, `raw_preview_*`, `brier_delta_signed` |

## Per-receiver mean CUE (official clipped DV)

| Receiver | matched | cross | random | irrelevant |
|----------|--------:|------:|-------:|-----------:|
| `gpt-oss:20b` | 8.16e-4 | 6.97e-4 | 7.22e-4 | 6.07e-4 |
| `gemma4:31b` | 7.80e-4 | 5.80e-4 | 6.02e-4 | 5.02e-4 |
| `nemotron-3-nano:30b` | 8.65e-4 | 5.08e-4 | 6.23e-4 | 4.41e-4 |

## Mean Brier Δ (clipped, co-primary) vs signed diagnostic

Clipped Δ is the official CUE numerator. Signed Δ is unclipped (can be negative).

| Receiver | arm | clipped Δ | signed Δ |
|----------|-----|----------:|---------:|
| `gemma4:31b` | matched | 0.415 | **+0.144** |
| | cross | 0.292 | **−0.491** |
| | random | 0.292 | **−0.554** |
| | irrelevant | 0.290 | **−0.582** |
| `gpt-oss:20b` | matched | 0.403 | −0.297 |
| | cross | 0.351 | −0.384 |
| | random | 0.351 | −0.396 |
| | irrelevant | 0.351 | −0.403 |
| `nemotron-3-nano:30b` | matched | 0.466 | **+0.205** |
| | cross | 0.254 | −0.312 |
| | random | 0.306 | −0.261 |
| | irrelevant | 0.249 | −0.405 |

CLOUD-01 Gemma matched clipped Δ was 0.61 vs cross 0.59 (same-template poems). Here **cross is a real control**: Gemma signed Δ is positive only on matched y, negative on cross/random/irrelevant.

## E7 (primary = `gemma4:31b`)

**Passed: False.** Length gate **passed** (mean |bits ratio − 1| = 0.109 < 0.20).

| Control | n | ΔCUE (matched − control) | BCa 95% CI | p_adj BY | verdict | pass |
|---------|---|--------------------------|------------|----------|---------|------|
| cross | 24 | +2.00e-4 | [+5.3e-5, +4.4e-4] | 0.093 | different | no (BY) |
| random | 24 | +1.78e-4 | [+5.1e-5, +4.4e-4] | 0.119 | indeterminate | no |
| irrelevant | 24 | +2.78e-4 | [+1.5e-4, +5.1e-4] | 0.025 | different | **yes** |

CLOUD-01 replication: random and irrelevant **passed** on the synthetic lyric panel. On this held-out panel, **irrelevant still passes**; random does not (clipped CUE still rises a little on shuffled text). Cross is no longer indeterminate-because-same-template: the point estimate is positive and the unadjusted test is `different`, but BY-adjusted p does not clear α=0.05. That is an informative fail, not a meter bug. PROTOCOL bars were not moved.

Signed Brier (not the official DV) separates matched from all three controls more cleanly than clipped CUE.

## E5a receiver rank agreement (matched-arm CUE)

**Passed: False** against frozen γ=0.80. **No `cue_inert` receivers** (gpt-oss is no longer a constant-zero column). Spearman is computed on all three.

| Pair | Spearman ρ (CI) | CCC (CI) | pass |
|------|-----------------|----------|------|
| gemma4:31b vs gpt-oss:20b | 0.880 [0.654, 0.977] | 0.904 [0.678, 0.965] | no (CI lower ≮ 0.80) |
| gemma4:31b vs nemotron-3-nano:30b | 0.767 [0.382, 0.902] | 0.916 [0.729, 0.971] | no |
| gpt-oss:20b vs nemotron-3-nano:30b | 0.789 [0.581, 0.916] | 0.865 [0.622, 0.945] | no |

Krippendorff α = 0.895 (CI lower 0.748). Point agreement is high; n=24 CIs do not clear the PROTOCOL 0.80 bar. CLOUD-01 had ρ=nan because gpt-oss was constant 0.

## E5b encoder R_D (`poetry_v2_ctx`)

**Passed: False.** Mean CV original/plain/technical = **3.93** (CI [2.51, 6.42], upper ≮ 0.15). Scrambled negative control mean CV = 0.89 (must NOT pass; **control_ok**). **10 unique item CVs** (CLOUD-01 had one CV for every synthetic item). The high CV remains after dropping the toy 8-probe bank, so this is a real encoder finding on frozen poetry_v2 probes, not a panel artifact.

## Protocol deviations / frozen bars

- Same free models as CLOUD-01; n=24 not PROTOCOL n=80.
- Did not raise γ=0.80 or drop the cross control to chase a pass.
- Official CUE still uses clipped Brier Δ; signed Δ is diagnostic only.
- One parse failure is missing (`cue=null`), not a silent 0.
