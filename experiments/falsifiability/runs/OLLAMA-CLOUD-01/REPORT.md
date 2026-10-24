# OLLAMA-CLOUD-01 — E5/E7 falsifiability (Ollama Cloud)

Live execution of the frozen E5/E7 protocol on Ollama Cloud. **PROTOCOL.md was not edited.**

This is a **quota-bounded first look at n=24**, not the pre-registered PROTOCOL n=80. Local PROTOCOL tags (`gemma2:2b`, `llama3.1:8b`, `phi4:14b`) are not on Ollama Cloud. Paid Cloud tags (`glm-5.1`, `qwen3.5:397b`) returned **402** on this account. CUE was scored on the **free-tier** band that actually completed chat: `gpt-oss:20b`, `gemma4:31b`, `nemotron-3-nano:30b`.

## Run metadata

| Knob | Value |
|------|--------|
| Run ID | `OLLAMA-CLOUD-01` |
| Date (UTC) | 2026-08-27 |
| Phase | A, `--synthetic` (no Postgres) |
| n | **24** (8/8/8 on domain clusters 0,1,2), seed **42** |
| CUE temperature | 0 |
| Workers | 2 |
| Encoder / device | MiniLM, CPU |
| Base URL | `https://ollama.com/v1` |
| CUE triples | **288 / 288** (24 × 3 receivers × 4 arms), 0 errors |
| Isolation | did **not** write `experiments/experiment1/results/cue_*.jsonl` |

## Models actually scored

| Band | PROTOCOL (local) | Planned Cloud (402) | Scored (this account) |
|------|------------------|---------------------|------------------------|
| Weak | `gemma2:2b` | `gemma4:31b` | `gpt-oss:20b` |
| Mid / Gemma (E7 judge) | `llama3.1:8b` | `glm-5.1` | `gemma4:31b` |
| Strong | `phi4:14b` | `qwen3.5:397b` | `nemotron-3-nano:30b` |

Smoke: `gemma4:31b` chat **200** after the key was set. `glm-5.1` / `qwen3.5:397b` / `kimi-k2.6` / `deepseek-v4-flash` → **402** subscription/extra usage.

## Per-receiver mean CUE (primary DV)

CUE is Brier-Δ / UTF-8 bits, so values are small. **z\* frozen from `y_matched`** (MiniLM `OutcomeAnnotator`); all 288 rows labeled `novel_structure`.

| Receiver | matched | cross | random | irrelevant |
|----------|--------:|------:|-------:|-----------:|
| `gpt-oss:20b` | 0 | 0 | 0 | 0 |
| `gemma4:31b` | 6.84e-4 | 6.63e-4 | 1.00e-4 | 0 |
| `nemotron-3-nano:30b` | 5.04e-4 | 3.04e-4 | 0 | 5.57e-5 |

## Per-receiver mean Brier Δ (length-free co-primary)

| Receiver | matched | cross | random | irrelevant |
|----------|--------:|------:|-------:|-----------:|
| `gpt-oss:20b` | 0.000 | 0.000 | 0.000 | 0.000 |
| `gemma4:31b` | **0.605** | 0.587 | 0.089 | 0.000 |
| `nemotron-3-nano:30b` | **0.445** | 0.269 | 0.000 | 0.049 |

`gpt-oss:20b` never improved Brier (96/96 CUE = 0). Gemma and Nemotron update on matched/cross poems and barely (or not) on random/irrelevant text.

## E7 (primary = `gemma4:31b`)

**Passed: False** (cross control indeterminate). Length gate **passed** (mean |bits ratio − 1| = 0.0023 < 0.20).

| Control | n | ΔCUE (matched − control) | BCa 95% CI | p_adj BY | verdict | pass |
|---------|---|--------------------------|------------|----------|---------|------|
| cross | 24 | +2.06e-5 | [−7.2e-6, +7.0e-5] | 0.79 | indeterminate | no |
| random | 24 | +5.83e-4 | [+5.0e-4, +6.4e-4] | 0.0014 | different | **yes** |
| irrelevant | 24 | +6.84e-4 | [+6.8e-4, +7.0e-4] | 0.0014 | different | **yes** |

Causal read: on this synthetic poetry panel, Gemma’s CUE rises for a real contribution vs shuffled/off-topic text, but **not** detectably vs another cluster’s poem (`cross`).

## E5a receiver rank agreement (matched-arm CUE)

**Passed: False.** Target: Spearman ρ and Lin’s CCC CI lower > 0.80.

| Pair | Spearman ρ (CI) | CCC (CI) | pass |
|------|-----------------|----------|------|
| gemma4:31b vs gpt-oss:20b | nan (gpt-oss constant 0) | 0 | no |
| gemma4:31b vs nemotron-3-nano:30b | 0.10 [−0.29, 0.48] | −0.009 | no |
| gpt-oss:20b vs nemotron-3-nano:30b | nan | 0 | no |

Krippendorff α = −0.29 (inadequate). Receivers do **not** rank the same items the same way on this panel.

## E5b encoder R_D (unchanged from the 401-blocked pass)

**Passed: False.** Mean CV original/plain/technical = 1.795 (CI upper ≮ 0.15). Scrambled negative control also high (control_ok). Degenerate CIs: every synthetic item produced the same CV.

## Protocol deviations

1. Receivers are free-tier Cloud IDs, not PROTOCOL local tags and not the 402 Cloud IDs.
2. n=24 not n=80; synthetic Phase A, not Postgres poems.
3. E7 judge is `gemma4:31b` (Gemma that authenticated), not `llama3.1:8b` / `glm-5.1`.
4. `score_cue_panel.py` now logs per-job API errors instead of aborting the pool (needed after the 402 crash).

## Artifacts

`y_panel.jsonl`, `e7_cue_panel.jsonl` (288 rows), `e5_rd_panel.jsonl`, `e7_report.*`, `e5_report.*`. Isolation: nothing under `experiments/experiment1/results/`.
