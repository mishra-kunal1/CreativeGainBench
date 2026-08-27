# OLLAMA-CLOUD-01 — E5/E7 falsifiability (Ollama Cloud)

Live execution of the frozen E5/E7 protocol on Ollama Cloud. **PROTOCOL.md was not edited.**

This is a **quota-bounded first look at n=24**, not the pre-registered PROTOCOL n=80. Gemma2:2b / Llama 3.1 8B / Phi-4 14B local tags were unavailable on Ollama Cloud. **CUE was not scored:** the cloud chat endpoint returned 401 before any (item, receiver, arm) triple was written. **Do not treat empty CUE cells as zeros.**

## Run metadata

| Knob | Value |
|------|--------|
| Run ID | `OLLAMA-CLOUD-01` |
| Date (UTC) | 2026-08-27 |
| Branch (code) | `cursor/falsifiability-e5-e7-c174` |
| Branch (this run) | `cursor/ollama-cloud-e5-e7-a873` |
| Phase | A, `--synthetic` (no Postgres) |
| n (this report) | **24** (stratified by `domain_cluster`; 8/8/8 on clusters 0,1,2) |
| Seed | **42** |
| CUE temperature | 0 (config; never reached a live completion) |
| Workers | 2 (config; CUE pool never started) |
| Encoder / device | MiniLM, CPU |
| Base URL | `https://ollama.com/v1` |
| Tags URL | `GET https://ollama.com/api/tags` → HTTP 200 (no auth) |
| Isolation | **did not write** `experiments/experiment1/results/cue_*.jsonl` |

## Actual model IDs (Ollama Cloud)

`GET https://ollama.com/api/tags` listed 18 public tags, including the planned override set. **None 404'd on the tags list.** Chat was not attempted past the smoke call (401).

| Band | PROTOCOL (local) | This run (Ollama Cloud) | Tags list |
|------|------------------|-------------------------|-----------|
| Weak / Gemma | `gemma2:2b` | `gemma4:31b` | present (only Gemma tag) |
| Mid (E7 primary judge) | `llama3.1:8b` | `glm-5.1` | present |
| Strong | `phi4:14b` | `qwen3.5:397b` | present |

Also present and unused: `gpt-oss:20b`, `kimi-k2.6`, `glm-5.2`, `glm-5.3-flash`, … Full list: `ollama_cloud_model_ids.json`.

PROTOCOL local tags **absent** from the cloud list: `gemma2:2b`, `llama3.1:8b`, `phi4:14b`. No Phi. No Llama 3.1 8B.

## Auth (CUE blocker)

- `os.environ["OLLAMA_API_KEY"]` was **missing**.
- Smoke: `OpenAI(base_url=https://ollama.com/v1, api_key=fallback "ollama")` → `chat.completions.create(model="gemma4:31b", temperature=0)`.
- **error_class:** `openai.AuthenticationError`
- **http_status:** `401`
- **error.type:** `api_error`
- **error.message:** `Unauthorized`
- **error.code:** `null`
- Same 401 with no `Authorization` header (`curl`).
- Tags listing does **not** require a key (HTTP 200).

No further CUE jobs were submitted. No fake scores.

## Protocol deviations (run card)

PROTOCOL.md stays frozen. Deviations in this execution:

1. **Receivers.** Cloud override table above (Gemma4 31B / GLM-5.1 / Qwen3.5 397B) instead of `gemma2:2b` / `llama3.1:8b` / `phi4:14b`.
2. **n=24 not n=80.** Pre-registered size is 80. This report is labeled n=24. Resume toward 80 was not attempted: chat never authenticated.
3. **Synthetic Phase A panel** (`--synthetic`), not Postgres poems. No `DATABASE_URL`.
4. **CUE not scored** (401). E7 and E5a therefore have no pass/fail from data.
5. **analyze_e5.py CLI** is `--cue-scores` / `--rd-scores`, not `--cue` / `--rd`. Used the real flags.
6. **score_rd_encoder.py import isolation.** `_load_frozen_stack` now pops the cached `experiments/falsifiability/lib` module before importing `construct_validity.metrics.pipeline` (`from lib import load_config` otherwise hits the wrong `lib` and `KeyError: max_chars`). PROTOCOL unchanged. Documented so the R_D path is runnable.
7. **Probe bank** is the synthetic 8-probe bank (`construct_probe_pairs.py --synthetic`), not poetry_v2 `meta.json` probes (those would require the same constructors; `--synthetic` was requested).
8. **Did not resume CUE to n=80** after analysis — CUE never started.

## Per-receiver mean CUE

n scored CUE triples: **0**. Table is structural only.

| Receiver | mean CUE matched | mean CUE cross | mean CUE random | mean CUE irrelevant |
|----------|------------------|----------------|-----------------|---------------------|
| `gemma4:31b` | *not scored* | *not scored* | *not scored* | *not scored* |
| `glm-5.1` | *not scored* | *not scored* | *not scored* | *not scored* |
| `qwen3.5:397b` | *not scored* | *not scored* | *not scored* | *not scored* |

## E7 (primary = `glm-5.1`)

**Pass/fail: not scored.**

Three planned PairedMeanDiff rows (CUE_matched − CUE_control):

| Control | n | estimate | CI | p | p_adj BY | verdict | pass |
|---------|---|----------|----|---|----------|---------|------|
| cross | 0 | — | — | — | — | CUE not scored | no |
| random | 0 | — | — | — | — | CUE not scored | no |
| irrelevant | 0 | — | — | — | — | CUE not scored | no |

Length gate `|bits_arm / bits_matched − 1| < 0.20`: not computed (no `bit_length` rows).

## E5a pairwise Spearman / CCC (matched-arm CUE)

**Pass: False** (n_items=0; no receiver pairs). Needs complete matched-arm CUE for all three receivers.

| Pair | Spearman ρ | CCC | pass |
|------|------------|-----|------|
| gemma4:31b vs glm-5.1 | *not scored* | *not scored* | no |
| gemma4:31b vs qwen3.5:397b | *not scored* | *not scored* | no |
| glm-5.1 vs qwen3.5:397b | *not scored* | *not scored* | no |

## E5b mean CV (R_D ran)

R_D encoder **did run** (no Ollama). Frozen poetry_v2 `idea_codebook.pt` + `domain_*_ctx.pkl` were present. MiniLM on CPU. n=24 synthetic matched y's. Probe encodings rebuilt for original / plain / technical + scrambled_0/1/2.

| Arm | n | mean CV | BCa 95% CI | threshold | pass |
|-----|---|---------|------------|-----------|------|
| original+plain+technical | 24 | 1.79508 | [1.79508, 1.79508] | &lt; 0.15 | **no** |
| scrambled negative control | 24 | 1.34179 | [1.34179, 1.34179] | must **not** pass 0.15 | control_ok=**yes** |

**E5b passed: False** (positive mean-CV CI upper is not &lt; 0.15). Scrambled control did not pass the bar, so the negative control discriminated. Point-identical CIs: every synthetic item produced the same finite CV (degenerate BCa). This is a synthetic-panel property, not a live-poem result.

**E5 overall (E5a ∧ E5b): False.**

## Artifacts (this directory)

| File | Role |
|------|------|
| `y_panel.jsonl` | n=24 four-arm panel, seed 42 |
| `probe_pairs.json` | synthetic probe banks |
| `e5_rd_panel.jsonl` | R_D scores |
| `e5_report.json` / `e5_report.md` | `analyze_e5.py` |
| `e7_report.json` / `e7_report.md` | E7 blocked on auth (not from `analyze_e7.py` on live scores) |
| `e7_cue_panel.jsonl` | **absent** (0 CUE rows) |
| `auth_error.json` | 401 class/status (no API key material) |
| `ollama_cloud_model_ids.json` | public tag names |
| `ollama_cloud_tags.json` | raw `GET /api/tags` body |

## Isolation

Did **not** write (or read as this suite's input glob) `experiments/experiment1/results/cue_*.jsonl`. All outputs live under `experiments/falsifiability/runs/OLLAMA-CLOUD-01/`.
