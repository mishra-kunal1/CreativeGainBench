# Experiment 1 — Multi-model poetry_v2 ladder

Reproduce the frozen `poetry_v2` human-vs-LLM measurement across a capability ladder of local Ollama models. Measurement artifacts (codebook, domain probes, \(\mathcal{H}\), splits) are **not** rebuilt.

Baseline single-model writeup: [`REPORT_baseline.md`](REPORT_baseline.md).

## Layout

```
experiments/experiment1/
  config.toml                 # models, gen knobs, cue subsample size
  lib.py                      # shared config loader
  run_all.py                  # orchestrator
  components/
    01_schema.py              # generations table + migrate gemma2:2b
    02_generate.py            # fill generations for each model (eval split)
    03_score_rd.py            # λ_D-normalized deformation R_D
    04_score_cue.py           # CUE + external z* on frozen IDs
    05_aggregate.py           # ladder_summary.json + results/REPORT.md
  results/                    # JSONL, summaries, reports
  logs/                       # per-step logs from run_all
```

## Model ladder (config.toml)

| Tier | Model |
|------|--------|
| Tiny | `gemma2:2b` |
| Mid | `mistral:latest` |
| Mid | `llama3.1:8b` |
| Stronger | `phi4:14b` |

## Run

```bash
# Full pipeline
python experiments/experiment1/run_all.py

# Smoke (10 gens/model)
python experiments/experiment1/run_all.py --limit 10

# Resume from a step
python experiments/experiment1/run_all.py --from 03_score_rd

# One model only
python experiments/experiment1/run_all.py --model llama3.1:8b

# Skip expensive CUE
python experiments/experiment1/run_all.py --skip-cue
```

Or run components directly:

```bash
python experiments/experiment1/components/01_schema.py
python experiments/experiment1/components/02_generate.py --model phi4:14b
```

## Hypotheses

1. Human−model \(R_D\) gap shrinks as models get stronger.
2. Model CUE / `novel_structure` rate rises with tier.
3. Domains that favored humans for gemma2:2b stay hardest for weak models.
