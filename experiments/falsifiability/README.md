# Falsifiability experiments (E5 / E7)

Receiver/encoder stability (**E5**) and causal contribution controls (**E7**).

These IDs are **not** the construct-validity E4/E5 checks. Read
[`PROTOCOL.md`](PROTOCOL.md) before any live scoring.

**Isolation:** this suite never writes `experiments/experiment1/results/cue_*.jsonl`.

## Layout

```
experiments/falsifiability/
  PROTOCOL.md                 # frozen pass rules (read first)
  config.toml                 # n, seed, receivers, base_url, workers
  lib.py
  construct_contributions.py  # four y arms → results/y_panel.jsonl
  construct_probe_pairs.py    # original/plain/technical + scrambled
  score_cue_panel.py          # Ollama CUE; cached priors; frozen z*
  score_rd_encoder.py         # R_D across probe banks (frozen H)
  analyze_e7.py / analyze_e5.py
  run_all.py
  results/                    # generated JSONL + reports (gitignored)
```

## Smoke (no Ollama / Postgres)

```bash
python experiments/falsifiability/run_all.py --phase a --limit 8 --synthetic --skip-score
python -m pytest tests/test_creativity_stats.py tests/test_falsifiability.py -q
```

`--synthetic` builds a tiny y-panel and probe bank so constructors are
testable. `--skip-score` skips live CUE / R_D (no GPU, no Ollama).

## Phase A (poetry; live Ollama — follow-up)

Requires local Ollama or Ollama Cloud, and Phase A poems (Postgres) **or**
`--from-jsonl` with `{id,prompt,body,domain_cluster}` rows.

```bash
# local daemon
python experiments/falsifiability/run_all.py --phase a --limit 4 \
  --base-url http://127.0.0.1:11434/v1

# Ollama Cloud
python experiments/falsifiability/run_all.py --phase a --limit 4 \
  --base-url https://ollama.com/v1
```

Resume from a step:

```bash
python experiments/falsifiability/run_all.py --phase a --from score_cue_panel
```

Individual constructors:

```bash
python experiments/falsifiability/construct_contributions.py --phase a --limit 8 --synthetic
python experiments/falsifiability/construct_probe_pairs.py --synthetic
python experiments/falsifiability/score_cue_panel.py --limit 4 --base-url http://127.0.0.1:11434/v1
python experiments/falsifiability/score_rd_encoder.py --limit 4
python experiments/falsifiability/analyze_e7.py
python experiments/falsifiability/analyze_e5.py
```

`score_cue_panel.py` is resume-safe JSONL, caches priors per `(receiver, prompt)`,
and freezes z* from `y_matched` (see PROTOCOL). Temperature is 0.

`--encoder hash` avoids MiniLM for z* smoke; default Phase A is MiniLM.

## Phase B (code + protocol; not a live run here)

```bash
python experiments/falsifiability/run_all.py --phase b --limit 8 --synthetic --skip-score
```

When `data/subset/eval_all_domains.jsonl` exists, omit `--synthetic` to panel
those prompts (still needs matched y via `--y-jsonl` or `--synthetic` bodies).

Math validity bit is an **LLM proxy** (`OllamaReceiverAgent.condition` +
verifier JSON). It is **not** a Lean 4 oracle.

## Receivers

| Band | Model |
|------|--------|
| Weak | `gemma2:2b` |
| Mid (E7 official judge) | `llama3.1:8b` |
| Strong | `phi4:14b` |
