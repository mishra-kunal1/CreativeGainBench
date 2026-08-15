# CreativeGainBench

Benchmark for evaluating creativity gain in agent-generated outputs across scientific proposal writing, creative writing, and mathematical proof writing.

Reported score (see `math_backing/Creativity/CUE/BenchmarkScore.lean`):

\[
R_{\mathrm{creativity}}
=
1[\mathrm{CUE}>0]
\cdot
1[R_D>\delta_D]
\cdot
\bigl(\mathrm{CUE}\cdot(1+\alpha R_B^{\to A})+\lambda_G G_k\bigr)
\]

`R_D` uses idea-level symbols (MiniLM → frozen codebook) with **CountNgram
ProbeCompressor** deformation (not KenLM).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Frozen artifacts (idea codebook, CountNgram domain contexts, boundary detector,
manifest, δ_D) are **not all committed**. Generate package `v1` artifacts once
after install — deterministic under seed 42 from the committed source JSONs in
`src/creativegainbench/artifacts/`:

```bash
prepare-artifacts
calibrate-delta-d
```

Optional `.env`:

```
HF_TOKEN=hf_...
OPENAI_API_KEY=sk_...
OLLAMA_API_KEY=ollama
GEMINI_API_KEY=...   # only needed for run-mas-agents --provider gemini
```

## End-to-end (Ollama)

```bash
# 1) Eval prompts (HF subsets if downloaded, else packaged held-out bank)
create-subset

# 2) Frozen artifacts (codebook + CountNgram contexts on decontaminated train)
prepare-artifacts
calibrate-delta-d

# 3) Generate with Ollama
python -m creativegainbench.model \
  --provider ollama --model gemma2:2b \
  --data data/subset/eval_all_domains.jsonl \
  --limit 20 --n 1 --workers 1

# 4) Score with calibrated CUE + Ollama receiver
run-benchmark \
  --results data/results/gemma2_2b/<timestamp>/eval_all_domains.jsonl \
  --cue-provider ollama --cue-model gemma2:2b \
  --receiver ollama --receiver-model gemma2:2b \
  --aggregate \
  --output data/evaluation/gemma2_2b_r_creativity.jsonl

# Optional multi-agent G_k path
run-mas --data data/subset/eval_all_domains.jsonl --limit 5 \
  --agents gemma2:2b,gemma2:2b --joint-model gemma2:2b
run-benchmark --results data/results/mas_.../<ts>/eval_all_domains.jsonl \
  --cue-provider ollama --receiver ollama --aggregate
```

## Multi-agent generation (Proposer-Critic-Verifier)

`run-mas-agents` runs a fixed 3-role triad instead of independent drafts:
each role first solves the task alone (`agent_texts`, the G_k baseline),
then Proposer drafts → Critic critiques → Proposer revises → Verifier
approves/rejects, looping up to `--max-revision-rounds`. Domain-specific
guidance is baked in per role (e.g. the math Verifier checks proof steps —
it's an LLM playing that role, **not** a real Lean 4 type-checker yet).
Supports `openai`, `gemini`, and `ollama` providers (all OpenAI-compatible
endpoints), with independent `--proposer-provider` / `--critic-provider` /
`--verifier-provider` overrides so e.g. the critic can run on a different
model than the other two roles. With ollama roles, `--workers` defaults to 2
(same GPU-thrash guard as `model.py`).

```bash
run-mas-agents \
  --data data/subset/eval_all_domains.jsonl --limit 5 \
  --provider openai --model gpt-4o \
  --critic-provider gemini --critic-model gemini-3.5-flash \
  --max-revision-rounds 1

# Fully local triad on Ollama:
run-mas-agents \
  --data data/subset/eval_all_domains.jsonl --limit 5 \
  --provider ollama --model gemma2:2b \
  --max-revision-rounds 1

# Output rows already have agent_texts/joint_text in the shape run-benchmark
# expects, so scoring needs no extra flags:
run-benchmark --results data/results/mas_agents_.../<ts>/eval_all_domains.jsonl \
  --cue-provider openai --receiver openai --aggregate
```

## Trajectory diagnostics (Step-CUE γ, Diverge-Converge)

Post-hoc diagnostic over a `run-mas-agents` transcript (see
`Creativity/CUE/Trajectory.lean`): re-scores CUE and `R_B^{→A}` at each
candidate-answer snapshot (the Proposer's draft, then each revision), fits
the Step-CUE curve `C∞(1−e^{−μt})+γt`, and checks the `R_B` sequence for a
diverge-then-converge peak. Needs ≥3 trajectory points for the DC score and
≥4 for a reliable curve fit — a triad that converges in 1 round (2 points,
common with strong critics/verifiers) reports `reliable: false` rather than
a fabricated fit.

```bash
run-trajectory-metrics \
  --results data/results/mas_agents_.../<ts>/eval_all_domains.jsonl \
  --cue-provider openai --cue-model gpt-4o-mini \
  --receiver openai --receiver-model gpt-4o-mini \
  --output data/evaluation/trajectory_metrics.jsonl
```

## Dataset download (optional)

```bash
download-datasets   # HF_TOKEN optional for public datasets
create-subset       # decontaminates vs frozen probe set P
```

## Package layout

```
src/creativegainbench/
├── ideas/            # MiniLM spans, codebook, artifact loader
├── metrics/          # cue (+ belief receiver), CountNgram R_D, R_B, G_k, trajectory (γ, DC)
├── receivers/        # hash / openai / ollama
├── artifacts/        # probes, train corpus, CountNgram contexts, δ_D, manifests
├── eval/             # benchmark_eval, mas_infer, mas_agents, trajectory_eval, report
├── utils/            # download, eval prompts, contamination
└── benchmark_score.py
```

## Tests

```bash
pytest -q
```

## Formal backing

`math_backing/` — Lean proofs for `Rcreativity`, gates, D-channel protocol,
the multi-agent bridge (`G_k`, `Creativity/MAS/`), and trajectory diagnostics
(Step-CUE `γ`, Diverge-Converge — `Creativity/CUE/Trajectory.lean`, mirrored
by `metrics/trajectory.py`).

## Planned work

- Swap MiniLM for fine-tuned LLM based idea segmentation and encoding (R_D
  already uses CountNgram ProbeCompressor deformation, not KenLM).
- Wire a real Lean 4 type-check oracle into the `mathematical_proof` domain's
  Verifier role (currently an LLM playing that role — see `eval/mas_agents.py`).
