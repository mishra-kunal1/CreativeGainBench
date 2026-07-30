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

`R_D` uses idea-level symbols (MiniLM → frozen codebook) compressed with **KenLM**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Frozen artifacts (idea codebook, KenLM model, boundary detector, manifest, δ_D)
are **not committed**. Generate them once after install — deterministic under
seed 42 from the committed source JSONs in `src/creativegainbench/artifacts/`:

```bash
prepare-artifacts
calibrate-delta-d
```

Optional `.env`:

```
HF_TOKEN=hf_...
OPENAI_API_KEY=sk_...
OLLAMA_API_KEY=ollama
```

## End-to-end (Ollama)

```bash
# 1) Eval prompts (HF subsets if downloaded, else packaged held-out bank)
create-subset

# 2) Frozen artifacts (codebook + KenLM on decontaminated train corpus)
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

## Dataset download (optional)

```bash
download-datasets   # HF_TOKEN optional for public datasets
create-subset       # decontaminates vs frozen probe set P
```

## Package layout

```
src/creativegainbench/
├── ideas/            # MiniLM spans, codebook, artifact loader
├── metrics/          # cue (+ belief receiver), KenLM R_D, R_B, G_k
├── receivers/        # hash / openai / ollama
├── artifacts/        # probes, train corpus, KenLM, δ_D, manifests
├── eval/             # benchmark_eval, mas_infer, report
├── utils/            # download, eval prompts, contamination
└── benchmark_score.py
```

## Tests

```bash
pytest -q
```

## Formal backing

`math_backing/` — Lean proofs for `Rcreativity`, gates, and D-channel protocol.

## Planned work

- Swap KenLM for fine-tuned LLM based idea segmentation and encoding (replacing
  the MiniLM span encoder + frozen codebook + KenLM n-gram compressor).
