# CreativeGainBench

Benchmark for evaluating creativity gain in agent-generated outputs across scientific proposal writing, creative writing, and mathematical proof writing.

## Setup

```bash
pip install -e .
```

Create a `.env` file with your tokens:

```
HF_TOKEN=hf_your_token_here
OPENAI_API_KEY=sk-your_key_here
OPENROUTER_API_KEY=sk-or-your_key_here
OLLAMA_API_KEY=your_ollama_cloud_key_here
```

| Variable | Used for |
|----------|----------|
| `HF_TOKEN` | Downloading Hugging Face datasets |
| `OPENAI_API_KEY` | Current OpenAI inference (`model.py`) |
| `OPENROUTER_API_KEY` | LLM-as-judge + live $/token prices for cost estimates |
| `OLLAMA_API_KEY` | Ollama Cloud open models (no local downloads); create at [ollama.com/settings/keys](https://ollama.com/settings/keys) |

## 1. Downloading the datasets

```bash
download-datasets
```

This downloads three datasets into `data/`:

| Domain | Source | Filter |
|--------|--------|--------|
| Scientific proposal writing | [TimSchopf/RINoBench](https://huggingface.co/datasets/TimSchopf/RINoBench) | `novelty_score <= 2` |
| Creative writing | [liweijiang/infinite-chats-taxonomy](https://huggingface.co/datasets/liweijiang/infinite-chats-taxonomy) | `category == "Creative Content Generation"` |
| Mathematical proof writing | [SphereLab/FormalMATH-All](https://huggingface.co/datasets/SphereLab/FormalMATH-All) | None (full dataset) |

## 2. Creating subsets

```bash
create-subset
```

Writes normalized `{"prompt": ...}` JSONL into `data/subset/`:

- **infinity_chat_subset.jsonl** — first user message (300 samples, seed=42)
- **formalmath_subset.jsonl** — Algebra / Number Theory `refined_statement` as prompt (300 samples)
- **rinobench_subset.jsonl** — research idea fields composed into a prompt (~299 rows)

## 3. Estimating multi-provider cost

After subsets exist:

```bash
estimate-cost --sample 10 --n 5
```

This does **not** run generation. It:

1. Samples prompts from all three domains
2. Fetches live OpenRouter prices for GPT / Claude / Gemini and for open-model **$/token cross-quotes**
3. Lists Ollama Cloud models (`/api/tags`) for Llama / DeepSeek / Kimi / GLM / Qwen availability (no local `ollama pull`)
4. Estimates **generation + LLM-as-judge** costs
5. Writes a Markdown summary and JSON under `data/evaluation/` (e.g. `cost_estimate_<timestamp>.md`)

Open-model execution path is **Ollama Cloud** (subscription / GPU-time quota, not $/token). Dollar figures for those families are OpenRouter cross-quotes for research comparison only.

Useful flags: `--domains`, `--assumed-completion-tokens`, `--assumed-judge-completion-tokens`, `--output-md`, `--output-json`.

## 4. Running inference

```bash
python -m creativegainbench.model \
  --data data/subset/infinity_chat_subset.jsonl \
  --limit 10 \
  --n 5
```

Runs gpt-4o-mini (temperature=1.0, top_p=0.9) on prompts and generates `n` responses per prompt. Results are saved to `data/results/<model>/<timestamp>/<input_filename>.jsonl`. Use `--workers` to control concurrency (default: min(limit, 64)).
