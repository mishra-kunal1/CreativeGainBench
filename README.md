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
```

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

Samples 300 items (seed=42) from each dataset into `data/subset/`:

- **infinity_chat_subset.jsonl** — first user message extracted as prompt
- **formalmath_subset.jsonl** — filtered to Algebra and Number Theory domains
- **rinobench_low_novelty.jsonl** - this dataset has only 299 prompts, so no need to do subset operation.

## 3. Running inference

```bash
python -m creativegainbench.model \
  --data data/subset/infinity_chat_subset.jsonl \
  --limit 10 \
  --n 5
```

Runs gpt-4o-mini (temperature=1.0, top_p=0.9) on prompts and generates `n` responses per prompt. Results are saved to `data/results/<model>/<timestamp>/<input_filename>.jsonl`. Use `--workers` to control concurrency (default: min(limit, 64)).
