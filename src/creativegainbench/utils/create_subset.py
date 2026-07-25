"""
Create 300-item subsets from the downloaded datasets.

- infinity_chat: extracts first user message content as prompt
- formalmath: filters to Algebra and Number Theory domains; writes prompt
- rinobench: maps research_idea fields to prompt (full set is ~299)
"""

import json
import random
from pathlib import Path

from creativegainbench.prompts import (
    prompt_from_formalmath,
    prompt_from_infinity_chat,
    prompt_from_rinobench,
)

SEED = 42
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SUBSET_DIR = DATA_DIR / "subset"
SUBSET_SIZE = 300


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_prompts(path: Path, prompts: list[dict]):
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


def subset_infinity_chat():
    items = load_jsonl(DATA_DIR / "infinity_chat_creative.jsonl")

    random.shuffle(items)
    subset = items[:SUBSET_SIZE]

    prompts = [{"prompt": prompt_from_infinity_chat(item)} for item in subset]

    out_path = SUBSET_DIR / "infinity_chat_subset.jsonl"
    write_prompts(out_path, prompts)
    print(f"Saved {len(prompts)} prompts to {out_path}")


def subset_formalmath():
    items = load_jsonl(DATA_DIR / "formalmath.jsonl")

    filtered = [
        item for item in items
        if "Algebra" in item["domain"] or "Number Theory" in item["domain"]
    ]

    random.shuffle(filtered)
    subset = filtered[:SUBSET_SIZE]

    prompts = [{"prompt": prompt_from_formalmath(item)} for item in subset]

    out_path = SUBSET_DIR / "formalmath_subset.jsonl"
    write_prompts(out_path, prompts)
    print(f"Saved {len(prompts)} prompts to {out_path}")


def subset_rinobench():
    items = load_jsonl(DATA_DIR / "rinobench_low_novelty.jsonl")
    prompts = [{"prompt": prompt_from_rinobench(item)} for item in items]

    out_path = SUBSET_DIR / "rinobench_subset.jsonl"
    write_prompts(out_path, prompts)
    print(f"Saved {len(prompts)} prompts to {out_path}")


def main():
    random.seed(SEED)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    subset_infinity_chat()
    subset_formalmath()
    subset_rinobench()

    print(f"\nDone. Subsets saved to: {SUBSET_DIR}")


if __name__ == "__main__":
    main()
