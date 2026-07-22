"""
Create 300-item subsets from the downloaded datasets.

- infinity_chat: extracts first user message content
- formalmath: filters to Algebra and Number Theory domains
"""

import json
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SUBSET_DIR = DATA_DIR / "subset"
SUBSET_SIZE = 300


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def subset_infinity_chat():
    items = load_jsonl(DATA_DIR / "infinity_chat_creative.jsonl")

    random.shuffle(items)
    subset = items[:SUBSET_SIZE]

    prompts = []
    for item in subset:
        first_user_msg = next(m["content"] for m in item["messages"] if m["role"] == "user")
        prompts.append({"prompt": first_user_msg})

    out_path = SUBSET_DIR / "infinity_chat_subset.jsonl"
    with open(out_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"Saved {len(prompts)} prompts to {out_path}")


def subset_formalmath():
    items = load_jsonl(DATA_DIR / "formalmath.jsonl")

    filtered = [
        item for item in items
        if "Algebra" in item["domain"] or "Number Theory" in item["domain"]
    ]

    random.shuffle(filtered)
    subset = filtered[:SUBSET_SIZE]

    out_path = SUBSET_DIR / "formalmath_subset.jsonl"
    with open(out_path, "w") as f:
        for item in subset:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(subset)} items to {out_path}")


def main():
    random.seed(SEED)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    subset_infinity_chat()
    subset_formalmath()

    print(f"\nDone. Subsets saved to: {SUBSET_DIR}")


if __name__ == "__main__":
    main()
