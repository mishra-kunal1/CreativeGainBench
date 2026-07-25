"""
Download datasets for CreativeGainBench evaluation.

Datasets:
1. RINoBench - low-novelty stratum (novelty_score 1-2)
2. INFINITY-CHAT - creative content generation subset
3. FormalMATH - full dataset, no filter

HF_TOKEN is optional for public datasets but recommended for rate limits.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data"


def _token_kwargs() -> dict:
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def download_rinobench():
    print("Downloading RINoBench...")
    ds = load_dataset("TimSchopf/RINoBench", **_token_kwargs())
    split = ds["train"]
    filtered = split.filter(lambda x: x["novelty_score"] <= 2)
    out_path = OUTPUT_DIR / "rinobench_low_novelty.jsonl"
    with open(out_path, "w") as f:
        for item in filtered:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(filtered)} prompts to {out_path}")


def download_infinity_chat():
    print("Downloading INFINITY-CHAT...")
    ds = load_dataset("liweijiang/infinite-chats-taxonomy", **_token_kwargs())
    split = ds["train"]
    filtered = split.filter(
        lambda x: any(
            cat["category"] == "Creative Content Generation"
            for cat in x["categories"]
        )
    )
    out_path = OUTPUT_DIR / "infinity_chat_creative.jsonl"
    with open(out_path, "w") as f:
        for item in filtered:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(filtered)} prompts to {out_path}")


def download_formalmath():
    print("Downloading FormalMATH...")
    ds = load_dataset("SphereLab/FormalMATH-All", **_token_kwargs())
    split = ds["train"]
    out_path = OUTPUT_DIR / "formalmath.jsonl"
    with open(out_path, "w") as f:
        for item in split:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(split)} items to {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not HF_TOKEN:
        print("Note: HF_TOKEN unset; using anonymous Hub access (rate limits apply).")

    tasks = [download_rinobench, download_infinity_chat, download_formalmath]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): fn.__name__ for fn in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    print("\nDone. All datasets saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
