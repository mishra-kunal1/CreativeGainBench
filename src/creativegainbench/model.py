import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"
TOP_P = 0.9
TEMPERATURE = 1.0


def _call_api(prompt: str, n: int) -> dict:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        n=n,
    )
    responses = [{f"response-{i}": choice.message.content} for i, choice in enumerate(completion.choices)]
    return {"prompt": prompt, "responses": responses}


def run_inference(prompts: list[str], n: int = 5, workers: int = 64) -> list[dict]:
    workers = min(workers, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda p: _call_api(p, n), prompts))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to .jsonl file with 'prompt' field")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"), help="Base output directory")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to run")
    parser.add_argument("--n", type=int, default=5, help="Number of responses per prompt")
    parser.add_argument("--workers", type=int, default=None, help="Max concurrent requests (default: min(limit, 64))")
    args = parser.parse_args()

    with open(args.data) as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    if args.limit:
        prompts = prompts[:args.limit]

    workers = args.workers or min(len(prompts), 64)
    results = run_inference(prompts, n=args.n, workers=workers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / MODEL / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.data.stem}.jsonl"
    with open(out_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(results)} items to {out_path}")


if __name__ == "__main__":
    main()
