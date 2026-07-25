"""
Generate model responses for CreativeGainBench prompts.

Supports OpenAI and Ollama (OpenAI-compatible local API).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
TOP_P = 0.9
TEMPERATURE = 1.0


def _make_client(provider: str, base_url: str | None) -> OpenAI:
    provider = provider.lower()
    if provider == "ollama":
        return OpenAI(
            base_url=base_url or DEFAULT_OLLAMA_BASE,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for provider=openai")
        return OpenAI(api_key=api_key, base_url=base_url)
    raise ValueError(f"Unknown provider: {provider!r} (use openai|ollama)")


def _call_api(
    client: OpenAI,
    model: str,
    prompt: str,
    n: int,
    *,
    provider: str,
) -> dict:
    """
    Generate n completions. Ollama's OpenAI-compatible API typically ignores
    `n`, so we issue n sequential/parallel single-completion calls there.
    """
    responses: list[dict] = []
    if provider == "ollama" and n > 1:
        for i in range(n):
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                n=1,
            )
            text = completion.choices[0].message.content or ""
            responses.append({f"response-{i}": text})
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            n=n,
        )
        responses = [
            {f"response-{i}": (choice.message.content or "")}
            for i, choice in enumerate(completion.choices)
        ]
    return {"prompt": prompt, "responses": responses}


def run_inference(
    prompts: list[str],
    *,
    client: OpenAI,
    model: str,
    provider: str,
    n: int = 5,
    workers: int = 64,
) -> list[dict]:
    workers = max(1, min(workers, len(prompts)))

    def _one(p: str) -> dict:
        return _call_api(client, model, p, n, provider=provider)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_one, prompts))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="JSONL with 'prompt' field")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n", type=int, default=5, help="Completions per prompt")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="Model provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gemma2:2b for ollama, gpt-4o-mini for openai)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override API base URL (ollama default http://127.0.0.1:11434/v1)",
    )
    args = parser.parse_args()

    model = args.model or (
        "gemma2:2b" if args.provider == "ollama" else "gpt-4o-mini"
    )
    client = _make_client(args.provider, args.base_url)

    with open(args.data) as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    if args.limit:
        prompts = prompts[: args.limit]

    workers = args.workers
    if workers is None:
        # Keep ollama concurrency modest to avoid GPU thrash.
        workers = min(len(prompts), 2 if args.provider == "ollama" else 64)

    print(f"Running {len(prompts)} prompts on {args.provider}/{model} (n={args.n}, workers={workers})")
    results = run_inference(
        prompts,
        client=client,
        model=model,
        provider=args.provider,
        n=args.n,
        workers=workers,
    )

    safe_model = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / safe_model / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.data.stem}.jsonl"
    with open(out_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(results)} items to {out_path}")


if __name__ == "__main__":
    main()
