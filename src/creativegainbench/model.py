"""
Generate model responses for CreativeGainBench prompts.

Supports OpenAI, Ollama (OpenAI-compatible local API), and OpenRouter.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg)


DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
TOP_P = 0.9
TEMPERATURE = 1.0

MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 60.0
RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError)

T = TypeVar("T")


def _with_retries(fn: Callable[[], T], *, tag: str) -> T:
    """Call fn(), retrying transient API failures with exponential backoff + full jitter."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except RETRYABLE_EXCEPTIONS + (APIStatusError,) as exc:
            retryable = isinstance(exc, RETRYABLE_EXCEPTIONS) or exc.status_code >= 500
            if not retryable or attempt == MAX_RETRIES:
                raise
            delay = random.uniform(0, min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * 2**attempt))
            _log(
                f"  !! [{tag}] {type(exc).__name__}, retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


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
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY required for provider=openrouter")
        return OpenAI(api_key=api_key, base_url=base_url or "https://openrouter.ai/api/v1")
    raise ValueError(f"Unknown provider: {provider!r} (use openai|ollama|openrouter)")


def _call_api(
    client: OpenAI,
    model: str,
    prompt: str,
    n: int,
    *,
    provider: str,
    domain: str | None = None,
    tag: str,
) -> dict:
    """
    Generate n completions. Ollama's OpenAI-compatible API typically ignores
    `n`, so we issue n sequential/parallel single-completion calls there.
    """
    responses: list[dict] = []
    if provider == "ollama" and n > 1:
        for i in range(n):
            completion = _with_retries(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    n=1,
                ),
                tag=tag,
            )
            text = completion.choices[0].message.content or ""
            responses.append({f"response-{i}": text})
    else:
        completion = _with_retries(
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                n=n,
            ),
            tag=tag,
        )
        responses = [
            {f"response-{i}": (choice.message.content or "")}
            for i, choice in enumerate(completion.choices)
        ]
    row: dict = {"prompt": prompt, "responses": responses}
    if domain is not None:
        row["domain"] = domain
    return row


def _record_key(row: dict) -> str:
    return row.get("id") or row["prompt"]


def load_done_keys(out_path: Path) -> set[str]:
    """Keys (id, or prompt text if no id) already present in a prior run's output file."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(_record_key(json.loads(line)))
    return done


def run_inference(
    records: list[dict[str, str]],
    *,
    client: OpenAI,
    model: str,
    provider: str,
    out_path: Path,
    n: int = 5,
    workers: int = 64,
    append: bool = False,
) -> int:
    """
    Run inference and write each record's row to out_path as soon as it
    completes (not after the whole batch finishes), so a crash mid-run only
    loses the in-flight requests, not everything already generated.
    """
    workers = max(1, min(workers, len(records)))

    def _one(record: dict[str, str]) -> dict:
        tag = record.get("id", "?")
        row = _call_api(
            client,
            model,
            record["prompt"],
            n,
            provider=provider,
            domain=record.get("domain"),
            tag=tag,
        )
        if "id" in record:
            row["id"] = record["id"]
        return row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "a" if append else "w") as f, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_one, record): record for record in records}
        for future in as_completed(futures):
            row = future.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="JSONL with 'prompt' field")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n", type=int, default=5, help="Completions per prompt")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama", "openrouter"],
        default="ollama",
        help="Model provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name (default: gemma2:2b for ollama, gpt-4o-mini for openai; "
            "required for openrouter, e.g. qwen/qwen3.6-27b)"
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "Override API base URL (ollama default http://127.0.0.1:11434/v1, "
            "openrouter default https://openrouter.ai/api/v1)"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to a previous run's output JSONL. Records already present there "
            "(matched by 'id', or by prompt text if the input has no id) are "
            "skipped, and new results are appended to this same file."
        ),
    )
    args = parser.parse_args()

    model = args.model or (
        "gemma2:2b" if args.provider == "ollama"
        else "gpt-4o-mini" if args.provider == "openai"
        else None
    )
    if model is None:
        parser.error("--model is required for provider=openrouter")
    client = _make_client(args.provider, args.base_url)

    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        records = records[: args.limit]

    if args.resume:
        out_path = args.resume
        done = load_done_keys(out_path)
        before = len(records)
        records = [r for r in records if _record_key(r) not in done]
        print(f"Resuming {out_path}: {before - len(records)} already done, {len(records)} remaining")
    else:
        safe_model = model.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args.output_dir / safe_model / timestamp / f"{args.data.stem}.jsonl"

    if not records:
        print("Nothing to do — all prompts already completed.")
        return

    workers = args.workers
    if workers is None:
        # Keep ollama concurrency modest to avoid GPU thrash.
        workers = min(len(records), 2 if args.provider == "ollama" else 64)

    print(
        f"Running {len(records)} prompts on {args.provider}/{model} "
        f"(n={args.n}, workers={workers})"
    )
    written = run_inference(
        records,
        client=client,
        model=model,
        provider=args.provider,
        out_path=out_path,
        n=args.n,
        workers=workers,
        append=bool(args.resume),
    )

    print(f"Saved {written} items to {out_path}")


if __name__ == "__main__":
    main()
