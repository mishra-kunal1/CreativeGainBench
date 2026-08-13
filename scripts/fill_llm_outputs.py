"""
Fill the `llm_output` column of the poems Postgres DB by generating a sample
from each row's `prompt` with a local Ollama model (OpenAI-compatible API,
same convention as creativegainbench.model).

Resumable: only rows with a prompt and NULL/empty llm_output are processed.

Usage:
  python scripts/fill_llm_outputs.py [--model gemma2:2b] [--workers 3] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg
from openai import OpenAI

DEFAULT_DB = "postgres://poems:poems@localhost:5432/poems"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
TEMPERATURE = 1.0
TOP_P = 0.9
# ~80-100 lines of verse; covers p90 of requested poem lengths while bounding
# runtime on the long-tail "~2000-line epic" prompts.
MAX_TOKENS = 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_DB))
    args = parser.parse_args()

    client = OpenAI(
        base_url=DEFAULT_OLLAMA_BASE,
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )

    with psycopg.connect(args.db) as conn:
        rows = conn.execute(
            "SELECT id, prompt FROM poems "
            "WHERE prompt IS NOT NULL AND btrim(prompt) <> '' "
            "AND (llm_output IS NULL OR btrim(llm_output) = '') "
            "ORDER BY id"
        ).fetchall()
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)
    print(f"{total} poems need llm_output (model={args.model}, workers={args.workers})", flush=True)
    if total == 0:
        return

    write_conn = psycopg.connect(args.db, autocommit=True)
    write_lock = threading.Lock()
    done = 0
    failed = 0
    start = time.time()

    def _generate(prompt: str) -> str:
        completion = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            n=1,
        )
        return (completion.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_generate, prompt): poem_id for poem_id, prompt in rows}
        for future in as_completed(futures):
            poem_id = futures[future]
            try:
                text = future.result()
                if not text:
                    raise RuntimeError("empty response")
                with write_lock:
                    write_conn.execute(
                        "UPDATE poems SET llm_output = %s WHERE id = %s", (text, poem_id)
                    )
                done += 1
            except Exception as e:
                failed += 1
                print(f"[warn] {poem_id}: {e}", flush=True)
            n = done + failed
            if n % 25 == 0 or n == total:
                rate = n / (time.time() - start)
                eta_min = (total - n) / rate / 60 if rate > 0 else float("inf")
                print(
                    f"progress {n}/{total} (ok={done} fail={failed}, "
                    f"{rate:.2f}/s, eta {eta_min:.0f} min)",
                    flush=True,
                )

    write_conn.close()
    print(f"DONE: ok={done} fail={failed} elapsed={(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
