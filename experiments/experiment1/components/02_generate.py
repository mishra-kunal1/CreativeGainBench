"""
Generate eval-split outputs into generations for each model in the ladder.
Resumable per (poem_id, model).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import load_config  # noqa: E402


def generate_model(cfg: dict, model: str, limit: int | None) -> None:
    gen = cfg["generation"]
    client = OpenAI(
        base_url=gen["ollama_base"],
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )
    with psycopg.connect(cfg["db_url"]) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.prompt
            FROM poems p
            WHERE p.split = %s
              AND p.prompt IS NOT NULL AND btrim(p.prompt) <> ''
              AND NOT EXISTS (
                SELECT 1 FROM generations g
                WHERE g.poem_id = p.id AND g.model = %s
              )
            ORDER BY p.id
            """,
            (cfg["split"], model),
        ).fetchall()
    if limit is not None:
        rows = rows[:limit]
    total = len(rows)
    print(f"[{model}] need {total} generations (workers={gen['workers']})", flush=True)
    if total == 0:
        return

    write_conn = psycopg.connect(cfg["db_url"], autocommit=True)
    lock = threading.Lock()
    done = failed = 0
    start = time.time()

    def _one(prompt: str) -> str:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(gen["temperature"]),
            top_p=float(gen["top_p"]),
            max_tokens=int(gen["max_tokens"]),
            n=1,
        )
        return (completion.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(max_workers=int(gen["workers"])) as ex:
        futs = {ex.submit(_one, prompt): pid for pid, prompt in rows}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                text = fut.result()
                if not text:
                    raise RuntimeError("empty response")
                with lock:
                    write_conn.execute(
                        """
                        INSERT INTO generations (poem_id, model, output, max_tokens, temperature)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (poem_id, model) DO UPDATE
                        SET output = EXCLUDED.output
                        """,
                        (
                            pid,
                            model,
                            text,
                            int(gen["max_tokens"]),
                            float(gen["temperature"]),
                        ),
                    )
                done += 1
            except Exception as e:
                failed += 1
                print(f"[warn] {model} {pid}: {e}", flush=True)
            n = done + failed
            if n % 25 == 0 or n == total:
                rate = n / max(time.time() - start, 1e-6)
                eta = (total - n) / rate / 60 if rate > 0 else float("inf")
                print(
                    f"[{model}] {n}/{total} ok={done} fail={failed} "
                    f"{rate:.2f}/s eta {eta:.0f}m",
                    flush=True,
                )
    write_conn.close()
    print(f"[{model}] DONE ok={done} fail={failed}", flush=True)


def generate_mas_model(cfg: dict, base_model: str, limit: int | None, *, score_edge_cue: bool) -> None:
    """PCV triad generations stored under model key pcv:<base_model> as JSON."""
    from openai import OpenAI as _OpenAI

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from creativegainbench.eval.mas_agents import run_triad_for_prompt
    from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver

    model_key = f"pcv:{base_model}"
    gen = cfg["generation"]
    client = _OpenAI(
        base_url=gen["ollama_base"],
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )
    receiver = None
    if score_edge_cue:
        receiver = CUEBeliefReceiver(
            CUEBeliefConfig(provider="ollama", model=base_model)
        )
    with psycopg.connect(cfg["db_url"]) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.prompt, p.domain_cluster
            FROM poems p
            WHERE p.split = %s
              AND p.prompt IS NOT NULL AND btrim(p.prompt) <> ''
              AND NOT EXISTS (
                SELECT 1 FROM generations g
                WHERE g.poem_id = p.id AND g.model = %s
              )
            ORDER BY p.id
            """,
            (cfg["split"], model_key),
        ).fetchall()
    if limit is not None:
        rows = rows[:limit]
    print(f"[{model_key}] need {len(rows)} MAS generations", flush=True)
    if not rows:
        return
    with psycopg.connect(cfg["db_url"], autocommit=True) as conn:
        for i, (pid, prompt, domain) in enumerate(rows, 1):
            try:
                row = run_triad_for_prompt(
                    client,
                    client,
                    client,
                    prompt,
                    base_model,
                    base_model,
                    base_model,
                    domain="creative_writing",
                    max_revision_rounds=1,
                    receiver=receiver,
                )
                conn.execute(
                    """
                    INSERT INTO generations (poem_id, model, output, max_tokens, temperature)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (poem_id, model) DO UPDATE SET output = EXCLUDED.output
                    """,
                    (
                        pid,
                        model_key,
                        json.dumps(row),
                        int(gen["max_tokens"]),
                        float(gen["temperature"]),
                    ),
                )
            except Exception as e:
                print(f"[warn] {model_key} {pid}: {e}", flush=True)
            if i % 10 == 0:
                print(f"[{model_key}] {i}/{len(rows)}", flush=True)
    print(f"[{model_key}] DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Single model; default=all in config")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["single", "mas"],
        default="single",
        help="single=per-model Ollama; mas=PCV triad under pcv:<model>",
    )
    parser.add_argument(
        "--score-edge-cue",
        action="store_true",
        help="With --mode mas, compute Edge-CUE during generation",
    )
    args = parser.parse_args()

    cfg = load_config()
    models = [args.model] if args.model else list(cfg["generation"]["models"])
    if args.mode == "mas":
        for model in models:
            generate_mas_model(
                cfg, model, args.limit, score_edge_cue=args.score_edge_cue
            )
    else:
        for model in models:
            generate_model(cfg, model, args.limit)
    print("DONE 02_generate")


if __name__ == "__main__":
    main()
