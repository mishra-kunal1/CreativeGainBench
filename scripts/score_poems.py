"""
Score human-written poems (body) and LLM generations (llm_output) from the
poems Postgres DB with the canonical CreativeGainBench metrics
(CUE, R_D, R_B, gates, R_creativity), using the same frozen v1 artifacts and
calibrated Ollama CUE receiver as run-benchmark.

Writes one JSONL row per poem with `human` and `llm` metric dicts.

Usage:
  python scripts/score_poems.py --sample 300 --output data/evaluation/poems_human_vs_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import psycopg

from creativegainbench.benchmark_score import compute_r_creativity
from creativegainbench.ideas.artifacts import ARTIFACTS_ROOT, load_artifacts, load_config
from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
from creativegainbench.metrics.delta_d import load_delta_d_thresholds, resolve_delta_d
from creativegainbench.receivers.hash_receiver import HashReceiverAgent
from creativegainbench.receivers.ollama_receiver import OllamaReceiverAgent

DEFAULT_DB = "postgres://poems:poems@localhost:5432/poems"
# Both CUE (bits-normalized) and R_D are length-sensitive; LLM outputs are
# capped at 1024 tokens (~4k chars), so clip both sides to the same budget
# for a fair human-vs-LLM comparison.
MAX_TEXT_CHARS = 4000


def _stratified_sample(rows: list[tuple], k: int, seed: int) -> list[tuple]:
    """Sample ~k rows, proportionally by source (index 3), deterministic."""
    if k >= len(rows):
        return rows
    rng = random.Random(seed)
    by_source: dict[str, list[tuple]] = {}
    for r in rows:
        by_source.setdefault(r[3], []).append(r)
    picked: list[tuple] = []
    for source, group in sorted(by_source.items()):
        n = max(1, round(k * len(group) / len(rows)))
        picked.extend(rng.sample(group, min(n, len(group))))
    rng.shuffle(picked)
    return picked[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_DB))
    parser.add_argument("--sample", type=int, default=None, help="Stratified sample size (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--receiver", choices=["ollama", "hash"], default="ollama")
    parser.add_argument("--receiver-model", default="gemma2:2b")
    parser.add_argument("--cue-model", default="gemma2:2b")
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/poems_human_vs_llm.jsonl"))
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    args = parser.parse_args()

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"scoring on device={args.device}", flush=True)

    with psycopg.connect(args.db) as conn:
        rows = conn.execute(
            "SELECT id, title, author, source, prompt, body, llm_output FROM poems "
            "WHERE prompt IS NOT NULL AND btrim(prompt) <> '' "
            "AND llm_output IS NOT NULL AND btrim(llm_output) <> '' "
            "ORDER BY id"
        ).fetchall()
    print(f"{len(rows)} poems have both prompt and llm_output", flush=True)
    if args.sample:
        rows = _stratified_sample(rows, args.sample, args.seed)
        print(f"stratified sample: {len(rows)} poems (seed={args.seed})", flush=True)

    cfg = load_config()
    version = cfg["artifacts"]["version"]
    pipeline = load_artifacts(version=version, device=args.device)
    thr_path = ARTIFACTS_ROOT / f"delta_d_thresholds_{version}.json"
    if not thr_path.exists():
        thr_path = ARTIFACTS_ROOT / "delta_d_thresholds.json"
    thresholds = load_delta_d_thresholds(thr_path)
    delta_d = resolve_delta_d(thresholds, "default")
    score_cfg = cfg["score"]
    rx_cfg = cfg["receiver_expansion"]

    if args.receiver == "ollama":
        receiver = OllamaReceiverAgent(
            span_encoder=pipeline.span_encoder,
            boundary_detector=pipeline.boundary_detector,
            boundary_threshold=pipeline.boundary_threshold,
            model=args.receiver_model,
        )
    else:
        receiver = HashReceiverAgent(
            span_encoder=pipeline.span_encoder,
            boundary_detector=pipeline.boundary_detector,
            boundary_threshold=pipeline.boundary_threshold,
            seed=pipeline.seed,
        )

    cue_receiver = CUEBeliefReceiver(
        CUEBeliefConfig(provider="ollama", model=args.cue_model)
    )

    # Resume: skip poem ids already scored in the output file.
    scored_ids: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                scored_ids.add(json.loads(line)["id"])
        print(f"resuming: {len(scored_ids)} poems already scored", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _score_text(prompt: str, text: str) -> dict:
        text = text.strip()[:MAX_TEXT_CHARS]
        _cue_val, cue_model, cue_diag = cue_receiver.compute_cue_for_output(prompt, text)
        if cue_model is None:
            raise RuntimeError(
                f"CUE parse failed: {cue_diag.get('cue_missing_reason') or cue_diag}"
            )
        result = compute_r_creativity(
            text,
            pipeline=pipeline,
            receiver=receiver,
            cue_model=cue_model,
            mas_outputs=None,
            alpha=float(score_cfg["alpha"]),
            lambda_g=float(score_cfg["lambda_g"]),
            delta_d=delta_d,
            n_samples=int(rx_cfg["n_samples"]),
            temperature=float(rx_cfg["temperature"]),
        )
        d = result.to_dict()
        d["outcome_label"] = cue_diag["outcome_label"]
        return d

    start = time.time()
    n_done = 0
    with open(args.output, "a") as fout:
        for poem_id, title, author, source, prompt, body, llm_output in rows:
            if str(poem_id) in scored_ids:
                continue
            try:
                human = _score_text(prompt, body)
                llm = _score_text(prompt, llm_output)
            except Exception as e:
                print(f"[warn] scoring failed for {poem_id} ({title!r}): {e}", flush=True)
                continue
            fout.write(
                json.dumps(
                    {
                        "id": str(poem_id),
                        "title": title,
                        "author": author,
                        "source": source,
                        "human": human,
                        "llm": llm,
                    }
                )
                + "\n"
            )
            fout.flush()
            n_done += 1
            if n_done % 5 == 0:
                rate = n_done / (time.time() - start)
                remaining = sum(1 for r in rows if str(r[0]) not in scored_ids) - n_done
                print(
                    f"scored {n_done} poems ({rate*60:.1f}/min, eta {remaining/rate/60:.0f} min)",
                    flush=True,
                )

    print(f"DONE: scored {n_done} poems -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
