"""
CUE with external z* on a frozen stratified subsample, for each model.
Same poem IDs across models; belief receiver fixed (config scoring.cue_model).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import psycopg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import REPO_ROOT, load_config  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.cue import cue_gate  # noqa: E402
from creativegainbench.metrics.cue_receiver import (  # noqa: E402
    CUEBeliefConfig,
    CUEBeliefReceiver,
)
from creativegainbench.metrics.outcome_annotator import OutcomeAnnotator  # noqa: E402


def _pick_ids(cfg: dict) -> list[str]:
    """Freeze subsample IDs to results/cue_ids.json (create once)."""
    results_dir = cfg["results_dir"]
    ids_path = results_dir / "cue_ids.json"
    if ids_path.exists():
        return json.loads(ids_path.read_text())

    with psycopg.connect(cfg["db_url"]) as conn:
        rows = conn.execute(
            """
            SELECT id, domain_cluster FROM poems
            WHERE split = %s AND domain_cluster IS NOT NULL
            ORDER BY id
            """,
            (cfg["split"],),
        ).fetchall()
    by_d: dict[int, list[str]] = defaultdict(list)
    for pid, d in rows:
        by_d[int(d)].append(str(pid))
    rng = random.Random(int(cfg["seed"]))
    n = int(cfg["scoring"]["cue_sample"])
    per = max(1, n // max(len(by_d), 1))
    ids: list[str] = []
    for d in sorted(by_d):
        ids.extend(rng.sample(by_d[d], min(per, len(by_d[d]))))
    ids = ids[:n]
    results_dir.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(json.dumps(ids, indent=2) + "\n")
    print(f"froze cue subsample n={len(ids)} → {ids_path}", flush=True)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ids = _pick_ids(cfg)
    models = [args.model] if args.model else list(cfg["generation"]["models"])
    max_chars = int(cfg["max_chars"])
    metric_version = cfg["metric_version"] + "_cue"
    results_dir = cfg["results_dir"]

    encoder = build_span_encoder("minilm", device=args.device)
    annotator = OutcomeAnnotator(span_encoder=encoder).fit()
    cue_rx = CUEBeliefReceiver(
        CUEBeliefConfig(provider="ollama", model=cfg["scoring"]["cue_model"])
    )

    with psycopg.connect(cfg["db_url"]) as conn:
        poem_map = {
            str(r[0]): r
            for r in conn.execute(
                "SELECT id, prompt, body FROM poems WHERE id = ANY(%s::uuid[])",
                (ids,),
            ).fetchall()
        }

    for model in models:
        out_path = results_dir / f"cue_{model.replace(':', '_').replace('/', '_')}.jsonl"
        # Resume: skip already scored ids
        done_ids: set[str] = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

        with psycopg.connect(cfg["db_url"]) as conn:
            gens = {
                str(pid): out
                for pid, out in conn.execute(
                    "SELECT poem_id, output FROM generations WHERE model = %s AND poem_id = ANY(%s::uuid[])",
                    (model, ids),
                ).fetchall()
            }

        print(f"[{model}] CUE {len(ids) - len(done_ids)} remaining / {len(ids)}", flush=True)
        start = time.time()
        n_new = 0
        with open(out_path, "a") as fout:
            for sid in ids:
                if sid in done_ids or sid not in poem_map or sid not in gens:
                    continue
                _, prompt, body = poem_map[sid]
                rec = {"id": sid, "model": model, "human": {}, "llm": {}}
                for side, text in (("human", body), ("llm", gens[sid])):
                    text = text.strip()[:max_chars]
                    z_star = annotator.annotate(text)
                    try:
                        cue_val, _m, diag = cue_rx.compute_cue_for_output(
                            prompt, text, external_outcome_index=z_star
                        )
                    except Exception as e:
                        print(f"[warn] CUE {model}/{sid}/{side}: {e}", flush=True)
                        cue_val, diag = None, {"outcome_label": "error", "error": str(e)}
                    rec[side] = {
                        "cue": None if cue_val is None else float(cue_val),
                        "cue_gate": 0.0 if cue_val is None else float(cue_gate(cue_val)),
                        "outcome_label": diag.get("outcome_label"),
                        "outcome_source": diag.get("outcome_source"),
                        "z_star_source": diag.get("z_star_source"),
                        "brier_delta": diag.get("brier_delta"),
                        "parse_ok": diag.get("parse_ok"),
                    }
                # Attach R_D gates from prior scores if present
                with psycopg.connect(cfg["db_url"]) as conn:
                    for side_key, db_side in (("human", "human"), ("llm", model)):
                        row = conn.execute(
                            """
                            SELECT payload FROM scores
                            WHERE poem_id = %s AND side = %s AND metric_version = %s
                            """,
                            (sid, db_side, cfg["metric_version"]),
                        ).fetchone()
                        if row:
                            payload = row[0]
                            if isinstance(payload, str):
                                payload = json.loads(payload)
                            rec[side_key]["r_d_norm"] = payload.get("r_d_norm")
                            rec[side_key]["r_d_gate"] = payload.get("r_d_gate")
                            cg = rec[side_key]["cue_gate"]
                            dg = float(payload.get("r_d_gate") or 0)
                            cue_side = rec[side_key]["cue"]
                            rec[side_key]["r_creativity"] = (
                                0.0
                                if cue_side is None
                                else float(cg * dg * cue_side)
                            )
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                # Persist CUE payloads with z_star_source for V7 stratification (F0.2).
                with psycopg.connect(cfg["db_url"]) as conn:
                    for side_key, db_side in (("human", "human"), ("llm", model)):
                        payload = dict(rec.get(side_key) or {})
                        payload["model"] = model
                        conn.execute(
                            """
                            INSERT INTO scores (poem_id, side, metric_version, payload)
                            VALUES (%s, %s, %s, %s::jsonb)
                            ON CONFLICT (poem_id, side, metric_version)
                            DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
                            """,
                            (
                                sid,
                                f"cue:{db_side}" if side_key == "llm" else "cue:human",
                                metric_version,
                                json.dumps(payload),
                            ),
                        )
                    conn.commit()
                n_new += 1
                if n_new % 10 == 0:
                    print(f"  [{model}] cue {n_new}", flush=True)
        print(
            f"[{model}] CUE wrote {n_new} in {(time.time()-start)/60:.1f}m → {out_path}",
            flush=True,
        )
    print("DONE 04_score_cue")


if __name__ == "__main__":
    main()
