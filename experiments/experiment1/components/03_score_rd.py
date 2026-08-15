"""
Score λ_D-normalized true deformation R_D for human + each model on eval split.
Uses frozen poetry_v2 artifacts. Writes JSONL per model and scores table rows.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import psycopg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import REPO_ROOT, load_config  # noqa: E402

# Package imports need repo on path
sys.path.insert(0, str(REPO_ROOT / "src"))

from creativegainbench.ideas.idea_extractor import (  # noqa: E402
    IdeaBoundaryDetector,
    poetry_line_splitter,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.deformation import compute_deformation  # noqa: E402
from creativegainbench.metrics.feasibility import feasibility_bit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-human", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts = cfg["artifacts"]
    meta = json.loads((artifacts / "meta.json").read_text())
    from creativegainbench.metrics.delta_d import load_delta_d_thresholds, resolve_delta_d

    thr_path = artifacts / "delta_d_thresholds.json"
    if not thr_path.exists():
        # Fall back to construct_validity export if poetry_v2 copy missing.
        cv_thr = (
            Path(__file__).resolve().parents[2]
            / "construct_validity"
            / "results"
            / "delta_d_thresholds.json"
        )
        if cv_thr.exists():
            thr_path = cv_thr
        else:
            raise SystemExit(
                f"Missing δ_D thresholds at {artifacts / 'delta_d_thresholds.json'} "
                f"(and {cv_thr}). Run construct_validity calibrate_delta_d first."
            )
    thresholds = load_delta_d_thresholds(thr_path)
    delta_d_map = {
        int(k): resolve_delta_d(thresholds, k) for k in thresholds if str(k).isdigit()
    }

    codebook_state = torch.load(
        artifacts / "idea_codebook.pt", map_location="cpu", weights_only=True
    )
    codebook = IdeaCodebook(centroids=codebook_state["centroids"])
    encoder = build_span_encoder("minilm", device=args.device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(artifacts / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()
    if args.device != "cpu":
        boundary = boundary.to(args.device)

    domain_ctx = {}
    for d_str in meta["domains"]:
        path = artifacts / f"domain_{int(d_str)}_ctx.pkl"
        if path.exists():
            with open(path, "rb") as f:
                domain_ctx[int(d_str)] = pickle.load(f)

    models = [args.model] if args.model else list(cfg["generation"]["models"])
    max_chars = int(cfg["max_chars"])
    metric_version = cfg["metric_version"]
    results_dir = cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(cfg["db_url"]) as conn:
        poems = conn.execute(
            """
            SELECT id, title, author, source, prompt, body, domain_cluster, split
            FROM poems
            WHERE split = %s AND domain_cluster IS NOT NULL
            ORDER BY id
            """,
            (cfg["split"],),
        ).fetchall()
        gens = {}
        for model in models:
            rows = conn.execute(
                """
                SELECT g.poem_id, g.output FROM generations g
                JOIN poems p ON p.id = g.poem_id
                WHERE g.model = %s AND p.split = %s
                """,
                (model, cfg["split"]),
            ).fetchall()
            gens[model] = {str(pid): out for pid, out in rows}
            print(f"loaded generations[{model}]={len(gens[model])}", flush=True)

    # F10 length protocol: clip to per-domain eval median chars (≤ max_chars)
    # so R_D is not dominated by length outliers.
    by_domain_lens: dict[int, list[int]] = defaultdict(list)
    for _pid, _t, _a, _s, _p, body, domain, _sp in poems:
        if domain is None:
            continue
        by_domain_lens[int(domain)].append(len((body or "").strip()))
    domain_clip: dict[int, int] = {}
    for d, lens in by_domain_lens.items():
        med = int(statistics.median(lens)) if lens else max_chars
        domain_clip[d] = max(200, min(max_chars, med))
    print(f"domain length clips (median eval chars): {domain_clip}", flush=True)

    def _clip(text: str, d: int) -> str:
        return (text or "").strip()[: domain_clip.get(d, max_chars)]

    # Human R_D cache path
    human_cache_path = results_dir / "rd_human.json"
    human_cache: dict[str, dict] = {}
    if human_cache_path.exists() and args.skip_human:
        human_cache = {
            r["id"]: r["human"]
            for r in (json.loads(l) for l in human_cache_path.read_text().splitlines() if l.strip())
        }
        print(f"loaded human R_D cache n={len(human_cache)}", flush=True)

    if not human_cache:
        print(f"scoring human R_D on {len(poems)} eval poems...", flush=True)
        start = time.time()
        human_rows = []
        for i, (pid, title, author, source, prompt, body, domain, split) in enumerate(poems):
            sid = str(pid)
            d = int(domain)
            if d not in domain_ctx:
                continue
            thr = delta_d_map.get(d, 0.0)
            text = _clip(body, d)
            rd = compute_deformation(
                text,
                domain_ctx[d],
                span_encoder=encoder,
                codebook=codebook,
                boundary_detector=boundary,
                sentence_splitter=poetry_line_splitter,
            )
            feasible = feasibility_bit(text, prompt)
            human = {
                "r_d_raw": rd.r_d_raw,
                "r_d_norm": rd.r_d_norm,
                "feasibility_bit": feasible,
                "r_d_gate": 1.0 if (feasible and rd.r_d_norm > thr) else 0.0,
                "y_n_symbols": rd.y_n_symbols,
                "delta_d_norm": thr,
                "length_clip_chars": domain_clip.get(d),
            }
            human_cache[sid] = human
            human_rows.append(
                {
                    "id": sid,
                    "title": title,
                    "author": author,
                    "source": source,
                    "domain_cluster": d,
                    "split": split,
                    "human": human,
                }
            )
            if (i + 1) % 50 == 0:
                print(f"  human {i+1}/{len(poems)}", flush=True)
        with open(human_cache_path, "w") as f:
            for r in human_rows:
                f.write(json.dumps(r) + "\n")
        print(f"human R_D done in {(time.time()-start)/60:.1f}m → {human_cache_path}", flush=True)

        with psycopg.connect(cfg["db_url"]) as conn:
            with conn.cursor() as cur:
                for sid, human in human_cache.items():
                    cur.execute(
                        """
                        INSERT INTO scores (poem_id, side, metric_version, payload)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (poem_id, side, metric_version)
                        DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
                        """,
                        (sid, "human", metric_version, json.dumps(human)),
                    )
            conn.commit()

    for model in models:
        out_path = results_dir / f"rd_{model.replace(':', '_').replace('/', '_')}.jsonl"
        print(f"scoring R_D for {model}...", flush=True)
        start = time.time()
        n_done = 0
        with open(out_path, "w") as fout, psycopg.connect(cfg["db_url"]) as conn:
            for pid, title, author, source, prompt, body, domain, split in poems:
                sid = str(pid)
                if sid not in human_cache or sid not in gens[model]:
                    continue
                d = int(domain)
                if d not in domain_ctx:
                    continue
                thr = delta_d_map.get(d, 0.0)
                text = _clip(gens[model][sid], d)
                rd = compute_deformation(
                    text,
                    domain_ctx[d],
                    span_encoder=encoder,
                    codebook=codebook,
                    boundary_detector=boundary,
                    sentence_splitter=poetry_line_splitter,
                )
                feasible = feasibility_bit(text, prompt)
                llm = {
                    "r_d_raw": rd.r_d_raw,
                    "r_d_norm": rd.r_d_norm,
                    "feasibility_bit": feasible,
                    "r_d_gate": 1.0 if (feasible and rd.r_d_norm > thr) else 0.0,
                    "y_n_symbols": rd.y_n_symbols,
                    "delta_d_norm": thr,
                    "model": model,
                    "length_clip_chars": domain_clip.get(d),
                }
                row = {
                    "id": sid,
                    "title": title,
                    "author": author,
                    "source": source,
                    "domain_cluster": d,
                    "split": split,
                    "model": model,
                    "human": human_cache[sid],
                    "llm": llm,
                }
                fout.write(json.dumps(row) + "\n")
                conn.execute(
                    """
                    INSERT INTO scores (poem_id, side, metric_version, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (poem_id, side, metric_version)
                    DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
                    """,
                    (sid, model, metric_version, json.dumps(llm)),
                )
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  [{model}] {n_done}", flush=True)
            conn.commit()
        print(
            f"[{model}] scored {n_done} in {(time.time()-start)/60:.1f}m → {out_path}",
            flush=True,
        )
    print("DONE 03_score_rd")


if __name__ == "__main__":
    main()
