"""
Poetry-v2 human-vs-LLM experiment.

Primary: λ_D-normalized true deformation R_D on all eval-split pairs.
Secondary: CUE with external z* (exemplar annotator) on a stratified subsample.
R_B: diagnostic only (optional; off by default — saturated in v1).

Usage:
  python scripts/score_poems_v2.py
  python scripts/score_poems_v2.py --cue-sample 400 --with-rb
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import psycopg
import torch

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector, poetry_line_splitter
from creativegainbench.ideas.idea_ngram import IdeaCodebook
from creativegainbench.ideas.span_encoder import build_span_encoder
from creativegainbench.metrics.cue import cue_gate
from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
from creativegainbench.metrics.deformation import compute_deformation
from creativegainbench.metrics.feasibility import feasibility_bit
from creativegainbench.metrics.outcome_annotator import OutcomeAnnotator

DEFAULT_DB = "postgres://poems:poems@localhost:5432/poems"
ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "creativegainbench"
    / "artifacts"
    / "poetry_v2"
)
METRIC_VERSION = "poetry_v2"
MAX_CHARS = 4000


def _bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = 42):
    if not diffs:
        return 0.0, (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return sum(diffs) / n, (lo, hi)


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_DB))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/poems_v2_human_vs_llm.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/evaluation/poems_v2_human_vs_llm.summary.json"))
    parser.add_argument("--cue-sample", type=int, default=400, help="Pairs for CUE (0=skip)")
    parser.add_argument("--cue-model", default="gemma2:2b")
    parser.add_argument("--with-rb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="eval", help="Which split to score (default: eval)")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    from creativegainbench.metrics.delta_d import load_delta_d_thresholds, resolve_delta_d

    thr_path = ARTIFACTS / "delta_d_thresholds.json"
    if not thr_path.exists():
        raise SystemExit(
            f"Missing {thr_path}. Run construct_validity calibrate_delta_d "
            "or copy negative-bank thresholds into poetry_v2."
        )
    thresholds = load_delta_d_thresholds(thr_path)
    delta_d_map = {int(k): resolve_delta_d(thresholds, k) for k in thresholds if k.isdigit()}

    codebook_state = torch.load(ARTIFACTS / "idea_codebook.pt", map_location="cpu", weights_only=True)
    codebook = IdeaCodebook(centroids=codebook_state["centroids"])
    encoder = build_span_encoder("minilm", device=args.device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(ARTIFACTS / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()
    if args.device != "cpu":
        boundary = boundary.to(args.device)

    domain_ctx = {}
    for d_str in meta["domains"]:
        path = ARTIFACTS / f"domain_{int(d_str)}_ctx.pkl"
        if path.exists():
            with open(path, "rb") as f:
                domain_ctx[int(d_str)] = pickle.load(f)

    with psycopg.connect(args.db) as conn:
        rows = conn.execute(
            "SELECT id, title, author, source, prompt, body, llm_output, "
            "domain_cluster, split FROM poems "
            "WHERE split = %s AND domain_cluster IS NOT NULL "
            "AND prompt IS NOT NULL AND llm_output IS NOT NULL "
            "ORDER BY id",
            (args.split,),
        ).fetchall()
    print(f"{len(rows)} {args.split} poems across {len(domain_ctx)} domains", flush=True)

    # --- Primary: R_D on all pairs ---
    results: dict[str, dict] = {}
    start = time.time()
    for i, (pid, title, author, source, prompt, body, llm_out, domain, split) in enumerate(rows):
        sid = str(pid)
        d = int(domain)
        if d not in domain_ctx:
            continue
        ctx = domain_ctx[d]
        human_rd = compute_deformation(
            body.strip()[:MAX_CHARS],
            ctx,
            span_encoder=encoder,
            codebook=codebook,
            boundary_detector=boundary,
            sentence_splitter=poetry_line_splitter,
        )
        llm_rd = compute_deformation(
            llm_out.strip()[:MAX_CHARS],
            ctx,
            span_encoder=encoder,
            codebook=codebook,
            boundary_detector=boundary,
            sentence_splitter=poetry_line_splitter,
        )
        thr = delta_d_map.get(d, 0.0)
        h_text = body.strip()[:MAX_CHARS]
        l_text = llm_out.strip()[:MAX_CHARS]
        h_feas = feasibility_bit(h_text, prompt)
        l_feas = feasibility_bit(l_text, prompt)
        results[sid] = {
            "id": sid,
            "title": title,
            "author": author,
            "source": source,
            "domain_cluster": d,
            "split": split,
            "human": {
                "r_d_raw": human_rd.r_d_raw,
                "r_d_norm": human_rd.r_d_norm,
                "feasibility_bit": h_feas,
                "r_d_gate": 1.0 if (h_feas and human_rd.r_d_norm > thr) else 0.0,
                "y_n_symbols": human_rd.y_n_symbols,
                "delta_d_norm": thr,
            },
            "llm": {
                "r_d_raw": llm_rd.r_d_raw,
                "r_d_norm": llm_rd.r_d_norm,
                "feasibility_bit": l_feas,
                "r_d_gate": 1.0 if (l_feas and llm_rd.r_d_norm > thr) else 0.0,
                "y_n_symbols": llm_rd.y_n_symbols,
                "delta_d_norm": thr,
            },
        }
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            rate = (i + 1) / max(time.time() - start, 1e-6)
            print(f"R_D {i+1}/{len(rows)} ({rate:.2f}/s)", flush=True)

    # --- Secondary: CUE on subsample ---
    cue_ids: list[str] = []
    if args.cue_sample > 0:
        by_domain: dict[int, list[str]] = defaultdict(list)
        for sid, rec in results.items():
            by_domain[rec["domain_cluster"]].append(sid)
        rng = random.Random(args.seed)
        # Stratified sample.
        per = max(1, args.cue_sample // max(len(by_domain), 1))
        for d, ids in sorted(by_domain.items()):
            cue_ids.extend(rng.sample(ids, min(per, len(ids))))
        cue_ids = cue_ids[: args.cue_sample]
        print(f"CUE subsample: {len(cue_ids)} pairs", flush=True)

        annotator = OutcomeAnnotator(span_encoder=encoder).fit()
        cue_rx = CUEBeliefReceiver(
            CUEBeliefConfig(provider="ollama", model=args.cue_model)
        )
        id_to_row = {str(r[0]): r for r in rows}
        for j, sid in enumerate(cue_ids):
            row = id_to_row[sid]
            prompt, body, llm_out = row[4], row[5], row[6]
            for side, text in (("human", body), ("llm", llm_out)):
                text = text.strip()[:MAX_CHARS]
                z_star = annotator.annotate(text)
                try:
                    cue_val, _model, diag = cue_rx.compute_cue_for_output(
                        prompt, text, external_outcome_index=z_star
                    )
                except Exception as e:
                    print(f"[warn] CUE failed {sid}/{side}: {e}", flush=True)
                    cue_val, diag = None, {"outcome_label": "error", "error": str(e)}
                results[sid][side]["cue"] = None if cue_val is None else float(cue_val)
                results[sid][side]["cue_gate"] = (
                    0.0 if cue_val is None else float(cue_gate(cue_val))
                )
                results[sid][side]["outcome_label"] = diag.get("outcome_label")
                results[sid][side]["outcome_source"] = diag.get("outcome_source")
                results[sid][side]["z_star_source"] = diag.get("z_star_source")
                results[sid][side]["brier_delta"] = diag.get("brier_delta")
                results[sid][side]["parse_ok"] = diag.get("parse_ok")
                # Composite: both gates × CUE (α R_B dropped — diagnostic)
                rd_gate = results[sid][side]["r_d_gate"]
                cg = results[sid][side]["cue_gate"]
                results[sid][side]["r_creativity"] = (
                    0.0 if cue_val is None else float(cg * rd_gate * cue_val)
                )
            if (j + 1) % 10 == 0 or j + 1 == len(cue_ids):
                print(f"CUE {j+1}/{len(cue_ids)}", flush=True)

    # Persist JSONL + DB scores.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        for sid in sorted(results.keys()):
            fout.write(json.dumps(results[sid]) + "\n")

    with psycopg.connect(args.db) as conn:
        with conn.cursor() as cur:
            for sid, rec in results.items():
                for side in ("human", "llm"):
                    cur.execute(
                        """
                        INSERT INTO scores (poem_id, side, metric_version, payload)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (poem_id, side, metric_version)
                        DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
                        """,
                        (sid, side, METRIC_VERSION, json.dumps(rec[side])),
                    )
        conn.commit()

    # --- Summary stats ---
    h_rd = [r["human"]["r_d_norm"] for r in results.values()]
    l_rd = [r["llm"]["r_d_norm"] for r in results.values()]
    diffs = [a - b for a, b in zip(h_rd, l_rd)]
    mean_diff, ci = _bootstrap_ci(diffs, seed=args.seed)
    h_wins = sum(1 for d in diffs if d > 1e-12)
    l_wins = sum(1 for d in diffs if d < -1e-12)
    ties = len(diffs) - h_wins - l_wins

    def _tail_share(xs, thr):
        return sum(1 for x in xs if x > thr) / max(len(xs), 1)

    # Tail threshold: pooled q90
    pooled = sorted(h_rd + l_rd)
    q90 = _quantile(pooled, 0.90)

    summary = {
        "metric_version": METRIC_VERSION,
        "n_pairs": len(results),
        "split": args.split,
        "r_d": {
            "human_mean": sum(h_rd) / max(len(h_rd), 1),
            "llm_mean": sum(l_rd) / max(len(l_rd), 1),
            "human_median": _quantile(h_rd, 0.5),
            "llm_median": _quantile(l_rd, 0.5),
            "human_q75": _quantile(h_rd, 0.75),
            "llm_q75": _quantile(l_rd, 0.75),
            "human_q90": _quantile(h_rd, 0.90),
            "llm_q90": _quantile(l_rd, 0.90),
            "human_q95": _quantile(h_rd, 0.95),
            "llm_q95": _quantile(l_rd, 0.95),
            "mean_diff_human_minus_llm": mean_diff,
            "ci95": list(ci),
            "significant": bool(ci[0] > 0 or ci[1] < 0),
            "human_wins": h_wins,
            "llm_wins": l_wins,
            "ties": ties,
            "tail_threshold_pooled_q90": q90,
            "human_tail_share": _tail_share(h_rd, q90),
            "llm_tail_share": _tail_share(l_rd, q90),
            "human_gate_rate": sum(r["human"]["r_d_gate"] for r in results.values()) / max(len(results), 1),
            "llm_gate_rate": sum(r["llm"]["r_d_gate"] for r in results.values()) / max(len(results), 1),
            "human_mean_symbols": sum(r["human"]["y_n_symbols"] for r in results.values()) / max(len(results), 1),
            "llm_mean_symbols": sum(r["llm"]["y_n_symbols"] for r in results.values()) / max(len(results), 1),
        },
    }

    cue_pairs = [r for r in results.values() if "cue" in r["human"]]
    if cue_pairs:
        h_cue = [r["human"]["cue"] for r in cue_pairs]
        l_cue = [r["llm"]["cue"] for r in cue_pairs]
        cue_diffs = [a - b for a, b in zip(h_cue, l_cue)]
        cue_mean, cue_ci = _bootstrap_ci(cue_diffs, seed=args.seed)
        both_h = sum(
            1 for r in cue_pairs
            if r["human"].get("cue_gate", 0) and r["human"].get("r_d_gate", 0)
        )
        both_l = sum(
            1 for r in cue_pairs
            if r["llm"].get("cue_gate", 0) and r["llm"].get("r_d_gate", 0)
        )
        from collections import Counter as C

        summary["cue"] = {
            "n": len(cue_pairs),
            "human_mean": sum(h_cue) / len(h_cue),
            "llm_mean": sum(l_cue) / len(l_cue),
            "mean_diff": cue_mean,
            "ci95": list(cue_ci),
            "significant": bool(cue_ci[0] > 0 or cue_ci[1] < 0),
            "human_gate_rate": sum(r["human"]["cue_gate"] for r in cue_pairs) / len(cue_pairs),
            "llm_gate_rate": sum(r["llm"]["cue_gate"] for r in cue_pairs) / len(cue_pairs),
            "human_both_gates_rate": both_h / len(cue_pairs),
            "llm_both_gates_rate": both_l / len(cue_pairs),
            "outcome_human": dict(C(r["human"].get("outcome_label") for r in cue_pairs)),
            "outcome_llm": dict(C(r["llm"].get("outcome_label") for r in cue_pairs)),
            "r_creativity_human_mean": sum(r["human"].get("r_creativity", 0) for r in cue_pairs) / len(cue_pairs),
            "r_creativity_llm_mean": sum(r["llm"].get("r_creativity", 0) for r in cue_pairs) / len(cue_pairs),
        }

    # Per-domain R_D
    by_d: dict[int, list] = defaultdict(list)
    for r in results.values():
        by_d[r["domain_cluster"]].append(r)
    summary["by_domain"] = {}
    for d, recs in sorted(by_d.items()):
        hd = [x["human"]["r_d_norm"] for x in recs]
        ld = [x["llm"]["r_d_norm"] for x in recs]
        dd = [a - b for a, b in zip(hd, ld)]
        md, dci = _bootstrap_ci(dd, seed=args.seed)
        summary["by_domain"][str(d)] = {
            "n": len(recs),
            "human_mean": sum(hd) / len(hd),
            "llm_mean": sum(ld) / len(ld),
            "mean_diff": md,
            "ci95": list(dci),
            "human_wins": sum(1 for x in dd if x > 0),
            "llm_wins": sum(1 for x in dd if x < 0),
        }

    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["r_d"], indent=2))
    if "cue" in summary:
        print(json.dumps(summary["cue"], indent=2))
    print(f"wrote {args.output} and {args.summary}", flush=True)


if __name__ == "__main__":
    main()
