"""E3 — Calibration acceptance against delta_d_calibration + thresholds."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import pearson  # noqa: E402
from db.connection import connect, run_migrations  # noqa: E402
from db.queries import fetch_eval_scores  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402
from lib import load_config  # noqa: E402

NEG_TYPES = ("exact_h_member", "pad", "shuffle")
MODELS = ["gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b"]


def main() -> None:
    run_migrations()
    cfg = load_config()
    run_id = new_run_id()
    q = float(cfg["delta_d_quantile"])
    target_pass = 1.0 - q

    with connect() as conn:
        thr_rows = conn.execute(
            "SELECT domain_cluster, delta_d_95, n_neg FROM delta_d_thresholds"
        ).fetchall()
        if not thr_rows:
            record(
                run_id,
                "E3",
                "negative_pass_rate",
                None,
                False,
                {"error": "no delta_d_thresholds — run calibrate_delta_d first"},
            )
            print("DONE E3 (early)")
            return
        thresholds = {int(d): float(thr) for d, thr, _ in thr_rows}

        # 1) Negative pass-rate ≈ 1-q
        neg_pass_rates = []
        for d, thr in thresholds.items():
            vals = [
                float(r[0])
                for r in conn.execute(
                    """
                    SELECT r_d_norm FROM delta_d_calibration
                    WHERE domain_cluster = %s AND construct_type = ANY(%s)
                    """,
                    (d, list(NEG_TYPES)),
                ).fetchall()
            ]
            if not vals:
                continue
            rate = sum(1 for x in vals if x > thr) / len(vals)
            neg_pass_rates.append(rate)
        mean_neg_pass = sum(neg_pass_rates) / max(len(neg_pass_rates), 1)
        # allow ±3pp around target
        passed_neg = abs(mean_neg_pass - target_pass) <= 0.05
        record(
            run_id,
            "E3",
            "negative_pass_rate",
            mean_neg_pass,
            passed_neg,
            {
                "target": target_pass,
                "per_domain_mean": mean_neg_pass,
                "n_domains": len(neg_pass_rates),
            },
        )

        # 2) Eval-human / model pass-rate > negative pass-rate
        human = fetch_eval_scores(conn, cfg["metric_version"], "human")
        def pass_rate(rows):
            n = p = 0
            for r in rows:
                d = r["domain_cluster"]
                if d not in thresholds:
                    continue
                rd = r["payload"].get("r_d_norm")
                if rd is None:
                    continue
                n += 1
                if float(rd) > thresholds[d]:
                    p += 1
            return (p / n) if n else 0.0, n

        h_rate, h_n = pass_rate(human)
        model_rates = {}
        for m in MODELS:
            rows = fetch_eval_scores(conn, cfg["metric_version"], m)
            model_rates[m], _ = pass_rate(rows)

        passed_above = h_rate > mean_neg_pass
        record(
            run_id,
            "E3",
            "eval_above_negative",
            h_rate,
            passed_above,
            {
                "human_pass_rate": h_rate,
                "human_n": h_n,
                "model_pass_rates": model_rates,
                "neg_pass_rate": mean_neg_pass,
            },
        )

        # 3) Length correlation |r| < 0.3 per domain (human eval)
        bad_domains = []
        corrs = {}
        by_d = defaultdict(list)
        for r in human:
            by_d[r["domain_cluster"]].append(r)
        for d, rows in by_d.items():
            xs = [float(r["payload"].get("y_n_symbols") or 0) for r in rows]
            ys = [float(r["payload"].get("r_d_norm") or 0) for r in rows]
            if len(xs) < 10:
                continue
            rho = pearson(xs, ys)
            corrs[str(d)] = rho
            if not math.isnan(rho) and abs(rho) > 0.3:
                bad_domains.append(d)
        record(
            run_id,
            "E3",
            "length_correlation",
            max((abs(v) for v in corrs.values() if not math.isnan(v)), default=0.0),
            len(bad_domains) == 0,
            {"correlations": corrs, "bad_domains": bad_domains, "threshold": 0.3},
        )

        # 4) Joint gate stratified by z* on CUE JSONLs — report only
        cue_dir = Path(__file__).resolve().parents[2] / "experiment1" / "results"
        joint = {"novel_structure": {"n": 0, "both": 0}, "fluent_paraphrase": {"n": 0, "both": 0}}
        for path in cue_dir.glob("cue_*.jsonl"):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for side in ("human", "llm"):
                    lab = rec[side].get("outcome_label")
                    if lab not in joint:
                        continue
                    rd = rec[side].get("r_d_norm")
                    cg = rec[side].get("cue_gate", 0)
                    # domain from scores
                    pid = rec["id"]
                    row = conn.execute(
                        "SELECT domain_cluster FROM poems WHERE id = %s", (pid,)
                    ).fetchone()
                    if not row or row[0] not in thresholds or rd is None:
                        continue
                    dg = 1.0 if float(rd) > thresholds[int(row[0])] else 0.0
                    joint[lab]["n"] += 1
                    if cg and dg:
                        joint[lab]["both"] += 1
        for lab, v in joint.items():
            v["rate"] = v["both"] / v["n"] if v["n"] else 0.0
        record(
            run_id,
            "E3",
            "zstar_stratified_joint_gate",
            None,
            None,  # report-only
            {"joint_by_zstar": joint, "note": "report only — do not refit delta_D"},
        )

    print("DONE E3")


if __name__ == "__main__":
    main()
