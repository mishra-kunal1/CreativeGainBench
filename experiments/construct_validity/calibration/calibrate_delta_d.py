"""
One-sided δ_D from negative bank quantiles — never fit to z* or human/model scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.connection import connect, run_migrations  # noqa: E402
from lib import load_config  # noqa: E402

from creativegainbench.metrics.delta_d import (  # noqa: E402
    quantile,
    write_delta_d_thresholds,
)

# Probe paraphrases often *raise* R_D (they reinforce P-overlapping n-grams),
# so they must not set the one-sided floor. Use constructs that should sit low.
NEG_TYPES = ("exact_h_member", "pad", "shuffle")
# Still scored/stored for diagnostics, but excluded from the quantile:
DIAGNOSTIC_TYPES = ("probe_paraphrase", "ood")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=float, default=None)
    args = parser.parse_args()

    run_migrations()
    cfg = load_config()
    q = float(args.quantile if args.quantile is not None else cfg["delta_d_quantile"])
    min_n = int(cfg["min_neg_per_domain"])

    thresholds = {}
    with connect() as conn:
        conn.execute("DELETE FROM delta_d_thresholds")
        domains = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT domain_cluster FROM delta_d_calibration ORDER BY 1"
            ).fetchall()
        ]
        for d in domains:
            vals = [
                float(r[0])
                for r in conn.execute(
                    """
                    SELECT r_d_norm FROM delta_d_calibration
                    WHERE domain_cluster = %s
                      AND construct_type = ANY(%s)
                      AND r_d_norm IS NOT NULL
                    """,
                    (d, list(NEG_TYPES)),
                ).fetchall()
            ]
            if len(vals) < min_n:
                print(
                    f"domain {d}: only {len(vals)} neg (need {min_n}) — skip",
                    flush=True,
                )
                continue
            thr = quantile(vals, q) + 1e-6
            conn.execute(
                """
                INSERT INTO delta_d_thresholds (domain_cluster, delta_d_95, n_neg, quantile)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (domain_cluster) DO UPDATE SET
                  delta_d_95 = EXCLUDED.delta_d_95,
                  n_neg = EXCLUDED.n_neg,
                  quantile = EXCLUDED.quantile,
                  created_at = now()
                """,
                (d, thr, len(vals), q),
            )
            by_type = {}
            for ctype in NEG_TYPES + DIAGNOSTIC_TYPES:
                tv = [
                    float(r[0])
                    for r in conn.execute(
                        """
                        SELECT r_d_norm FROM delta_d_calibration
                        WHERE domain_cluster = %s AND construct_type = %s
                        """,
                        (d, ctype),
                    ).fetchall()
                ]
                if tv:
                    by_type[ctype] = {
                        "n": len(tv),
                        "mean": sum(tv) / len(tv),
                        "pass_rate_at_thr": sum(1 for x in tv if x > thr) / len(tv),
                    }
            thresholds[str(d)] = {
                "delta_d_95": thr,
                "n_neg": len(vals),
                "quantile": q,
                "by_type": by_type,
            }
            print(
                f"domain {d}: δ_D={thr:.6g} n_neg={len(vals)} "
                f"exact_h_pass={by_type.get('exact_h_member', {}).get('pass_rate_at_thr')}",
                flush=True,
            )

    out = Path(cfg["results_dir"]) / "delta_d_thresholds.json"
    write_delta_d_thresholds(out, thresholds)
    # Also publish into poetry_v2 so experiment1 gates use the same contract.
    poetry_v2 = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "creativegainbench"
        / "artifacts"
        / "poetry_v2"
        / "delta_d_thresholds.json"
    )
    write_delta_d_thresholds(poetry_v2, thresholds)
    print(f"wrote {out}")
    print(f"wrote {poetry_v2}")
    print("DONE calibrate_delta_d")


if __name__ == "__main__":
    main()
