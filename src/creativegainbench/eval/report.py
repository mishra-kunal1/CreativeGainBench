"""
Aggregate R_creativity score reports (mean / std / CI / gate pass rates).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.stdev(xs))


def _ci95(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = _mean(xs)
    if len(xs) < 2:
        return m, m
    se = _std(xs) / math.sqrt(len(xs))
    return m - 1.96 * se, m + 1.96 * se


def aggregate(rows: list[dict]) -> dict:
    scores = [float(r["score"]) for r in rows]
    cues = [float(r["cue"]) for r in rows]
    rds = [float(r["r_d"]) for r in rows]
    rbs = [float(r["r_b"]) for r in rows]
    gks = [float(r.get("g_k", 0.0)) for r in rows]
    cue_pass = sum(1 for r in rows if float(r["cue_gate"]) > 0) / max(len(rows), 1)
    d_pass = sum(1 for r in rows if float(r["d_gate"]) > 0) / max(len(rows), 1)
    both = sum(
        1 for r in rows if float(r["cue_gate"]) > 0 and float(r["d_gate"]) > 0
    ) / max(len(rows), 1)
    lo, hi = _ci95(scores)

    by_domain: dict[str, dict] = {}
    domains = {r.get("domain") for r in rows if r.get("domain")}
    for d in sorted(x for x in domains if x):
        sub = [r for r in rows if r.get("domain") == d]
        by_domain[d] = {
            "n": len(sub),
            "mean_score": _mean([float(r["score"]) for r in sub]),
            "gate_pass_rate": sum(
                1
                for r in sub
                if float(r["cue_gate"]) > 0 and float(r["d_gate"]) > 0
            )
            / max(len(sub), 1),
        }

    return {
        "n": len(rows),
        "mean_score": _mean(scores),
        "std_score": _std(scores),
        "ci95_score": [lo, hi],
        "mean_cue": _mean(cues),
        "mean_r_d": _mean(rds),
        "mean_r_b": _mean(rbs),
        "mean_g_k": _mean(gks),
        "cue_gate_pass_rate": cue_pass,
        "d_gate_pass_rate": d_pass,
        "both_gates_pass_rate": both,
        "by_domain": by_domain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark score JSONL")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: alongside --scores)",
    )
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.scores.read_text().splitlines() if l.strip()]
    summary = aggregate(rows)
    out = args.output or args.scores.with_suffix(".summary.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary → {out}")


if __name__ == "__main__":
    main()
