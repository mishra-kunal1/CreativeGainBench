"""Render markdown summary table from experiment_results (latest run per check)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.connection import connect, run_migrations  # noqa: E402
from lib import load_config  # noqa: E402

ORDER = [
    ("E0", "orphan_scores_count", "Orphan scores", "0 rows"),
    ("E0", "min_cell_n", "Min domain×model n", "≥30"),
    ("E1", "provenance_negative_control", "Provenance negative control", "CI includes 0"),
    ("E1", "domain_ordering_consistency", "Domain ordering (0,2,9 vs 6,8)", "≥3/4 models"),
    ("E1", "legacy_vs_rerun_stability", "Legacy vs rerun stability", "exact≥0.95 or KS small"),
    ("E2", "exact_copy_invariance", "Exact-copy invariance", "≥95% under tol"),
    ("E2", "anti_padding", "Anti-padding", "≥95%"),
    ("E2", "monotonicity_sanity", "Monotonicity sanity", "≥95%"),
    ("E3", "negative_pass_rate", "Negative pass-rate", "≈5% at q=0.95"),
    ("E3", "eval_above_negative", "Eval above negative", "human > neg"),
    ("E3", "length_correlation", "Length correlation", "|r|<0.3"),
    ("E3", "zstar_stratified_joint_gate", "z*-stratified joint gate", "reported"),
    ("E4", "cue_rd_correlation", "CUE/R_D correlation", "0.1–0.5"),
    ("E5", "judge_agreement", "Judge agreement", "reported, no refit"),
]


def main() -> None:
    run_migrations()
    cfg = load_config()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (experiment, check_name)
              experiment, check_name, metric_value, passed, details, created_at
            FROM experiment_results
            ORDER BY experiment, check_name, created_at DESC
            """
        ).fetchall()
    latest = {(e, c): (mv, p, det) for e, c, mv, p, det, _ in rows}

    lines = [
        "# Construct validity report",
        "",
        "| Experiment | Check | Pass criterion | Status | Value |",
        "|---|---|---|---|---|",
    ]
    for exp, check, label, crit in ORDER:
        row = latest.get((exp, check))
        if not row:
            status = "pending"
            val = ""
        else:
            mv, passed, det = row
            if passed is True:
                status = "PASS"
            elif passed is False:
                status = "FAIL"
            else:
                status = "REPORT"
            val = "" if mv is None else f"{mv:.4g}"
        lines.append(f"| {exp} | {label} | {crit} | {status} | {val} |")

    out = Path(cfg["results_dir"]) / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
