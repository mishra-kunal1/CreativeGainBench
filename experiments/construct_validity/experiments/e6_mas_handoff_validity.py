"""
E6 — MAS handoff construct validity (HandoffGain / Edge-CUE).

Stratifies handoff_gain_rate by domain/model, checks length confounds on
Edge-CUE (mirror E3 length checks), and reports whether Critic/Verifier
roles show any construct signal before trusting them as creativity metrics.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import pearson  # noqa: E402
from db.connection import connect, run_migrations  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402
from lib import load_config  # noqa: E402


def main() -> None:
    run_migrations()
    cfg = load_config()
    run_id = new_run_id()
    metric_version = cfg.get("metric_version", "poetry_v2")

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.poem_id, s.side, s.payload, p.domain_cluster, length(p.body) AS blen
            FROM scores s
            JOIN poems p ON p.id = s.poem_id
            WHERE s.metric_version = %s
              AND s.side IN ('edge:critic_to_proposer', 'edge:handoff_gain')
            """,
            (metric_version,),
        ).fetchall()

    if not rows:
        record(
            run_id,
            "E6",
            "mas_scores_present",
            0.0,
            False,
            {"error": "no edge:* scores — run experiment1 MAS generate + 03b"},
        )
        print("DONE E6 (no data)")
        return

    record(run_id, "E6", "mas_scores_present", float(len(rows)), True, {"n": len(rows)})

    by_domain: dict[int, list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    cues: list[float] = []
    lengths: list[float] = []

    for _pid, side, payload, domain, blen in rows:
        if isinstance(payload, str):
            payload = json.loads(payload)
        hg = payload.get("handoff_gain_rate")
        if hg is None:
            continue
        hg = float(hg)
        if domain is not None:
            by_domain[int(domain)].append(hg)
        model = str(payload.get("model", "?"))
        by_model[model].append(hg)
        if side == "edge:critic_to_proposer" and payload.get("mean_edge_cue") is not None:
            cues.append(float(payload["mean_edge_cue"]))
            lengths.append(float(blen or 0))

    domain_means = {str(d): (sum(v) / len(v) if v else None) for d, v in by_domain.items()}
    model_means = {m: (sum(v) / len(v) if v else None) for m, v in by_model.items()}
    record(
        run_id,
        "E6",
        "handoff_by_domain",
        None,
        None,
        {"means": domain_means, "n_domains": len(domain_means)},
    )
    record(
        run_id,
        "E6",
        "handoff_by_model",
        None,
        None,
        {"means": model_means},
    )

    if len(cues) >= 10:
        r = pearson(cues, lengths)
        # Fail if |corr| with length is extreme (same spirit as R_D length check).
        passed = abs(r) < 0.5
        record(
            run_id,
            "E6",
            "edge_cue_length_corr",
            float(r),
            passed,
            {"n": len(cues), "threshold_abs": 0.5},
        )
    else:
        record(
            run_id,
            "E6",
            "edge_cue_length_corr",
            None,
            None,
            {"error": "insufficient pairs", "n": len(cues)},
        )

    # Role signal: mean gate rate on non-diagnostic edges > 0 somewhere.
    any_signal = any((v or 0) > 0 for v in model_means.values())
    record(
        run_id,
        "E6",
        "critic_role_signal",
        1.0 if any_signal else 0.0,
        any_signal,
        {"model_means": model_means},
    )
    print("DONE E6")


if __name__ == "__main__":
    main()
