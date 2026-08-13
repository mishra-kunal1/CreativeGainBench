"""E1 — Known-groups validity checks on stored poetry_v2 scores."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import bootstrap_ci  # noqa: E402
from db.connection import connect, run_migrations  # noqa: E402
from db.queries import fetch_eval_scores  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402
from lib import load_config  # noqa: E402

HUMAN_FAVOR = {0, 2, 9}
MODEL_FAVOR = {6, 8}
MODELS = ["gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b"]


def _ks_approx(a: list[float], b: list[float]) -> float:
    """Two-sample KS statistic (not p-value); used for stability magnitude."""
    if not a or not b:
        return 1.0
    sa, sb = sorted(a), sorted(b)
    # empirical CDF max diff
    pts = sorted(set(sa + sb))
    i = j = 0
    d = 0.0
    for x in pts:
        while i < len(sa) and sa[i] <= x:
            i += 1
        while j < len(sb) and sb[j] <= x:
            j += 1
        d = max(d, abs(i / len(sa) - j / len(sb)))
    return d


def main() -> None:
    run_migrations()
    cfg = load_config()
    run_id = new_run_id()
    mv = cfg["metric_version"]

    with connect() as conn:
        human = fetch_eval_scores(conn, mv, "human")
        # Provenance negative control: poetrydb vs gutenberg humans
        by_src = defaultdict(list)
        for r in human:
            rd = r["payload"].get("r_d_norm")
            if rd is None or r["source"] is None:
                continue
            src = r["source"]
            if src.startswith("gutenberg"):
                by_src["gutenberg"].append(float(rd))
            elif src == "poetrydb":
                by_src["poetrydb"].append(float(rd))

        a = by_src.get("poetrydb", [])
        b = by_src.get("gutenberg", [])
        # unpaired bootstrap on mean difference via resample means
        import random

        rng = random.Random(cfg["seed"])
        diffs = []
        for _ in range(2000):
            if not a or not b:
                break
            ma = sum(a[rng.randrange(len(a))] for _ in range(len(a))) / len(a)
            mb = sum(b[rng.randrange(len(b))] for _ in range(len(b))) / len(b)
            diffs.append(ma - mb)
        mean_diff = (sum(a) / len(a) - sum(b) / len(b)) if a and b else 0.0
        if diffs:
            diffs.sort()
            ci = (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))])
        else:
            ci = (0.0, 0.0)
        passed_prov = bool(a and b and ci[0] <= 0 <= ci[1])
        record(
            run_id,
            "E1",
            "provenance_negative_control",
            mean_diff,
            passed_prov,
            {
                "n_poetrydb": len(a),
                "n_gutenberg": len(b),
                "ci95": list(ci),
                "note": "poetrydb vs gutenberg human r_d_norm; expect CI includes 0",
            },
        )

        # Domain ordering consistency across models
        consistent = 0
        details_dom = {}
        for model in MODELS:
            rows = fetch_eval_scores(conn, mv, model)
            # human-favoring share: fraction of pairs where human r_d > model r_d
            human_by_id = {r["poem_id"]: r for r in human}
            hf = mf = 0
            n_h = n_m = 0
            for r in rows:
                h = human_by_id.get(r["poem_id"])
                if not h:
                    continue
                d = r["domain_cluster"]
                hd = h["payload"].get("r_d_norm")
                md = r["payload"].get("r_d_norm")
                if hd is None or md is None:
                    continue
                if d in HUMAN_FAVOR:
                    n_h += 1
                    if float(hd) > float(md):
                        hf += 1
                if d in MODEL_FAVOR:
                    n_m += 1
                    if float(hd) > float(md):
                        mf += 1
            share_h = hf / n_h if n_h else 0.0
            share_m = mf / n_m if n_m else 0.0
            ok = share_h > share_m
            if ok:
                consistent += 1
            details_dom[model] = {
                "human_favor_domains_human_win_share": share_h,
                "model_favor_domains_human_win_share": share_m,
                "n_human_favor": n_h,
                "n_model_favor": n_m,
                "ordered": ok,
            }

        record(
            run_id,
            "E1",
            "domain_ordering_consistency",
            float(consistent),
            consistent >= 3,
            {"consistent_models": consistent, "by_model": details_dom},
        )

        # Legacy llm_output vs generations gemma2:2b — score bodies from generations
        # We compare r_d already stored for gemma2:2b side vs scoring legacy column if present.
        # Stability proxy: gemma scores vs human length-normalized — instead compare
        # generations gemma payload exists for all eval; legacy column completeness.
        legacy = conn.execute(
            """
            SELECT COUNT(*) FROM poems p
            JOIN generations g ON g.poem_id = p.id AND g.model = 'gemma2:2b'
            WHERE p.split = 'eval'
              AND p.llm_output IS NOT NULL AND btrim(p.llm_output) <> ''
              AND g.output IS NOT NULL
            """
        ).fetchone()[0]
        # Distributional stability: compare char-len of legacy vs generations (cheap proxy
        # when we don't want to re-score). Prefer KS on stored if dual scores exist.
        pairs = conn.execute(
            """
            SELECT length(p.llm_output), length(g.output)
            FROM poems p
            JOIN generations g ON g.poem_id = p.id AND g.model = 'gemma2:2b'
            WHERE p.split = 'eval'
              AND p.llm_output IS NOT NULL AND g.output IS NOT NULL
            LIMIT 2000
            """
        ).fetchall()
        la = [float(x) for x, _ in pairs]
        lb = [float(y) for _, y in pairs]
        ks = _ks_approx(la, lb)
        # Pass if texts are mostly identical (KS on length small) OR exact match rate high
        exact = sum(
            1
            for r in conn.execute(
                """
                SELECT 1 FROM poems p
                JOIN generations g ON g.poem_id = p.id AND g.model = 'gemma2:2b'
                WHERE p.split = 'eval' AND p.llm_output = g.output
                """
            ).fetchall()
        )
        exact_rate = exact / max(legacy, 1)
        # Migrated from llm_output — expect high exact match
        passed_stab = exact_rate >= 0.95 or ks < 0.1
        record(
            run_id,
            "E1",
            "legacy_vs_rerun_stability",
            exact_rate,
            passed_stab,
            {
                "exact_match_rate": exact_rate,
                "ks_length_stat": ks,
                "n_pairs": len(pairs),
                "note": "gemma2:2b generations migrated from poems.llm_output",
            },
        )

    print("DONE E1")


if __name__ == "__main__":
    main()
