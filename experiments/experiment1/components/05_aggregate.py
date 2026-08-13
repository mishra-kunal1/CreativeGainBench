"""
Aggregate R_D (+ optional CUE) across the model ladder into a summary JSON + REPORT.md.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import load_config  # noqa: E402


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
    return sum(diffs) / n, (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))]


def _safe_model_fname(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def summarize_rd(path: Path, seed: int) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    h = [r["human"]["r_d_norm"] for r in rows]
    l = [r["llm"]["r_d_norm"] for r in rows]
    diffs = [a - b for a, b in zip(h, l)]
    mean_diff, ci = _bootstrap_ci(diffs, seed=seed)
    pooled = sorted(h + l)
    q90 = _quantile(pooled, 0.90)
    by_d: dict[int, list] = defaultdict(list)
    for r in rows:
        by_d[r["domain_cluster"]].append(r)
    domains = {}
    for d, recs in sorted(by_d.items()):
        hd = [x["human"]["r_d_norm"] for x in recs]
        ld = [x["llm"]["r_d_norm"] for x in recs]
        dd = [a - b for a, b in zip(hd, ld)]
        md, dci = _bootstrap_ci(dd, seed=seed)
        domains[str(d)] = {
            "n": len(recs),
            "mean_diff": md,
            "ci95": list(dci),
            "significant_human": dci[0] > 0,
            "significant_llm": dci[1] < 0,
            "human_wins": sum(1 for x in dd if x > 0),
            "llm_wins": sum(1 for x in dd if x < 0),
        }
    return {
        "n": len(rows),
        "model": rows[0].get("model"),
        "human_mean": sum(h) / len(h),
        "llm_mean": sum(l) / len(l),
        "human_median": _quantile(h, 0.5),
        "llm_median": _quantile(l, 0.5),
        "human_q90": _quantile(h, 0.90),
        "llm_q90": _quantile(l, 0.90),
        "mean_diff": mean_diff,
        "ci95": list(ci),
        "significant": bool(ci[0] > 0 or ci[1] < 0),
        "human_wins": sum(1 for d in diffs if d > 0),
        "llm_wins": sum(1 for d in diffs if d < 0),
        "human_tail_share": sum(1 for x in h if x > q90) / len(h),
        "llm_tail_share": sum(1 for x in l if x > q90) / len(l),
        "human_mean_symbols": sum(r["human"]["y_n_symbols"] for r in rows) / len(rows),
        "llm_mean_symbols": sum(r["llm"]["y_n_symbols"] for r in rows) / len(rows),
        "by_domain": domains,
    }


def summarize_cue(path: Path, seed: int) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    h = [r["human"]["cue"] for r in rows]
    l = [r["llm"]["cue"] for r in rows]
    diffs = [a - b for a, b in zip(h, l)]
    mean_diff, ci = _bootstrap_ci(diffs, seed=seed)
    return {
        "n": len(rows),
        "model": rows[0].get("model"),
        "human_mean": sum(h) / len(h),
        "llm_mean": sum(l) / len(l),
        "mean_diff": mean_diff,
        "ci95": list(ci),
        "significant": bool(ci[0] > 0 or ci[1] < 0),
        "human_gate_rate": sum(r["human"]["cue_gate"] for r in rows) / len(rows),
        "llm_gate_rate": sum(r["llm"]["cue_gate"] for r in rows) / len(rows),
        "outcome_human": dict(Counter(r["human"].get("outcome_label") for r in rows)),
        "outcome_llm": dict(Counter(r["llm"].get("outcome_label") for r in rows)),
    }


def main() -> None:
    cfg = load_config()
    results_dir = cfg["results_dir"]
    seed = int(cfg["seed"])
    models = list(cfg["generation"]["models"])

    ladder = []
    for model in models:
        fname = _safe_model_fname(model)
        rd = summarize_rd(results_dir / f"rd_{fname}.jsonl", seed)
        cue = summarize_cue(results_dir / f"cue_{fname}.jsonl", seed)
        if rd is None and cue is None:
            continue
        ladder.append({"model": model, "r_d": rd, "cue": cue})

    summary = {
        "metric_version": cfg["metric_version"],
        "split": cfg["split"],
        "models": models,
        "ladder": ladder,
    }
    out_json = results_dir / "ladder_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Experiment 1 — multi-model ladder results",
        "",
        f"Frozen measurement: `{cfg['metric_version']}` artifacts. "
        f"Split: `{cfg['split']}`.",
        "",
        "## R_D (human − model), λ_D-normalized",
        "",
        "| Model | n | Human mean | Model mean | Δ | 95% CI | Human wins | Model wins | Tail human | Tail model |",
        "|-------|--:|-----------:|-----------:|--:|--------|----------:|-----------:|-----------:|-----------:|",
    ]
    for entry in ladder:
        rd = entry["r_d"]
        if not rd:
            continue
        ci = rd["ci95"]
        lines.append(
            f"| `{entry['model']}` | {rd['n']} | {rd['human_mean']:.3e} | "
            f"{rd['llm_mean']:.3e} | {rd['mean_diff']:+.3e} | "
            f"[{ci[0]:+.2e}, {ci[1]:+.2e}] | {rd['human_wins']} | {rd['llm_wins']} | "
            f"{rd['human_tail_share']:.1%} | {rd['llm_tail_share']:.1%} |"
        )

    lines += ["", "## CUE (external z*)", ""]
    lines.append(
        "| Model | n | Human mean | Model mean | Δ | Human gate | Model gate |"
    )
    lines.append("|-------|--:|-----------:|-----------:|--:|-----------:|-----------:|")
    for entry in ladder:
        cue = entry["cue"]
        if not cue:
            continue
        lines.append(
            f"| `{entry['model']}` | {cue['n']} | {cue['human_mean']:.3e} | "
            f"{cue['llm_mean']:.3e} | {cue['mean_diff']:+.3e} | "
            f"{cue['human_gate_rate']:.1%} | {cue['llm_gate_rate']:.1%} |"
        )

    lines += [
        "",
        "## Hypotheses",
        "",
        "- **H1:** Human−model R_D gap decreases with model tier.",
        "- **H2:** Model `novel_structure` / CUE gate rises with tier.",
        "- **H3:** Domains 0/2/9/10 stay hardest for weak models.",
        "",
        f"Machine-readable: `{out_json}`",
        "",
    ]
    report = results_dir / "REPORT.md"
    report.write_text("\n".join(lines))
    print(report.read_text())
    print(f"wrote {out_json} and {report}")
    print("DONE 05_aggregate")


if __name__ == "__main__":
    main()
