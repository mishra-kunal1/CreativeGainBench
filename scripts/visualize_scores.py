"""
Visualize human-vs-LLM CreativeGainBench metric scores produced by
scripts/score_poems.py.

Reads the paired JSONL (one row per poem with `human` and `llm` metric dicts)
and writes:
  - <output>.png : per-metric distribution comparison + mean deltas + gate rates
  - <output>.summary.json : aggregate stats used in the figure

Usage:
  python scripts/visualize_scores.py \
    --scores data/evaluation/poems_human_vs_llm.jsonl \
    --output data/evaluation/poems_human_vs_llm_metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ["score", "cue", "r_d", "r_b"]
METRIC_LABELS = {
    "score": "R_creativity (gated score)",
    "cue": "CUE (belief update / bit)",
    "r_d": "R_D (structural novelty, bits)",
    "r_b": "R_B (receiver expansion)",
    "cue_gate": "CUE gate pass rate",
    "d_gate": "R_D gate pass rate",
}
HUMAN_COLOR = "#4C72B0"
LLM_COLOR = "#DD8452"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("data/evaluation/poems_human_vs_llm.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/poems_human_vs_llm_metrics"))
    args = parser.parse_args()

    rows = load_rows(args.scores)
    if not rows:
        raise SystemExit(f"No rows in {args.scores}")
    n = len(rows)

    data = {
        side: {m: np.array([r[side][m] for r in rows], dtype=float) for m in METRICS + ["cue_gate", "d_gate"]}
        for side in ("human", "llm")
    }

    summary: dict = {"n_poems": n, "metrics": {}}
    for m in METRICS + ["cue_gate", "d_gate"]:
        h, l = data["human"][m], data["llm"][m]
        diff = h - l
        # Paired bootstrap 95% CI on the mean difference (human - llm).
        rng = np.random.default_rng(42)
        boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
        summary["metrics"][m] = {
            "human_mean": float(h.mean()),
            "human_median": float(np.median(h)),
            "llm_mean": float(l.mean()),
            "llm_median": float(np.median(l)),
            "mean_diff_human_minus_llm": float(diff.mean()),
            "diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        }

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f"CreativeGainBench metrics: human-written poems vs LLM generations (n={n} paired prompts)",
        fontsize=14,
        fontweight="bold",
    )
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # Row 1: distribution (violin + box) per continuous metric.
    for i, m in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        h, l = data["human"][m], data["llm"][m]
        parts = ax.violinplot([h, l], positions=[1, 2], showmeans=False, showextrema=False)
        for body, color in zip(parts["bodies"], [HUMAN_COLOR, LLM_COLOR]):
            body.set_facecolor(color)
            body.set_alpha(0.5)
        ax.boxplot(
            [h, l],
            positions=[1, 2],
            widths=0.15,
            showfliers=False,
            medianprops={"color": "black"},
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["human", "LLM"])
        ax.set_title(METRIC_LABELS[m], fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # Row 2, cols 1-2: mean difference (human - llm) with bootstrap CI, per metric (z-scored for comparability).
    ax = fig.add_subplot(gs[1, 0:2])
    effect = []
    errs = []
    for m in METRICS:
        h, l = data["human"][m], data["llm"][m]
        pooled_std = np.concatenate([h, l]).std() or 1.0
        s = summary["metrics"][m]
        effect.append(s["mean_diff_human_minus_llm"] / pooled_std)
        errs.append(
            [
                (s["mean_diff_human_minus_llm"] - s["diff_ci95"][0]) / pooled_std,
                (s["diff_ci95"][1] - s["mean_diff_human_minus_llm"]) / pooled_std,
            ]
        )
    y = np.arange(len(METRICS))
    colors = [HUMAN_COLOR if e > 0 else LLM_COLOR for e in effect]
    ax.barh(y, effect, xerr=np.array(errs).T, color=colors, alpha=0.75, height=0.55)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
    ax.set_xlabel("standardized mean difference (human − LLM), 95% bootstrap CI")
    ax.set_title("Which side scores higher, per metric?", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Row 2, col 3: gate pass rates.
    ax = fig.add_subplot(gs[1, 2])
    gates = ["cue_gate", "d_gate"]
    x = np.arange(len(gates))
    w = 0.35
    ax.bar(x - w / 2, [data["human"][g].mean() for g in gates], w, label="human", color=HUMAN_COLOR, alpha=0.8)
    ax.bar(x + w / 2, [data["llm"][g].mean() for g in gates], w, label="LLM", color=LLM_COLOR, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["CUE > 0", "R_D > δ_D"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_title("Gate pass rates", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Row 2, col 4: paired scatter of the composite score.
    ax = fig.add_subplot(gs[1, 3])
    h, l = data["human"]["score"], data["llm"]["score"]
    lim = max(h.max(), l.max()) * 1.05 or 1.0
    ax.scatter(h, l, s=12, alpha=0.45, color="#55A868", edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("human R_creativity")
    ax.set_ylabel("LLM R_creativity")
    frac_h = float((h > l).mean())
    ax.set_title(f"Paired scores (human wins {frac_h:.0%})", fontsize=10)
    ax.grid(alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {png_path} and {summary_path}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
