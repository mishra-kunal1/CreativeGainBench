"""E7 analysis: PairedMeanDiff on item-yoked CUE (and brier_delta) contrasts.

Reads ``results/e7_cue_panel.jsonl``. Writes JSON/MD under
``experiments/falsifiability/results/`` — never experiment1 cue_*.jsonl.

Pass (PROTOCOL): for each control in {cross, random, irrelevant},
PairedMeanDiff(CUE_matched − CUE_control) is DIFFERENT with positive estimate
after Benjamini–Yekutieli; length mean |bits_arm/bits_matched − 1| < 0.20.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    CONTROLS,
    assert_output_isolated,
    ensure_results_dir,
    load_config,
    read_jsonl,
)

from creativegainbench.stats import (
    CreativityMetric,
    HedgesG,
    MeasurementLevel,
    PairedMeanDiff,
    Resampler,
    Sample,
    Verdict,
    benjamini_yekutieli,
)


class CUEMetric(CreativityMetric):
    name = "cue"
    framework = "falsifiability_e7"
    level = MeasurementLevel.CONTINUOUS
    bounds = (0.0, None)


class BrierDeltaMetric(CreativityMetric):
    name = "brier_delta"
    framework = "falsifiability_e7"
    level = MeasurementLevel.CONTINUOUS
    bounds = (0.0, None)


def _pivot(
    rows: list[dict[str, Any]],
    *,
    receiver: str | None,
    field: str,
) -> dict[str, dict[str, float]]:
    """item_id → {arm: value} for one receiver (or the only receiver present)."""
    by_rx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get(field) is None:
            continue
        by_rx[str(r["receiver"])].append(r)
    if not by_rx:
        return {}
    if receiver and receiver in by_rx:
        chosen = receiver
    elif receiver and receiver not in by_rx and len(by_rx) == 1:
        chosen = next(iter(by_rx))
    elif receiver and receiver not in by_rx:
        chosen = sorted(by_rx)[0]
    else:
        chosen = sorted(by_rx)[0] if len(by_rx) == 1 else (
            receiver if receiver in by_rx else sorted(by_rx)[0]
        )
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in by_rx[chosen]:
        out[str(r["item_id"])][str(r["arm"])] = float(r[field])
    return dict(out)


def _aligned_pairs(
    pivoted: dict[str, dict[str, float]], control: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids, h, m = [], [], []
    for iid, arms in sorted(pivoted.items()):
        if "matched" in arms and control in arms:
            ids.append(iid)
            h.append(arms[control])
            m.append(arms["matched"])
    return np.asarray(h, float), np.asarray(m, float), ids


def _length_ratio_mean(rows: list[dict[str, Any]], *, receiver: str | None) -> dict[str, Any]:
    by_item: dict[str, dict[str, float]] = defaultdict(dict)
    recs = rows
    if receiver:
        filtered = [r for r in rows if r.get("receiver") == receiver]
        if filtered:
            recs = filtered
    for r in recs:
        if r.get("bit_length"):
            by_item[str(r["item_id"])][str(r["arm"])] = float(r["bit_length"])
    abs_dev = []
    per_arm: dict[str, list[float]] = defaultdict(list)
    for arms in by_item.values():
        matched = arms.get("matched")
        if not matched or matched <= 0:
            continue
        for arm in CONTROLS:
            bits = arms.get(arm)
            if not bits or bits <= 0:
                continue
            d = abs(bits / matched - 1.0)
            abs_dev.append(d)
            per_arm[arm].append(d)
    mean_abs = float(np.mean(abs_dev)) if abs_dev else float("nan")
    return {
        "mean_abs_rel_bits": mean_abs,
        "n": len(abs_dev),
        "tol": 0.20,
        "passed": bool(np.isfinite(mean_abs) and mean_abs < 0.20),
        "by_arm": {k: float(np.mean(v)) for k, v in per_arm.items()},
    }


def analyze_e7(
    rows: list[dict[str, Any]],
    *,
    primary_receiver: str | None = "llama3.1:8b",
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    rs = Resampler(n_boot=n_boot, n_perm=n_perm, ci_level=ci_level, seed=seed)
    cue_pivot = _pivot(rows, receiver=primary_receiver, field="cue")
    brier_pivot = _pivot(rows, receiver=primary_receiver, field="brier_delta")
    used_rx = primary_receiver
    if rows:
        present = sorted({str(r["receiver"]) for r in rows})
        if used_rx not in present:
            used_rx = present[0] if len(present) == 1 else present[0]

    def _contrasts(pivoted):
        results = []
        for control in CONTROLS:
            h, m, ids = _aligned_pairs(pivoted, control)
            if len(ids) < 3:
                results.append(
                    {
                        "control": control,
                        "n": len(ids),
                        "passed": False,
                        "reason": "n<3",
                    }
                )
                continue
            sample = Sample(h, m, item_ids=np.asarray(ids))
            msr = PairedMeanDiff(alpha=alpha)
            result = msr.evaluate(sample, rs)
            # Unpaired Hedges g is diagnostic only (ignores pairing).
            g = HedgesG(alpha=alpha).statistic(h, m)
            results.append(
                {
                    "control": control,
                    "n": len(ids),
                    "measure": result.name,
                    "estimate": result.estimate,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "margin": result.margin,
                    "p_value": result.p_value,
                    "verdict": result.verdict.value,
                    "hedges_g_unpaired_diagnostic": float(g),
                    "positive_estimate": bool(result.estimate > 0),
                    "_result": result,
                }
            )
        pvals = [r.get("p_value") if isinstance(r.get("p_value"), (int, float)) else float("nan") for r in results]
        padj = benjamini_yekutieli(pvals)
        out = []
        for r, pa in zip(results, padj):
            pa_f = None if (isinstance(pa, float) and math.isnan(pa)) else (None if pa is None else float(pa))
            r.pop("_result", None)
            r["p_adj_by"] = pa_f
            r["passed"] = bool(
                r.get("verdict") == Verdict.DIFFERENT.value
                and r.get("positive_estimate")
                and pa_f is not None
                and pa_f < alpha
            )
            out.append(r)
        return out

    cue_contrasts = _contrasts(cue_pivot)
    brier_contrasts = _contrasts(brier_pivot)
    length = _length_ratio_mean(rows, receiver=used_rx)
    cue_pass = all(c.get("passed") for c in cue_contrasts) and len(cue_contrasts) == 3
    passed = bool(cue_pass and length["passed"])
    return {
        "experiment": "E7",
        "note": "Not construct-validity E4. Isolation: did not read/write experiments/experiment1/results/cue_*.jsonl.",
        "primary_receiver": used_rx,
        "n_items_cue": len(cue_pivot),
        "passed": passed,
        "cue_contrasts": cue_contrasts,
        "brier_delta_contrasts": brier_contrasts,
        "length": length,
        "pass_rule": (
            "PairedMeanDiff DIFFERENT + positive estimate + BY p_adj < alpha "
            "for {cross, random, irrelevant}; mean |bits_arm/bits_matched-1| < 0.20"
        ),
    }


def _to_md(report: dict[str, Any]) -> str:
    lines = [
        "# E7 causal contribution controls",
        "",
        f"**Passed:** {report['passed']}",
        f"**Receiver:** `{report['primary_receiver']}`",
        f"**Items:** {report['n_items_cue']}",
        "",
        report["pass_rule"],
        "",
        "| Control | n | estimate | CI | p | p_adj BY | verdict | pass |",
        "|---------|---|----------|----|---|----------|---------|------|",
    ]
    for c in report["cue_contrasts"]:
        if "estimate" not in c:
            lines.append(f"| {c['control']} | {c.get('n')} | — | — | — | — | {c.get('reason')} | no |")
            continue
        lines.append(
            f"| {c['control']} | {c['n']} | {c['estimate']:+.4g} | "
            f"[{c['ci_low']:+.4g}, {c['ci_high']:+.4g}] | "
            f"{c['p_value']:.4g} | {c['p_adj_by']:.4g} | {c['verdict']} | {c['passed']} |"
        )
    length = report["length"]
    lines += [
        "",
        f"Length mean |bits_arm/bits_matched − 1| = {length['mean_abs_rel_bits']:.4f} "
        f"(pass {length['passed']}, tol 0.20)",
        "",
        "_Co-primary brier_delta contrasts are in the JSON report._",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--receiver", default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--n-perm", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    results = ensure_results_dir(cfg)
    scores_path = args.scores or (results / "e7_cue_panel.jsonl")
    if not scores_path.exists():
        raise SystemExit(f"missing {scores_path}")
    rows = read_jsonl(scores_path)
    report = analyze_e7(
        rows,
        primary_receiver=args.receiver or cfg.get("e7", {}).get("primary_receiver"),
        n_boot=int(args.n_boot if args.n_boot is not None else cfg.get("n_boot", 2000)),
        n_perm=int(args.n_perm if args.n_perm is not None else cfg.get("n_perm", 2000)),
        seed=int(cfg.get("seed", 42)),
        alpha=float(cfg.get("alpha", 0.05)),
        ci_level=float(cfg.get("ci_level", 0.95)),
    )
    out_json = assert_output_isolated(args.out_json or (results / "e7_report.json"))
    out_md = assert_output_isolated(args.out_md or (results / "e7_report.md"))
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    print(f"E7 passed={report['passed']} → {out_json}", flush=True)


if __name__ == "__main__":
    main()
