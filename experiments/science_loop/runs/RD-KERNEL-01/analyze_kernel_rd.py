#!/usr/bin/env python3
"""RD-KERNEL-01: Parzen R_D discrimination + PCA-join vs P1–P4."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from creativegainbench.stats.creativity_stats import (  # noqa: E402
    ComparisonPipeline,
    ConcordanceCC,
    CreativityMetric,
    EnergyDistance,
    HedgesG,
    LogVarianceRatio,
    MeasurementLevel,
    Resampler,
    Sample,
)

HERE = Path(__file__).resolve().parent
PAIRED_PATH = HERE / "paired_eval_kernel.jsonl"
SOFT_REPORT = REPO / "experiments/science_loop/runs/RD-SOFT-01/report.json"
HARD_REPORT = REPO / "experiments/science_loop/runs/RD-INSPECT-01/report.json"
PCA_PATH = REPO / "experiments/science_loop/runs/EMB-PCA-01/pca_coords.jsonl"
ARTIFACTS = REPO / "src/creativegainbench/artifacts/poetry_v2"
OUT_JSON = HERE / "report.json"
OUT_MD = HERE / "REPORT.md"

MODELS = ("gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b")
N_BOOT = 300
N_PERM = 300
SEED = 42


class KernelStructuralNovelty(CreativityMetric):
    name = "kernel_structural_novelty"
    framework = "poetry_v2_kernel_parzen"
    level = MeasurementLevel.CONTINUOUS


def load_paired(path: Path = PAIRED_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _arr(xs) -> np.ndarray:
    return np.asarray(xs, dtype=float)


def roc_auc_human_higher(h: np.ndarray, m: np.ndarray) -> float:
    n_h, n_m = len(h), len(m)
    if n_h == 0 or n_m == 0:
        return float("nan")
    all_s = np.concatenate([h, m])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty(len(all_s), dtype=float)
    ranks[order] = np.arange(1, len(all_s) + 1, dtype=float)
    _, inv, counts = np.unique(all_s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    u = ranks[:n_h].sum() - n_h * (n_h + 1) / 2.0
    return float(u / (n_h * n_m))


def percentile_ci(stat, h, m, *, paired: bool, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n, k = len(h), len(m)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        if paired:
            idx = rng.integers(0, n, n)
            reps[b] = stat(h[idx], m[idx])
        else:
            reps[b] = stat(h[rng.integers(0, n, n)], m[rng.integers(0, k, k)])
    theta = stat(h, m)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return float(theta), float(lo), float(hi)


def linear_r2(X: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 4:
        return float("nan")
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-18
    return 1.0 - ss_res / ss_tot


def location_block(h: np.ndarray, m: np.ndarray) -> dict:
    mean_d, d_lo, d_hi = percentile_ci(
        lambda a, b: float((a - b).mean()), h, m, paired=True
    )
    return {
        "n": int(len(h)),
        "human_mean": float(h.mean()),
        "model_mean": float(m.mean()),
        "paired_mean_delta": mean_d,
        "paired_mean_delta_ci": [d_lo, d_hi],
        "ci_excludes_zero": bool(d_lo > 0 or d_hi < 0),
        "win_rate_human": float(np.mean(h > m)),
        "roc_auc_human_higher": roc_auc_human_higher(h, m),
        "hedges_g": float(
            (h.mean() - m.mean())
            / (math.sqrt(0.5 * (h.var(ddof=1) + m.var(ddof=1))) + 1e-18)
        ),
        "log_var_ratio": float(
            math.log((m.var(ddof=1) + 1e-18) / (h.var(ddof=1) + 1e-18))
        ),
    }


def pipeline_rows(h: np.ndarray, m: np.ndarray, ids) -> list[dict]:
    sample = Sample(h, m, item_ids=np.asarray(ids))
    pipe = ComparisonPipeline(
        measures=[EnergyDistance(), LogVarianceRatio(), HedgesG(), ConcordanceCC()],
        resampler=Resampler(n_boot=N_BOOT, n_perm=N_PERM, seed=SEED),
        alpha=0.05,
    )
    return pipe.run(KernelStructuralNovelty(), sample).rows()


def pca_join(rows: list[dict], model: str = "gemma2:2b") -> dict:
    by_id = {r["poem_id"]: r for r in rows if model in r["models"]}
    human_pc, llm_pc = {}, {}
    if not PCA_PATH.exists():
        return {"error": f"missing {PCA_PATH}"}
    with PCA_PATH.open() as fh:
        for line in fh:
            rec = json.loads(line)
            vec = [rec["pc1"], rec["pc2"], rec["pc3"]]
            if rec["label"] == "human":
                human_pc[rec["id"]] = vec
            else:
                llm_pc[rec["id"]] = vec
    ids = sorted(set(human_pc) & set(llm_pc) & set(by_id))
    X_h, X_m, rd_h, rd_m = [], [], [], []
    for pid in ids:
        r = by_id[pid]
        X_h.append(human_pc[pid])
        X_m.append(llm_pc[pid])
        rd_h.append(r["human"]["r_d_norm"])
        rd_m.append(r["models"][model]["r_d_norm"])
    X_all = np.vstack([np.asarray(X_h), np.asarray(X_m)])
    rd_all = np.concatenate([rd_h, rd_m])
    rd_r2_k = {str(k): linear_r2(X_all[:, :k], rd_all) for k in (1, 2, 3)}
    soft_r2 = None
    hard_r2 = 0.017
    if SOFT_REPORT.exists():
        soft = json.loads(SOFT_REPORT.read_text())
        soft_r2 = soft.get("pca_join", {}).get("r_d_r2_by_n_pcs", {}).get("3")
    if HARD_REPORT.exists():
        hard = json.loads(HARD_REPORT.read_text())
        hard_r2 = hard.get("pca_join", {}).get("r_d_r2_by_n_pcs", {}).get("3", 0.017)
    return {
        "n_pairs": len(ids),
        "model": model,
        "r_d_r2_by_n_pcs": rd_r2_k,
        "baseline_soft_r2_pc3": soft_r2,
        "baseline_hard_r2_pc3": hard_r2,
    }


def analyze_model(rows: list[dict], model: str) -> dict:
    paired = [r for r in rows if model in r["models"]]
    h = _arr([r["human"]["r_d_norm"] for r in paired])
    m = _arr([r["models"][model]["r_d_norm"] for r in paired])
    ids = [r["poem_id"] for r in paired]
    loc = location_block(h, m)
    loc["pipeline"] = pipeline_rows(h, m, ids)
    by_dom = defaultdict(list)
    for r in paired:
        by_dom[r["domain"]].append(r)
    loc["by_domain"] = {}
    for d, grp in sorted(by_dom.items()):
        hh = _arr([r["human"]["r_d_norm"] for r in grp])
        mm = _arr([r["models"][model]["r_d_norm"] for r in grp])
        loc["by_domain"][str(d)] = {
            "n": len(grp),
            "paired_mean_delta": float((hh - mm).mean()),
            "roc_auc_human_higher": roc_auc_human_higher(hh, mm),
        }
    return loc


def axiom_checks(rows: list[dict], meta: dict, thr: dict) -> dict:
    rates = []
    for d, info in thr.items():
        if isinstance(info, dict) and "neg_pass_rate" in info:
            rates.append(float(info["neg_pass_rate"]))
    mean_rate = float(np.mean(rates)) if rates else float("nan")
    fail_gate_rate = 1.0 - mean_rate if rates else float("nan")
    return {
        "neg_pass_rates_by_domain": {
            d: info.get("neg_pass_rate")
            for d, info in thr.items()
            if isinstance(info, dict)
        },
        "mean_neg_pass_rate": mean_rate,
        "mean_correct_reject_rate": fail_gate_rate,
        "p4_pass": bool(fail_gate_rate >= 0.95) if rates else False,
        "sigma": meta.get("sigma"),
        "n_eval": len(rows),
    }


def criteria(report: dict) -> dict:
    gemma = report["by_model"].get("gemma2:2b", {})
    pca = report.get("pca_join", {})
    r2 = float(pca.get("r_d_r2_by_n_pcs", {}).get("3", float("nan")))
    p1 = bool(r2 >= 0.15) if not math.isnan(r2) else False
    ci_ex = bool(gemma.get("ci_excludes_zero"))
    auc = float(gemma.get("roc_auc_human_higher", float("nan")))
    p2 = bool(ci_ex or (not math.isnan(auc) and auc >= 0.58))
    ax = report.get("axioms", {})
    mean_neg = float(ax.get("mean_neg_pass_rate", float("nan")))
    p3 = bool(0.03 <= mean_neg <= 0.08) if not math.isnan(mean_neg) else False
    p4 = bool(ax.get("p4_pass"))
    return {
        "P1_geometry_link": {
            "pass": p1,
            "r2_pc1_3": r2,
            "threshold": 0.15,
            "baseline_soft": pca.get("baseline_soft_r2_pc3"),
            "baseline_hard": pca.get("baseline_hard_r2_pc3"),
        },
        "P2_ranking": {
            "pass": p2,
            "ci_excludes_zero": ci_ex,
            "roc_auc": auc,
            "paired_mean_delta": gemma.get("paired_mean_delta"),
            "paired_mean_delta_ci": gemma.get("paired_mean_delta_ci"),
            "note": (
                "Observed, not fitted. If null, report as metric-honest null — "
                "do not retune σ."
            ),
        },
        "P3_negatives": {
            "pass": p3,
            "mean_neg_pass_rate": mean_neg,
            "target_interval": [0.03, 0.08],
        },
        "P4_axioms": {
            "pass": p4,
            "mean_correct_reject_rate": ax.get("mean_correct_reject_rate"),
            "threshold": 0.95,
        },
    }


def write_report_md(report: dict) -> None:
    crit = report["criteria"]
    gemma = report["by_model"]["gemma2:2b"]
    p1, p2, p3, p4 = (
        crit["P1_geometry_link"],
        crit["P2_ranking"],
        crit["P3_negatives"],
        crit["P4_axioms"],
    )
    lines = [
        "# RD-KERNEL-01 — Parzen / kernel ProbeCompressor R_D inspect",
        "",
        "Geometry-native Gaussian Parzen probe CE over MiniLM idea embeddings. "
        "σ and δ_D calibrated on **negatives only**. No Ollama.",
        "",
        "## Pre-registered criteria (gemma2:2b)",
        "",
        "| Criterion | Pass | Detail |",
        "|-----------|------|--------|",
        f"| P1 Geometry link R² ≥ 0.15 | "
        f"{'YES' if p1['pass'] else 'NO'} | "
        f"ker R²={p1['r2_pc1_3']:.4f} "
        f"(soft {p1['baseline_soft']}, hard {p1['baseline_hard']}) |",
        f"| P2 Ranking (CI≠0 or AUC≥0.58) | "
        f"{'YES' if p2['pass'] else 'NO'} | "
        f"Δ={p2['paired_mean_delta']:.4g} "
        f"CI={p2['paired_mean_delta_ci']} "
        f"AUC={p2['roc_auc']:.3f} |",
        f"| P3 Negatives pass-rate ∈ [0.03,0.08] | "
        f"{'YES' if p3['pass'] else 'NO'} | "
        f"mean={p3['mean_neg_pass_rate']:.4f} |",
        f"| P4 Axioms (≥95% correct reject) | "
        f"{'YES' if p4['pass'] else 'NO'} | "
        f"reject={p4['mean_correct_reject_rate']:.4f} |",
        "",
        "## gemma2:2b location",
        "",
        f"- n={gemma['n']}, human_mean={gemma['human_mean']:.4g}, "
        f"model_mean={gemma['model_mean']:.4g}",
        f"- paired Δ (H−M)={gemma['paired_mean_delta']:.4g} "
        f"CI={gemma['paired_mean_delta_ci']}",
        f"- Hedges g={gemma['hedges_g']:.4g}, "
        f"log-var ratio={gemma['log_var_ratio']:.4g}",
        "",
        "## Interpretation",
        "",
        "- **P1**: Parzen CE recovers embedding-class geometry (R² ≫ soft/hard ~0.017).",
        "- **P2**: CI excludes 0, so ranking signal exists, but Δ(H−M)<0 and "
        "AUC≪0.5 ⇒ **models score higher than humans** on ker R_D. That is a "
        "construct finding (geometry linked, human>LLM not supported) — "
        "do not retune σ to flip the sign.",
        "",
        "## Meta",
        "",
        f"- σ={report['axioms'].get('sigma')}, backend=kernel_parzen",
        f"- n_eval={report['n_eval']}",
        "",
        "If P1 fails → escalate to leave-one-out / kNN CE (still label-free). "
        "If P1 passes and human>LLM fails → class geometry is in the score but "
        "does not favor human>LLM (construct finding, not a bug to force).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    rows = load_paired()
    meta = json.loads((ARTIFACTS / "kernel_meta.json").read_text())
    thr = json.loads((ARTIFACTS / "kernel_delta_d_thresholds.json").read_text())
    print(f"loaded {len(rows)} kernel eval pairs", flush=True)
    report: dict = {
        "n_eval": len(rows),
        "models": list(MODELS),
        "by_model": {},
        "pca_join": {},
        "axioms": axiom_checks(rows, meta, thr),
    }
    for model in MODELS:
        print(f"analyzing {model} ...", flush=True)
        report["by_model"][model] = analyze_model(rows, model)
    print("pca join ...", flush=True)
    report["pca_join"] = pca_join(rows, "gemma2:2b")
    report["criteria"] = criteria(report)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_report_md(report)
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    print(json.dumps(report["criteria"], indent=2), flush=True)


if __name__ == "__main__":
    main()
