#!/usr/bin/env python3
"""Offline R_D human-vs-LLM inspection (no LLM calls).

Reads paired_eval.jsonl + EMB-PCA-01 coords. Writes report.json.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as spstats

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
PAIRED_PATH = HERE / "paired_eval.jsonl"
PCA_PATH = REPO / "experiments/science_loop/runs/EMB-PCA-01/pca_coords.jsonl"
OUT_PATH = HERE / "report.json"

MODELS = ("gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b")
HUMAN_FAVOR = {0, 2, 9}
LLM_FAVOR = {6, 8}
N_BOOT = 300
N_PERM = 300
SEED = 42
HIST_BINS = 24


class StructuralNovelty(CreativityMetric):
    name = "structural_novelty"
    framework = "poetry_v2_count_ngram"
    level = MeasurementLevel.CONTINUOUS


def load_paired(path: Path = PAIRED_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _arr(xs) -> np.ndarray:
    return np.asarray(xs, dtype=float)


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)) + 1e-18))


def iqr(a: np.ndarray) -> float:
    return float(np.subtract(*np.quantile(a, [0.75, 0.25])) + 1e-18)


def roc_auc_human_higher(h: np.ndarray, m: np.ndarray) -> float:
    """P(human r_d > model r_d) + 0.5 P(equal). Mann-Whitney / (n_h n_m)."""
    n_h, n_m = len(h), len(m)
    if n_h == 0 or n_m == 0:
        return float("nan")
    # ranks of human among all; average ties
    all_s = np.concatenate([h, m])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty(len(all_s), dtype=float)
    ranks[order] = np.arange(1, len(all_s) + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(all_s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    u = ranks[:n_h].sum() - n_h * (n_h + 1) / 2.0
    return float(u / (n_h * n_m))


def cohen_d(h: np.ndarray, m: np.ndarray) -> float:
    return float((h.mean() - m.mean()) / pooled_sd(h, m))


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


def histogram_pair(h: np.ndarray, m: np.ndarray, bins: int = HIST_BINS) -> dict:
    lo = float(min(h.min(), m.min()))
    hi = float(max(h.max(), m.max()))
    if lo == hi:
        hi = lo + 1e-12
    edges = np.linspace(lo, hi, bins + 1)
    hc, _ = np.histogram(h, bins=edges)
    mc, _ = np.histogram(m, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bin_centers": [float(x) for x in centers],
        "human": [int(x) for x in hc],
        "model": [int(x) for x in mc],
    }


def entropy_hist(x: np.ndarray, edges: np.ndarray) -> float:
    c, _ = np.histogram(x, bins=edges)
    p = c.astype(float)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def linear_r2(X: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 4:
        return float("nan")
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-18
    return 1.0 - ss_res / ss_tot


def logistic_cv(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = SEED) -> dict:
    """Nearest-centroid CV (stable without sklearn). Class 0 = human, 1 = LLM."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    h_idx = idx[y[idx] == 0]
    m_idx = idx[y[idx] == 1]
    folds = min(folds, max(2, min(len(h_idx), len(m_idx))))
    h_chunks = np.array_split(h_idx, folds)
    m_chunks = np.array_split(m_idx, folds)
    acc = []
    for f in range(folds):
        te = np.concatenate([h_chunks[f], m_chunks[f]])
        tr = np.setdiff1d(idx, te, assume_unique=False)
        ch = X[tr][y[tr] == 0].mean(axis=0)
        cm = X[tr][y[tr] == 1].mean(axis=0)
        dh = ((X[te] - ch) ** 2).sum(axis=1)
        dm = ((X[te] - cm) ** 2).sum(axis=1)
        pred = (dm < dh).astype(float)
        acc.append(float((pred == y[te]).mean()))
    return {
        "method": "nearest_centroid",
        "cv_accuracy_mean": float(np.mean(acc)),
        "cv_accuracy_std": float(np.std(acc)),
        "folds": folds,
        "chance": 0.5,
    }


def pipeline_rows(h: np.ndarray, m: np.ndarray, ids) -> list[dict]:
    sample = Sample(h, m, item_ids=np.asarray(ids))
    pipe = ComparisonPipeline(
        measures=[EnergyDistance(), LogVarianceRatio(), HedgesG(), ConcordanceCC()],
        resampler=Resampler(n_boot=N_BOOT, n_perm=N_PERM, seed=SEED),
        alpha=0.05,
    )
    return pipe.run(StructuralNovelty(), sample).rows()


def shape_companions(h: np.ndarray, m: np.ndarray) -> dict:
    w1 = float(spstats.wasserstein_distance(h, m))
    psd = pooled_sd(h, m)
    hiqr = iqr(h)
    w1_s, w1_lo, w1_hi = percentile_ci(
        lambda a, b: spstats.wasserstein_distance(a, b), h, m, paired=False
    )
    ks = spstats.ks_2samp(h, m, alternative="two-sided", mode="auto")
    try:
        ad = spstats.anderson_ksamp([h, m])
        ad_stat = float(ad.statistic)
        ad_p = float(ad.pvalue) if ad.pvalue is not None else None
    except Exception:
        ad_stat, ad_p = float("nan"), None
    bf = spstats.levene(h, m, center="median")
    edges = np.linspace(min(h.min(), m.min()), max(h.max(), m.max()) + 1e-18, 21)
    q05, q95 = np.quantile(h, [0.05, 0.95])
    coverage_90 = float(np.mean((m >= q05) & (m <= q95)))
    coverage_full = float(np.mean((m >= h.min()) & (m <= h.max())))
    return {
        "wasserstein_1": w1,
        "wasserstein_1_over_pooled_sd": w1 / psd,
        "wasserstein_1_over_human_iqr": w1 / hiqr,
        "wasserstein_1_ci": [w1_lo, w1_hi],
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "anderson_darling_stat": ad_stat,
        "anderson_darling_pvalue": ad_p,
        "brown_forsythe_stat": float(bf.statistic),
        "brown_forsythe_pvalue": float(bf.pvalue),
        "sd_ratio_model_over_human": float(m.std(ddof=1) / (h.std(ddof=1) + 1e-18)),
        "entropy_human": entropy_hist(h, edges),
        "entropy_model": entropy_hist(m, edges),
        "llm_coverage_human_q05_q95": coverage_90,
        "llm_coverage_human_minmax": coverage_full,
    }


def location_block(h: np.ndarray, m: np.ndarray, ids) -> dict:
    delta = h - m
    mean_d, d_lo, d_hi = percentile_ci(lambda a, b: float((a - b).mean()), h, m, paired=True)
    win = float(np.mean(h > m))
    tie = float(np.mean(h == m))
    gate_h = None
    return {
        "n": int(len(h)),
        "human_mean": float(h.mean()),
        "model_mean": float(m.mean()),
        "human_median": float(np.median(h)),
        "model_median": float(np.median(m)),
        "human_q90": float(np.quantile(h, 0.90)),
        "model_q90": float(np.quantile(m, 0.90)),
        "paired_mean_delta": mean_d,
        "paired_mean_delta_ci": [d_lo, d_hi],
        "ci_excludes_zero": bool(d_lo > 0 or d_hi < 0),
        "win_rate_human": win,
        "tie_rate": tie,
        "cohen_d_human_minus_model": cohen_d(h, m),
        "roc_auc_human_higher": roc_auc_human_higher(h, m),
        "snr_abs_delta_over_sd_human": float(abs(mean_d) / (h.std(ddof=1) + 1e-18)),
        "snr_abs_delta_over_iqr_human": float(abs(mean_d) / iqr(h)),
        "histogram": histogram_pair(h, m),
    }


def gate_block(rows: list[dict], model: str) -> dict:
    hg, mg = [], []
    for r in rows:
        mv = r["models"].get(model)
        if not mv:
            continue
        hg.append(float(r["human"]["r_d_gate"] or 0.0))
        mg.append(float(mv["r_d_gate"] or 0.0))
    h, m = _arr(hg), _arr(mg)
    delta, lo, hi = percentile_ci(lambda a, b: float(a.mean() - b.mean()), h, m, paired=True)
    return {
        "human_pass_rate": float(h.mean()),
        "model_pass_rate": float(m.mean()),
        "paired_pass_delta": delta,
        "paired_pass_delta_ci": [lo, hi],
        "both_open": float(np.mean((h == 1) & (m == 1))),
        "feasibility_human": 1.0,  # filled later if needed
    }


def human_halves_energy(h: np.ndarray, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(h))
    mid = len(h) // 2
    a, b = h[idx[:mid]], h[idx[mid:]]
    sample = Sample(a, b)
    pipe = ComparisonPipeline(
        measures=[EnergyDistance()],
        resampler=Resampler(n_boot=N_BOOT, n_perm=N_PERM, seed=seed),
    )
    row = pipe.run(StructuralNovelty(), sample).rows()[0]
    return {
        "n_half": int(mid),
        "energy_distance": row["estimate"],
        "ci": [row["ci_low"], row["ci_high"]],
        "verdict": row["verdict"],
        "margin": row["margin"],
        "p_value": row["p_value"],
    }


def provenance_split(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        src = r.get("source") or "unknown"
        key = "poetrydb" if "poetrydb" in src else (
            "gutenberg" if "gutenberg" in src else "other"
        )
        by[key].append(r["human"]["r_d_norm"])
    out = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in by.items()}
    if "poetrydb" in by and "gutenberg" in by:
        a, b = _arr(by["poetrydb"]), _arr(by["gutenberg"])
        d, lo, hi = percentile_ci(lambda x, y: float(x.mean() - y.mean()), a, b, paired=False)
        out["poetrydb_minus_gutenberg"] = {"delta": d, "ci": [lo, hi]}
    return out


def analyze_model(rows: list[dict], model: str) -> dict:
    paired = [r for r in rows if model in r["models"]]
    h = _arr([r["human"]["r_d_norm"] for r in paired])
    m = _arr([r["models"][model]["r_d_norm"] for r in paired])
    ids = [r["poem_id"] for r in paired]
    loc = location_block(h, m, ids)
    loc["pipeline"] = pipeline_rows(h, m, ids)
    loc["shape"] = shape_companions(h, m)
    loc["gates"] = gate_block(paired, model)
    # per domain
    by_dom = defaultdict(list)
    for r in paired:
        by_dom[r["domain"]].append(r)
    domains = {}
    for d, grp in sorted(by_dom.items(), key=lambda kv: (kv[0] is None, kv[0])):
        hh = _arr([r["human"]["r_d_norm"] for r in grp])
        mm = _arr([r["models"][model]["r_d_norm"] for r in grp])
        domains[str(d)] = {
            "n": len(grp),
            "paired_mean_delta": float((hh - mm).mean()),
            "win_rate_human": float(np.mean(hh > mm)),
            "roc_auc_human_higher": roc_auc_human_higher(hh, mm),
            "human_mean": float(hh.mean()),
            "model_mean": float(mm.mean()),
        }
    loc["by_domain"] = domains
    favor_h = [r for r in paired if r["domain"] in HUMAN_FAVOR]
    favor_m = [r for r in paired if r["domain"] in LLM_FAVOR]
    def _delta(grp):
        if not grp:
            return None
        hh = _arr([r["human"]["r_d_norm"] for r in grp])
        mm = _arr([r["models"][model]["r_d_norm"] for r in grp])
        return float((hh - mm).mean())
    loc["stratum"] = {
        "human_favor_domains_0_2_9_delta": _delta(favor_h),
        "llm_favor_domains_6_8_delta": _delta(favor_m),
        "n_human_favor": len(favor_h),
        "n_llm_favor": len(favor_m),
    }
    return loc


def length_bands(rows: list[dict], model: str) -> dict:
    paired = [r for r in rows if model in r["models"]]
    h_sym = _arr([r["human"]["y_n_symbols"] for r in paired])
    m_sym = _arr([r["models"][model]["y_n_symbols"] for r in paired])
    h_rd = _arr([r["human"]["r_d_norm"] for r in paired])
    m_rd = _arr([r["models"][model]["r_d_norm"] for r in paired])
    # spearman
    sp_h = spstats.spearmanr(h_sym, h_rd)
    sp_m = spstats.spearmanr(m_sym, m_rd)
    mid_sym = 0.5 * (h_sym + m_sym)
    qs = np.quantile(mid_sym, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bands = []
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        mask = (mid_sym >= lo) & (mid_sym <= hi if i == 4 else mid_sym < hi)
        if mask.sum() < 8:
            continue
        d = h_rd[mask] - m_rd[mask]
        bands.append({
            "band": i,
            "sym_lo": float(lo),
            "sym_hi": float(hi),
            "n": int(mask.sum()),
            "mean_delta": float(d.mean()),
            "win_rate_human": float(np.mean(h_rd[mask] > m_rd[mask])),
        })
    return {
        "human_mean_symbols": float(h_sym.mean()),
        "model_mean_symbols": float(m_sym.mean()),
        "spearman_symbols_rd_human": {"rho": float(sp_h.statistic), "pvalue": float(sp_h.pvalue)},
        "spearman_symbols_rd_model": {"rho": float(sp_m.statistic), "pvalue": float(sp_m.pvalue)},
        "matched_symbol_quintiles": bands,
    }


def pca_join(rows: list[dict], model: str = "gemma2:2b") -> dict:
    by_id = {r["poem_id"]: r for r in rows if model in r["models"]}
    human_pc = {}
    llm_pc = {}
    with PCA_PATH.open() as fh:
        for line in fh:
            rec = json.loads(line)
            vec = [rec["pc1"], rec["pc2"], rec["pc3"]]
            if rec["label"] == "human":
                human_pc[rec["id"]] = vec
            else:
                llm_pc[rec["id"]] = vec
    ids = sorted(set(human_pc) & set(llm_pc) & set(by_id))
    X_h, X_m, rd_h, rd_m, d_rd, d_pc = [], [], [], [], [], []
    scatter = []
    for pid in ids:
        r = by_id[pid]
        ph, pm = np.array(human_pc[pid]), np.array(llm_pc[pid])
        rh, rm = r["human"]["r_d_norm"], r["models"][model]["r_d_norm"]
        X_h.append(ph)
        X_m.append(pm)
        rd_h.append(rh)
        rd_m.append(rm)
        d_rd.append(rh - rm)
        d_pc.append(float(np.linalg.norm(ph - pm)))
        scatter.append({
            "id": pid,
            "domain": r["domain"],
            "pc1_h": float(ph[0]),
            "pc1_m": float(pm[0]),
            "delta_rd": float(rh - rm),
            "delta_pc_l2": float(np.linalg.norm(ph - pm)),
            "rd_h": float(rh),
            "rd_m": float(rm),
        })
    X_h, X_m = np.asarray(X_h), np.asarray(X_m)
    X_all = np.vstack([X_h, X_m])
    y_all = np.concatenate([np.zeros(len(X_h)), np.ones(len(X_m))])
    rd_all = np.concatenate([rd_h, rd_m])
    d_rd = _arr(d_rd)
    d_pc = _arr(d_pc)

    class_k = {}
    rd_r2_k = {}
    for k in (1, 2, 3):
        class_k[k] = logistic_cv(X_all[:, :k], y_all)
        rd_r2_k[k] = linear_r2(X_all[:, :k], rd_all)

    sp_delta = spstats.spearmanr(d_pc, d_rd)
    sp_pc1 = spstats.spearmanr(X_all[:, 0], rd_all)
    return {
        "n_pairs": len(ids),
        "model": model,
        "pc_means_human": [float(x) for x in X_h.mean(axis=0)],
        "pc_means_llm": [float(x) for x in X_m.mean(axis=0)],
        "class_cv_by_n_pcs": class_k,
        "r_d_r2_by_n_pcs": {str(k): v for k, v in rd_r2_k.items()},
        "spearman_delta_pc_l2_vs_delta_rd": {
            "rho": float(sp_delta.statistic),
            "pvalue": float(sp_delta.pvalue),
        },
        "spearman_pc1_vs_rd": {
            "rho": float(sp_pc1.statistic),
            "pvalue": float(sp_pc1.pvalue),
        },
        "mean_abs_delta_rd": float(np.abs(d_rd).mean()),
        "mean_delta_pc_l2": float(d_pc.mean()),
        "scatter_sample": scatter[:80],  # compact for canvas
    }


def main() -> None:
    rows = load_paired()
    print(f"loaded {len(rows)} eval pairs", flush=True)
    h_all = _arr([r["human"]["r_d_norm"] for r in rows])
    report = {
        "n_eval": len(rows),
        "models": list(MODELS),
        "reliability": {
            "human_rater_matrix": False,
            "note": (
                "No multi-rater human scores; Krippendorff/ICC cannot run. "
                "SNR and human-half energy are the proxies."
            ),
            "human_sd": float(h_all.std(ddof=1)),
            "human_iqr": iqr(h_all),
            "human_halves_energy": human_halves_energy(h_all),
            "provenance": provenance_split(rows),
        },
        "by_model": {},
        "length_bands": {},
    }
    for model in MODELS:
        print(f"analyzing {model} ...", flush=True)
        report["by_model"][model] = analyze_model(rows, model)
        report["length_bands"][model] = length_bands(rows, model)
    print("pca join ...", flush=True)
    report["pca_join"] = pca_join(rows, "gemma2:2b")
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
