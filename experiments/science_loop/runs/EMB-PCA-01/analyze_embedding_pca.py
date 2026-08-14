#!/usr/bin/env python3
"""
EMB-PCA-01 — PCA / linear separability of human vs gemma2:2b embeddings.

Uses the same idea-level MiniLM + poetry boundary mean-pool as the measurement
stack (mean_pool_idea_embeddings), with F10 per-domain median length clips.

Writes JSON + Markdown under this run directory for the science-in-the-loop crew.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "experiment1"))

from creativegainbench.ideas.idea_extractor import (  # noqa: E402
    IdeaBoundaryDetector,
    poetry_line_splitter,
)
from creativegainbench.ideas.idea_ngram import mean_pool_idea_embeddings  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from lib import load_config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = "postgres://poems:poems@localhost:5432/poems"
MODEL = "gemma2:2b"


def _domain_clips(poems: list, max_chars: int) -> dict[int, int]:
    by: dict[int, list[int]] = defaultdict(list)
    for _pid, body, domain in poems:
        if domain is None:
            continue
        by[int(domain)].append(len((body or "").strip()))
    out = {}
    for d, lens in by.items():
        med = int(statistics.median(lens)) if lens else max_chars
        out[d] = max(200, min(max_chars, med))
    return out


def _pca(X: np.ndarray, n_comp: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scores, explained_variance_ratio, components). Centered SVD PCA."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]
    ev = (S**2) / max(n - 1, 1)
    total = ev.sum() or 1.0
    ratio = ev / total
    k = min(n_comp, Vt.shape[0])
    scores = Xc @ Vt[:k].T
    return scores, ratio[:k], Vt[:k]


def _silhouette_2class(X2: np.ndarray, y: np.ndarray) -> float:
    """Mean silhouette for binary labels in 2D (or kD) Euclidean space."""
    if len(X2) < 4:
        return float("nan")
    # pairwise distances
    n = len(X2)
    sil = []
    for i in range(n):
        same = y == y[i]
        other = ~same
        if same.sum() < 2 or other.sum() < 1:
            continue
        d = np.linalg.norm(X2 - X2[i], axis=1)
        a = d[same & (np.arange(n) != i)].mean()
        b = d[other].mean()
        sil.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(sil)) if sil else float("nan")


def _centroid_stats(X: np.ndarray, y: np.ndarray) -> dict:
    h = X[y == 0]
    m = X[y == 1]
    ch, cm = h.mean(0), m.mean(0)
    # within-class mean distance to own centroid
    wh = np.linalg.norm(h - ch, axis=1).mean() if len(h) else 0.0
    wm = np.linalg.norm(m - cm, axis=1).mean() if len(m) else 0.0
    between = float(np.linalg.norm(ch - cm))
    pooled = 0.5 * (wh + wm) or 1e-12
    return {
        "centroid_distance": between,
        "mean_within_human": float(wh),
        "mean_within_llm": float(wm),
        "separation_ratio": float(between / pooled),
    }


def _linear_cv_acc(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 42) -> dict:
    """Simple ridge-logistic via least-squares on one-hot — fallback without sklearn.

    Prefer sklearn LogisticRegression if available.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = min(folds, n // 10) if n >= 50 else max(2, min(folds, n // 5))
    if folds < 2:
        return {"cv_accuracy_mean": float("nan"), "cv_accuracy_std": float("nan"), "folds": 0}

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=cv)
        return {
            "method": "sklearn_logreg",
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "folds": folds,
            "chance": 0.5,
        }
    except ImportError:
        pass

    # Manual stratified fold + least-squares classifier on [X, 1]
    y_bin = y.astype(np.float64)
    fold_acc = []
    # rough stratify
    h_idx = idx[y[idx] == 0]
    m_idx = idx[y[idx] == 1]
    h_chunks = np.array_split(h_idx, folds)
    m_chunks = np.array_split(m_idx, folds)
    for f in range(folds):
        te = np.concatenate([h_chunks[f], m_chunks[f]])
        tr = np.setdiff1d(idx, te, assume_unique=False)
        Xtr = np.hstack([X[tr], np.ones((len(tr), 1))])
        Xte = np.hstack([X[te], np.ones((len(te), 1))])
        w, *_ = np.linalg.lstsq(Xtr, y_bin[tr], rcond=None)
        pred = (Xte @ w) >= 0.5
        fold_acc.append(float((pred == y_bin[te]).mean()))
    return {
        "method": "lstsq_fallback",
        "cv_accuracy_mean": float(np.mean(fold_acc)),
        "cv_accuracy_std": float(np.std(fold_acc)),
        "folds": folds,
        "chance": 0.5,
    }


@torch.no_grad()
def embed_texts(
    texts: list[str],
    *,
    encoder,
    boundary,
    batch_log_every: int = 100,
) -> np.ndarray:
    rows = []
    t0 = time.time()
    for i, text in enumerate(texts):
        emb = mean_pool_idea_embeddings(
            text,
            span_encoder=encoder,
            boundary_detector=boundary,
            sentence_splitter=poetry_line_splitter,
            boundary_threshold=0.5,
        )
        rows.append(emb.detach().cpu().numpy().astype(np.float64))
        if (i + 1) % batch_log_every == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  embedded {i+1}/{len(texts)} ({rate:.1f}/s)", flush=True)
    return np.stack(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--db-url", default=DEFAULT_DB)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="paired poems cap")
    parser.add_argument(
        "--stratified-limit",
        type=int,
        default=None,
        help="cap pairs via per-domain stratified sample (recommended ≥300)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    max_chars = int(cfg["max_chars"])
    artifacts = Path(cfg["artifacts"])

    with psycopg.connect(args.db_url) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.body, p.domain_cluster, g.output
            FROM poems p
            JOIN generations g ON g.poem_id = p.id AND g.model = %s
            WHERE p.split = %s
              AND p.domain_cluster IS NOT NULL
              AND p.body IS NOT NULL AND btrim(p.body) <> ''
              AND g.output IS NOT NULL AND btrim(g.output) <> ''
            ORDER BY p.id
            """,
            (args.model, cfg["split"]),
        ).fetchall()
    if args.stratified_limit is not None and args.stratified_limit < len(rows):
        import random

        rng = random.Random(args.seed)
        by_d: dict[int, list] = defaultdict(list)
        for r in rows:
            by_d[int(r[2])].append(r)
        # proportional allocation, at least 1 per domain when possible
        target = args.stratified_limit
        domains = sorted(by_d)
        alloc = {d: max(1, int(round(target * len(by_d[d]) / len(rows)))) for d in domains}
        # fix rounding to exact target
        while sum(alloc.values()) > target:
            d = max(alloc, key=lambda k: alloc[k])
            if alloc[d] > 1:
                alloc[d] -= 1
            else:
                break
        while sum(alloc.values()) < target:
            d = max(domains, key=lambda k: len(by_d[k]) - alloc[k])
            if alloc[d] < len(by_d[d]):
                alloc[d] += 1
            else:
                break
        sampled = []
        for d in domains:
            pool = by_d[d]
            k = min(alloc[d], len(pool))
            sampled.extend(rng.sample(pool, k))
        rows = sampled
        print(f"stratified sample n={len(rows)} (target={target}) alloc={alloc}", flush=True)
    elif args.limit is not None:
        rows = rows[: args.limit]
    if len(rows) < 20:
        raise SystemExit(f"need ≥20 paired eval rows, got {len(rows)}")
    poems_for_clip = [(pid, body, d) for pid, body, d, _ in rows]
    clips = _domain_clips(poems_for_clip, max_chars)
    print(f"paired n={len(rows)} model={args.model} clips={clips}", flush=True)

    human_texts, llm_texts, meta = [], [], []
    for pid, body, domain, out in rows:
        d = int(domain)
        clip = clips.get(d, max_chars)
        h = (body or "").strip()[:clip]
        m = (out or "").strip()[:clip]
        if not h or not m:
            continue
        human_texts.append(h)
        llm_texts.append(m)
        meta.append({"id": str(pid), "domain": d, "clip": clip, "len_h": len(h), "len_m": len(m)})

    n = len(human_texts)
    print(f"usable pairs={n}", flush=True)

    encoder = build_span_encoder("minilm", device=args.device)
    boundary = IdeaBoundaryDetector(hidden_dim=384)
    bpath = artifacts / "idea_boundary.pt"
    if bpath.exists():
        boundary.load_state_dict(
            torch.load(bpath, map_location="cpu", weights_only=True)
        )
    boundary.eval()

    print("embedding human…", flush=True)
    H = embed_texts(human_texts, encoder=encoder, boundary=boundary)
    print("embedding llm…", flush=True)
    M = embed_texts(llm_texts, encoder=encoder, boundary=boundary)

    X = np.vstack([H, M])
    y = np.array([0] * n + [1] * n)  # 0=human, 1=llm
    labels = ["human", "llm"]

    scores, ev_ratio, comps = _pca(X, n_comp=3)
    sil2 = _silhouette_2class(scores[:, :2], y)
    cents = _centroid_stats(X, y)
    cents_pca = _centroid_stats(scores[:, :2], y)
    cv = _linear_cv_acc(X, y, folds=5, seed=args.seed)
    cv_pca = _linear_cv_acc(scores[:, : min(3, scores.shape[1])], y, folds=5, seed=args.seed)

    # Paired vector difference stats
    delta = M - H
    delta_norm = np.linalg.norm(delta, axis=1)
    # cosine between paired human/llm
    cos = []
    for i in range(n):
        a, b = H[i], M[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            cos.append(float("nan"))
        else:
            cos.append(float(np.dot(a, b) / (na * nb)))
    cos_arr = np.array(cos, dtype=np.float64)
    cos_arr = cos_arr[~np.isnan(cos_arr)]

    report = {
        "id": "EMB-PCA-01",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "embedding": "mean_pool_idea_embeddings (MiniLM + poetry boundary)",
        "length_protocol": "F10 per-domain eval median clip (same as 03_score_rd)",
        "n_pairs": n,
        "embedding_dim": int(X.shape[1]),
        "pca_explained_variance_ratio": [float(x) for x in ev_ratio],
        "pca_cumulative_variance": [float(ev_ratio[: k + 1].sum()) for k in range(len(ev_ratio))],
        "silhouette_pca2": sil2,
        "centroid_full_space": cents,
        "centroid_pca2": cents_pca,
        "linear_probe_full_space": cv,
        "linear_probe_pca3": cv_pca,
        "paired_cosine_human_llm": {
            "mean": float(cos_arr.mean()) if len(cos_arr) else float("nan"),
            "std": float(cos_arr.std()) if len(cos_arr) else float("nan"),
            "median": float(np.median(cos_arr)) if len(cos_arr) else float("nan"),
        },
        "paired_delta_l2": {
            "mean": float(delta_norm.mean()),
            "std": float(delta_norm.std()),
            "median": float(np.median(delta_norm)),
        },
        "interpretation_guide": {
            "cv_acc_near_0.5": "embeddings do not linearly separate human vs llm",
            "cv_acc_ge_0.7": "substantial linear class signal in embedding space",
            "silhouette_le_0": "overlapping clouds in PCA plane",
            "separation_ratio": "centroid distance / mean within-class radius",
        },
    }

    # Save PCA coords for plotting / crew follow-ups
    coords = []
    for i in range(n):
        coords.append(
            {
                **meta[i],
                "label": "human",
                "pc1": float(scores[i, 0]),
                "pc2": float(scores[i, 1]) if scores.shape[1] > 1 else 0.0,
                "pc3": float(scores[i, 2]) if scores.shape[1] > 2 else 0.0,
            }
        )
        coords.append(
            {
                **meta[i],
                "label": "llm",
                "pc1": float(scores[n + i, 0]),
                "pc2": float(scores[n + i, 1]) if scores.shape[1] > 1 else 0.0,
                "pc3": float(scores[n + i, 2]) if scores.shape[1] > 2 else 0.0,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pca_report.json").write_text(json.dumps(report, indent=2))
    (OUT_DIR / "pca_coords.jsonl").write_text(
        "\n".join(json.dumps(c) for c in coords) + "\n"
    )

    # Optional scatter
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            scores[y == 0, 0],
            scores[y == 0, 1],
            s=12,
            alpha=0.45,
            label="human",
            c="#1f4e79",
        )
        ax.scatter(
            scores[y == 1, 0],
            scores[y == 1, 1],
            s=12,
            alpha=0.45,
            label=args.model,
            c="#c45c26",
        )
        ax.set_xlabel(f"PC1 ({100*ev_ratio[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100*ev_ratio[1]:.1f}% var)" if len(ev_ratio) > 1 else "PC2")
        ax.set_title(f"EMB-PCA-01 idea embeddings: human vs {args.model}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "pca_scatter.png", dpi=140)
        plt.close(fig)
        report["scatter_plot"] = "pca_scatter.png"
        (OUT_DIR / "pca_report.json").write_text(json.dumps(report, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc}", flush=True)

    acc = cv["cv_accuracy_mean"]
    if isinstance(acc, float) and not math.isnan(acc):
        if acc < 0.58:
            verdict = "WEAK / overlapping — embedding space does not cleanly separate human vs LLM"
        elif acc < 0.70:
            verdict = "MODERATE — some linear signal; classes largely overlap"
        else:
            verdict = "STRONG — embeddings carry substantial human-vs-LLM class information"
    else:
        verdict = "UNKNOWN (CV failed)"

    md = f"""# EMB-PCA-01 — Embedding PCA / separability

**Generated:** {report['generated_at']}  
**Model:** `{args.model}`  
**Pairs:** {n} eval (F10 length-clipped)  
**Embedding:** MiniLM idea mean-pool + poetry boundary (measurement stack)

## Verdict
**{verdict}**

## PCA variance
| Component | Explained | Cumulative |
|-----------|-----------|------------|
"""
    cum = 0.0
    for i, r in enumerate(ev_ratio):
        cum += float(r)
        md += f"| PC{i+1} | {100*r:.2f}% | {100*cum:.2f}% |\n"

    md += f"""
## Separability
| Metric | Value |
|--------|------:|
| Linear probe CV acc (full dim) | {cv['cv_accuracy_mean']:.4f} ± {cv['cv_accuracy_std']:.4f} ({cv.get('method')}) |
| Linear probe CV acc (PCA-3) | {cv_pca['cv_accuracy_mean']:.4f} ± {cv_pca['cv_accuracy_std']:.4f} |
| Chance | 0.50 |
| Silhouette (PCA-2) | {sil2:.4f} |
| Centroid distance (full) | {cents['centroid_distance']:.4f} |
| Separation ratio (full) | {cents['separation_ratio']:.4f} |
| Mean paired cosine(human, llm) | {report['paired_cosine_human_llm']['mean']:.4f} |
| Mean paired ‖Δ‖₂ | {report['paired_delta_l2']['mean']:.4f} |

## Artifacts
- `pca_report.json` — machine-readable summary
- `pca_coords.jsonl` — PC1–3 per example
- `pca_scatter.png` — PC1/PC2 scatter (if matplotlib available)

## Science-loop notes
- If CV ≈ 0.5 and silhouette ≤ 0: measurement may be asking R_D/CUE to separate classes that **share the same embedding manifold** — consider richer encoders or features beyond MiniLM idea pools.
- If CV ≫ 0.5 but E4/R_D still fail: class signal exists in embeddings but **is not used** by current creativity metrics (metric-design problem, not representation collapse).
- Does **not** fit δ_D; diagnostic only.
"""
    (OUT_DIR / "REPORT.md").write_text(md)
    # Stable pointer for the crew
    latest = OUT_DIR.parent.parent / "embedding_pca_latest.md"
    latest.write_text(md)
    print(json.dumps({k: report[k] for k in ("n_pairs", "linear_probe_full_space", "silhouette_pca2", "pca_explained_variance_ratio")}, indent=2))
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {latest}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
