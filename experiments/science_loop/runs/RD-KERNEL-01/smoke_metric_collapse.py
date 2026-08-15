#!/usr/bin/env python3
"""
Smoke test: do R_D, R_B, and R_creativity collapse into the same construct?

Hypothesis (user): the frozen codebook may be unnecessary for R_D, but if
R_B still routes through the same codebook soft-clustering, metrics may
become highly correlated — measuring the same thing.

Design (offline, HashReceiver, no Ollama):
  * r_d_ker     — Parzen CE, no codebook
  * r_d_hard    — CountNgram order-3 over VQ symbols (codebook)
  * r_b_cb      — soft-cluster entropy over frozen codebook (current R_B)
  * r_b_cont    — codebook-free soft diversity over receiver embeddings
  * r_creativity — stub CUE × d_gate(ker) × (CUE · (1 + α r_b_cb))

Verdict: high |ρ(ker, hard)| or |ρ(ker, r_b_cb)| ⇒ collapse; low ⇒ distinct.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from creativegainbench.ideas.artifacts import load_kernel_backend  # noqa: E402
from creativegainbench.ideas.idea_extractor import (  # noqa: E402
    IdeaBoundaryDetector,
    poetry_line_splitter,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.cue import cue_gate, stub_positive_cue  # noqa: E402
from creativegainbench.metrics.delta_d import d_gate, resolve_delta_d  # noqa: E402
from creativegainbench.metrics.deformation import (  # noqa: E402
    compute_deformation,
    compute_kernel_deformation,
)
from creativegainbench.metrics.feasibility import feasibility_bit  # noqa: E402
from creativegainbench.metrics.receiver_expansion import soft_cluster_entropy  # noqa: E402
from creativegainbench.receivers.hash_receiver import HashReceiverAgent  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = REPO / "src/creativegainbench/artifacts/poetry_v2"
CORPUS = REPO / "experiments/science_loop/runs/RD-SOFT-01/corpus_by_domain.jsonl"
PAIRED = HERE / "paired_eval_kernel.jsonl"
OUT_JSON = HERE / "smoke_metric_collapse.json"
OUT_MD = HERE / "SMOKE_METRIC_COLLAPSE.md"
MODEL = "gemma2:2b"


def continuous_receiver_entropy(
    embeddings: torch.Tensor, temperature: float = 1.0
) -> float:
    """
    Codebook-free diversity: soft-assignment entropy treating the receiver
    sample cloud as its own support (self soft-clustering), normalized by log n.
    """
    if embeddings.numel() == 0 or embeddings.size(0) < 2:
        return 0.0
    n = embeddings.size(0)
    dists = torch.cdist(embeddings, embeddings)
    soft = F.softmax(-dists / max(temperature, 1e-8), dim=-1)
    avg = soft.mean(dim=0).clamp_min(1e-12)
    h = float(-(avg * torch.log(avg)).sum().item())
    return float(max(0.0, min(1.0, h / math.log(n))))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-18 or b.std() < 1e-18:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-18 or b.std() < 1e-18:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sample_rows(n_pairs: int, seed: int = 42) -> list[dict]:
    """Balanced sample of eval pairs that have kernel scores + corpus body/gens."""
    paired = [json.loads(l) for l in PAIRED.read_text().splitlines() if l.strip()]
    corpus = {
        r["poem_id"]: r
        for r in (json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip())
    }
    # Prefer domains with strong prior separation (6, 9, 3) + a weaker one (0).
    prefer = {6, 9, 3, 0, 8}
    cand = [
        r
        for r in paired
        if MODEL in r["models"]
        and r["poem_id"] in corpus
        and int(r["domain"]) in prefer
        and (corpus[r["poem_id"]].get("generations") or {}).get(MODEL)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    # Stratify a bit by domain
    by_dom: dict[int, list] = {}
    for r in cand:
        by_dom.setdefault(int(r["domain"]), []).append(r)
    out: list[dict] = []
    per = max(1, n_pairs // max(len(by_dom), 1))
    for d, grp in by_dom.items():
        for r in grp[:per]:
            c = corpus[r["poem_id"]]
            out.append(
                {
                    "poem_id": r["poem_id"],
                    "domain": int(r["domain"]),
                    "label": "human",
                    "text": (c.get("body") or "").strip(),
                    "ker_cached": r["human"]["r_d_norm"],
                }
            )
            out.append(
                {
                    "poem_id": r["poem_id"],
                    "domain": int(r["domain"]),
                    "label": "llm",
                    "text": (c["generations"][MODEL] or "").strip(),
                    "ker_cached": r["models"][MODEL]["r_d_norm"],
                }
            )
            if len(out) // 2 >= n_pairs:
                return out[: 2 * n_pairs]
    return out[: 2 * n_pairs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-tasks", type=int, default=3)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    rows = sample_rows(args.n_pairs)
    print(f"scoring {len(rows)} texts ({args.n_pairs} pairs) on {device}", flush=True)

    codebook = IdeaCodebook(
        centroids=torch.load(
            ART / "idea_codebook.pt", map_location="cpu", weights_only=True
        )["centroids"]
    )
    encoder = build_span_encoder("minilm", device=device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(ART / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()

    kernel = load_kernel_backend(verify_hashes=True)
    hard_ctx: dict = {}
    hard_delta = json.loads((ART / "delta_d_thresholds.json").read_text())
    for d in {r["domain"] for r in rows}:
        p = ART / f"domain_{d}_ctx.pkl"
        with open(p, "rb") as fh:
            hard_ctx[d] = pickle.load(fh)

    battery = json.loads(
        (REPO / "src/creativegainbench/artifacts/task_battery_v1.json").read_text()
    )["tasks"][: args.n_tasks]
    receiver = HashReceiverAgent(
        span_encoder=encoder,
        boundary_detector=boundary,
        sentence_splitter=poetry_line_splitter,
        boundary_threshold=0.5,
        seed=42,
    )

    records = []
    for i, r in enumerate(rows):
        d = r["domain"]
        text = r["text"]
        if not text:
            continue

        # --- R_D kernel ---
        kctx = kernel.context_for(d)
        kdelta = kernel.delta_d_for(d)
        kres = compute_kernel_deformation(
            text,
            kctx,
            span_encoder=encoder,
            boundary_detector=boundary,
            sentence_splitter=poetry_line_splitter,
        )

        # --- R_D hard (codebook) ---
        hres = compute_deformation(
            text,
            hard_ctx[d],
            span_encoder=encoder,
            codebook=codebook,
            boundary_detector=boundary,
            sentence_splitter=poetry_line_splitter,
        )
        hdelta = resolve_delta_d(hard_delta, d)

        # --- R_B codebook + continuous from the SAME receiver samples ---
        cents = codebook.centroids.to(device)
        cb_ents, cont_vals = [], []
        for q in battery:
            cond = receiver.condition(q["input"], context=text)
            _s, emb = receiver.sample_with_embeddings(cond, n=args.n_samples)
            emb = emb.to(device)
            cb_ents.append(soft_cluster_entropy(emb, cents, temperature=1.0))
            cont_vals.append(continuous_receiver_entropy(emb))
        max_h = math.log(max(cents.size(0), 2))
        r_b_cb = float(
            max(0.0, min(1.0, (sum(cb_ents) / len(cb_ents)) / max_h))
        ) if cb_ents else 0.0
        r_b_cont = float(sum(cont_vals) / len(cont_vals)) if cont_vals else 0.0

        # --- R_creativity (stub CUE × ker d_gate × (CUE · (1 + α R_B))) ---
        cue_val, _ = stub_positive_cue(text)
        feas = bool(feasibility_bit(text))
        cg = cue_gate(cue_val)
        dg = d_gate(kres.r_d_norm, kdelta, feasible=feas)
        alpha = 1.0
        r_creativity = cg * dg * (cue_val * (1.0 + alpha * r_b_cb))

        rec = {
            "poem_id": r["poem_id"],
            "domain": d,
            "label": r["label"],
            "r_d_ker": kres.r_d_norm,
            "r_d_ker_gate": dg,
            "r_d_hard": hres.r_d_norm,
            "r_d_hard_gate": d_gate(hres.r_d_norm, hdelta, feasible=feas),
            "r_b_cb": r_b_cb,
            "r_b_cont": r_b_cont,
            "r_creativity": r_creativity,
            "cue": cue_val,
            "d_gate": dg,
            "y_n_ideas_ker": kres.y_n_symbols,
        }
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    keys = ["r_d_ker", "r_d_hard", "r_b_cb", "r_b_cont", "r_creativity"]
    mat = {k: np.array([r[k] for r in records], dtype=float) for k in keys}
    corr = {
        f"{a}__{b}": {
            "spearman": spearman(mat[a], mat[b]),
            "pearson": pearson(mat[a], mat[b]),
        }
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    }

    def auc_human_higher(h, m):
        na, nb = len(h), len(m)
        allv = np.concatenate([h, m])
        order = np.argsort(allv, kind="mergesort")
        ranks = np.empty(len(allv))
        ranks[order] = np.arange(1, len(allv) + 1)
        _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, ranks)
            ranks = (sums / counts)[inv]
        u = ranks[:na].sum() - na * (na + 1) / 2.0
        return float(u / (na * nb))

    labels = np.array([r["label"] for r in records])
    sep = {}
    for k in keys:
        h = mat[k][labels == "human"]
        m = mat[k][labels == "llm"]
        a = auc_human_higher(h, m)
        sep[k] = {
            "auc_human_higher": a,
            "direction": "human_higher" if a > 0.5 else "model_higher",
            "human_mean": float(h.mean()),
            "model_mean": float(m.mean()),
            "frac_nonzero": float(np.mean(mat[k] != 0)),
        }

    # Hypothesis verdicts
    rho_ker_hard = abs(corr["r_d_ker__r_d_hard"]["spearman"])
    rho_ker_rb = abs(corr["r_d_ker__r_b_cb"]["spearman"])
    rho_rb_cb_cont = abs(corr["r_b_cb__r_b_cont"]["spearman"])
    rho_ker_rb_cont = abs(corr["r_d_ker__r_b_cont"]["spearman"])
    verdict = {
        "collapse_ker_vs_hard": rho_ker_hard >= 0.7,
        "collapse_ker_vs_rb_codebook": rho_ker_rb >= 0.7,
        "rb_codebook_equals_continuous": rho_rb_cb_cont >= 0.7,
        "ker_tracks_continuous_rb": rho_ker_rb_cont >= 0.7,
        "note": (
            "collapse := |Spearman|≥0.7. If ker≢hard and ker≢r_b_cb, codebook "
            "is not forcing all metrics to one signal. If r_b_cb≈r_b_cont, the "
            "codebook soft-cluster is redundant for R_B on this receiver."
        ),
    }

    report = {
        "n_texts": len(records),
        "n_pairs": len(records) // 2,
        "model": MODEL,
        "n_tasks": args.n_tasks,
        "n_samples": args.n_samples,
        "correlations": corr,
        "separation": sep,
        "verdict": verdict,
        "gate_open_rate_ker": float(np.mean([r["r_d_ker_gate"] for r in records])),
        "gate_open_rate_hard": float(np.mean([r["r_d_hard_gate"] for r in records])),
        "r_creativity_nonzero_rate": float(np.mean([r["r_creativity"] != 0 for r in records])),
        "records_sample": records[:8],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_md(report)
    print(json.dumps({"correlations": corr, "verdict": verdict, "separation": sep}, indent=2))
    print("wrote", OUT_JSON, OUT_MD)


def write_md(report: dict) -> None:
    c = report["correlations"]
    s = report["separation"]
    v = report["verdict"]

    def row(a, b):
        k = f"{a}__{b}"
        return f"| {a} vs {b} | {c[k]['spearman']:.3f} | {c[k]['pearson']:.3f} |"

    lines = [
        "# Smoke: R_D / R_B / R_creativity collapse?",
        "",
        f"n={report['n_texts']} texts ({report['n_pairs']} human↔{report['model']} pairs), "
        f"HashReceiver, stub CUE, {report['n_tasks']} tasks × {report['n_samples']} samples.",
        "",
        "## Hypothesis",
        "",
        "Codebook may be unnecessary for R_D, but shared codebook soft-clustering "
        "in R_B could make metrics measure the **same** thing. Collapse threshold: "
        "|Spearman| ≥ 0.7.",
        "",
        "## Correlations (Spearman / Pearson)",
        "",
        "| Pair | Spearman | Pearson |",
        "|------|----------|---------|",
        row("r_d_ker", "r_d_hard"),
        row("r_d_ker", "r_b_cb"),
        row("r_d_ker", "r_b_cont"),
        row("r_d_hard", "r_b_cb"),
        row("r_b_cb", "r_b_cont"),
        row("r_d_ker", "r_creativity"),
        row("r_b_cb", "r_creativity"),
        "",
        "## Human vs LLM separation",
        "",
        "| Metric | AUC(H>M) | Direction | human μ | model μ | nonzero |",
        "|--------|----------|-----------|---------|---------|---------|",
    ]
    for k, info in s.items():
        lines.append(
            f"| {k} | {info['auc_human_higher']:.3f} | {info['direction']} | "
            f"{info['human_mean']:.4g} | {info['model_mean']:.4g} | "
            f"{info['frac_nonzero']:.2f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- ker vs hard collapse: **{v['collapse_ker_vs_hard']}**",
        f"- ker vs R_B(codebook) collapse: **{v['collapse_ker_vs_rb_codebook']}**",
        f"- R_B codebook ≈ continuous: **{v['rb_codebook_equals_continuous']}**",
        f"- ker tracks continuous R_B: **{v['ker_tracks_continuous_rb']}**",
        f"- ker d_gate open rate: {report['gate_open_rate_ker']:.2f}; "
        f"hard: {report['gate_open_rate_hard']:.2f}; "
        f"R_creativity nonzero: {report['r_creativity_nonzero_rate']:.2f}",
        "",
        v["note"],
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
