#!/usr/bin/env python3
"""
Build poetry_v2 Parzen / kernel R_D contexts; calibrate σ and δ_D on negatives.

Reuses MiniLM embedding cache from soft build when available.
Never uses human/LLM labels to choose σ or δ_D.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from creativegainbench.ideas.idea_extractor import (  # noqa: E402
    IdeaBoundaryDetector,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.delta_d import (  # noqa: E402
    quantile,
    write_delta_d_thresholds,
)
from creativegainbench.metrics.deformation import (  # noqa: E402
    KernelDomainDeformationContext,
    build_kernel_domain_context,
)
from creativegainbench.metrics.feasibility import feasibility_bit  # noqa: E402
from creativegainbench.metrics.kernel_probe_ce import (  # noqa: E402
    kernel_deformation_gain,
)
from creativegainbench.metrics.count_ngram import lambda_d_normalize  # noqa: E402

ARTIFACTS = REPO / "src/creativegainbench/artifacts/poetry_v2"
CORPUS = (
    REPO / "experiments/science_loop/runs/RD-SOFT-01/corpus_by_domain.jsonl"
)
EMB_CACHE = ARTIFACTS / "soft_emb_cache"
SEED = 42
MIN_NEG = 50
NEG_TYPES = ("exact_h_member", "pad", "shuffle")
MAX_BANK = 4096


def _load_soft_helpers():
    spec = importlib.util.spec_from_file_location(
        "build_soft_poetry_contexts",
        REPO / "scripts" / "build_soft_poetry_contexts.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def median_pairwise_dist(bank: torch.Tensor, sample: int = 512, seed: int = SEED) -> float:
    """Median pairwise L2 among a subsample — used to scale σ grid."""
    n = bank.size(0)
    if n < 2:
        return 1.0
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=rng)[: min(sample, n)]
    sub = bank[idx]
    d = torch.cdist(sub, sub)
    # upper triangle
    iu = torch.triu_indices(d.size(0), d.size(0), offset=1)
    vals = d[iu[0], iu[1]]
    return float(vals.median().item()) if vals.numel() else 1.0


def score_from_emb(
    emb: torch.Tensor, ctx: KernelDomainDeformationContext
) -> float:
    if emb.size(0) == 0:
        return 0.0
    raw = kernel_deformation_gain(emb, ctx.probe_emb_seqs, ctx.base_lm)
    return float(
        lambda_d_normalize(
            raw,
            n_probes=ctx.n_probes,
            vocab_size=ctx.vocab_size,
            mean_probe_len=ctx.mean_probe_len,
        )
    )


def save_kernel_ctx(path: Path, ctx: KernelDomainDeformationContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "domain": ctx.domain,
            "sigma": ctx.sigma,
            "embedding_dim": ctx.embedding_dim,
            "max_bank": ctx.max_bank,
            "bank": ctx.base_lm.bank.detach().cpu(),
            "eps": ctx.base_lm.eps,
            "probe_texts": ctx.probe_texts,
            "probe_emb_seqs": [s.detach().cpu() for s in ctx.probe_emb_seqs],
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sigma", type=float, default=None, help="Fix σ (skip grid)")
    parser.add_argument("--max-bank", type=int, default=MAX_BANK)
    parser.add_argument("--n-neg-per-type", type=int, default=40)
    parser.add_argument("--domains", default=None)
    args = parser.parse_args()

    helpers = _load_soft_helpers()
    if not CORPUS.exists():
        raise SystemExit(f"Missing {CORPUS}")

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    by_dom: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_dom[int(r["domain"])][r["split"]].append(r)

    domain_ids = sorted(by_dom)
    if args.domains:
        want = {int(x) for x in args.domains.split(",")}
        domain_ids = [d for d in domain_ids if d in want]

    codebook_state = torch.load(
        ARTIFACTS / "idea_codebook.pt", map_location="cpu", weights_only=True
    )
    codebook = IdeaCodebook(centroids=codebook_state["centroids"])
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable — falling back to cpu", flush=True)
        device = "cpu"
    encoder = build_span_encoder("minilm", device=device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(ARTIFACTS / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()

    usable = [d for d in domain_ids if by_dom[d]["train"] and by_dom[d]["probe"]]
    emb_cache: dict[int, dict] = {}
    EMB_CACHE.mkdir(parents=True, exist_ok=True)

    for d in usable:
        train_texts = [r["body"] for r in by_dom[d]["train"]]
        probe_texts = [r["body"] for r in by_dom[d]["probe"]]
        train_path = helpers.emb_cache_path(EMB_CACHE, d, "train")
        probe_path = helpers.emb_cache_path(EMB_CACHE, d, "probe")
        train_embs = helpers.load_emb_cache(train_path, train_texts)
        if train_embs is None:
            print(f"Encoding domain {d} train...", flush=True)
            train_embs = helpers.encode_embeddings_list(
                train_texts, encoder=encoder, codebook=codebook, boundary=boundary
            )
            helpers.save_emb_cache(train_path, train_texts, train_embs)
        else:
            print(f"Loaded train emb cache domain {d}", flush=True)
        probe_embs = helpers.load_emb_cache(probe_path, probe_texts)
        if probe_embs is None:
            print(f"Encoding domain {d} probe...", flush=True)
            probe_embs = helpers.encode_embeddings_list(
                probe_texts, encoder=encoder, codebook=codebook, boundary=boundary
            )
            helpers.save_emb_cache(probe_path, probe_texts, probe_embs)
        else:
            print(f"Loaded probe emb cache domain {d}", flush=True)
        emb_cache[d] = {
            "train_texts": train_texts,
            "probe_texts": probe_texts,
            "train_embs": train_embs,
            "probe_embs": probe_embs,
        }

    # Scale σ grid from median pairwise distance on first usable domain bank.
    sample_bank = torch.cat(
        [e for e in emb_cache[usable[0]]["train_embs"] if e.numel() > 0], dim=0
    )
    if sample_bank.size(0) > args.max_bank:
        rng = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(sample_bank.size(0), generator=rng)[: args.max_bank]
        sample_bank = sample_bank[idx]
    med = median_pairwise_dist(sample_bank)
    sigma_grid = [med * m for m in (0.25, 0.5, 1.0, 1.5, 2.0)]
    print(f"median pairwise L2={med:.4f}; σ grid={sigma_grid}", flush=True)

    if args.sigma is not None:
        best_sigma = float(args.sigma)
        sigma_report: dict = {"fixed": best_sigma}
    else:
        print("Calibrating σ on negatives...", flush=True)
        cal_domains = usable[:4]
        scores: dict[float, list[float]] = {s: [] for s in sigma_grid}
        for d in cal_domains:
            cache = emb_cache[d]
            eval_bodies = [r["body"] for r in by_dom[d]["eval"]] or cache["train_texts"]
            target = max(200, min(4000, helpers._median_chars(eval_bodies)))
            negs = helpers.build_negatives(
                cache["train_texts"],
                cache["probe_texts"],
                target_chars=target,
                n_per_type=min(args.n_neg_per_type, 25),
                seed=SEED + d,
            )
            neg_texts = [t for ctype, t in negs if ctype in NEG_TYPES and feasibility_bit(t)]
            print(f"  domain {d}: encoding {len(neg_texts)} negatives...", flush=True)
            neg_embs = helpers.encode_embeddings_list(
                neg_texts, encoder=encoder, codebook=codebook, boundary=boundary
            )
            for sigma in sigma_grid:
                ctx = build_kernel_domain_context(
                    domain=d,
                    train_embs=cache["train_embs"],
                    probe_embs=cache["probe_embs"],
                    probe_texts=cache["probe_texts"],
                    sigma=sigma,
                    max_bank=args.max_bank,
                    seed=SEED + d,
                )
                for emb in neg_embs:
                    scores[sigma].append(score_from_emb(emb, ctx))

        best_sigma, best_gap = float(sigma_grid[0]), 1e9
        sigma_report = {"median_pairwise_l2": med}
        for sigma, vals in scores.items():
            if len(vals) < 20:
                sigma_report[str(sigma)] = {"n": len(vals), "skip": True}
                continue
            thr = quantile(vals, 0.95) + 1e-6
            rate = sum(1 for v in vals if v > thr) / len(vals)
            gap = abs(rate - 0.05)
            sigma_report[str(sigma)] = {
                "n": len(vals),
                "delta_d_95": thr,
                "neg_pass_rate": rate,
                "gap_to_0.05": gap,
            }
            if gap < best_gap:
                best_gap, best_sigma = gap, float(sigma)
        print(f"Selected σ={best_sigma}", flush=True)

    thresholds: dict[str, dict] = {}
    meta = {
        "sigma": best_sigma,
        "backend": "kernel_parzen",
        "max_bank": args.max_bank,
        "embedding_dim": int(sample_bank.size(1)),
        "sigma_calibration": sigma_report,
        "domains": {},
    }

    for d in usable:
        cache = emb_cache[d]
        eval_bodies = [r["body"] for r in by_dom[d]["eval"]] or cache["train_texts"]
        target = max(200, min(4000, helpers._median_chars(eval_bodies)))
        print(f"Building kernel ctx domain {d} at σ={best_sigma}...", flush=True)
        ctx = build_kernel_domain_context(
            domain=d,
            train_embs=cache["train_embs"],
            probe_embs=cache["probe_embs"],
            probe_texts=cache["probe_texts"],
            sigma=best_sigma,
            max_bank=args.max_bank,
            seed=SEED + d,
        )
        out_path = ARTIFACTS / f"domain_{d}_kernel_ctx.pt"
        save_kernel_ctx(out_path, ctx)

        negs = helpers.build_negatives(
            cache["train_texts"],
            cache["probe_texts"],
            target_chars=target,
            n_per_type=args.n_neg_per_type,
            seed=SEED + 100 + d,
        )
        neg_texts = [t for ctype, t in negs if ctype in NEG_TYPES and feasibility_bit(t)]
        neg_embs = helpers.encode_embeddings_list(
            neg_texts, encoder=encoder, codebook=codebook, boundary=boundary
        )
        neg_vals = [score_from_emb(e, ctx) for e in neg_embs]
        if len(neg_vals) >= MIN_NEG:
            thr = quantile(neg_vals, 0.95) + 1e-6
            rate = sum(1 for v in neg_vals if v > thr) / len(neg_vals)
            thresholds[str(d)] = {
                "delta_d_95": thr,
                "n_neg": len(neg_vals),
                "quantile": 0.95,
                "neg_pass_rate": rate,
            }
            meta["domains"][str(d)] = {
                "n_train": len(cache["train_texts"]),
                "n_probe": len(cache["probe_texts"]),
                "n_eval": len(by_dom[d]["eval"]),
                "n_bank": ctx.base_lm.n_bank,
                "target_chars": target,
                "kernel_ctx": out_path.name,
            }
            print(
                f"  δ_D={thr:.6g} n_neg={len(neg_vals)} pass_rate={rate:.3f} "
                f"bank={ctx.base_lm.n_bank}",
                flush=True,
            )
        else:
            print(f"  only {len(neg_vals)} negs — skip δ_D", flush=True)

    thr_path = ARTIFACTS / "kernel_delta_d_thresholds.json"
    write_delta_d_thresholds(thr_path, thresholds)
    meta_path = ARTIFACTS / "kernel_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # Freeze JSON hashes for in-repo verification; .pt contexts are local.
    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    local_contexts = [info["kernel_ctx"] for info in meta["domains"].values()]
    committed = {
        "kernel_meta.json": _sha256(meta_path),
        "kernel_delta_d_thresholds.json": _sha256(thr_path),
    }
    manifest_path = ARTIFACTS / "kernel_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "kernel_parzen",
                "sigma": best_sigma,
                "max_bank": args.max_bank,
                "files": committed,
                "local_contexts": local_contexts,
                "note": (
                    "Rebuild domain_*_kernel_ctx.pt with this script "
                    "(--device cuda recommended); files are gitignored."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {thr_path}, {meta_path}, and {manifest_path}")


if __name__ == "__main__":
    main()
