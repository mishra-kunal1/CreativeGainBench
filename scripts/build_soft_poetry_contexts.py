#!/usr/bin/env python3
"""
Build poetry_v2 Soft ProbeCompressor contexts and calibrate τ / δ_D on negatives.

Never uses human/LLM labels to choose τ or δ_D.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from creativegainbench.ideas.idea_extractor import (  # noqa: E402
    IdeaBoundaryDetector,
    poetry_line_splitter,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.delta_d import (  # noqa: E402
    quantile,
    write_delta_d_thresholds,
)
from creativegainbench.metrics.deformation import SoftDomainDeformationContext  # noqa: E402
from creativegainbench.metrics.feasibility import feasibility_bit  # noqa: E402

ARTIFACTS = REPO / "src/creativegainbench/artifacts/poetry_v2"
CORPUS = (
    REPO
    / "experiments/science_loop/runs/RD-SOFT-01/corpus_by_domain.jsonl"
)
TAU_GRID = (0.25, 0.5, 1.0, 2.0)
NEG_TYPES = ("exact_h_member", "pad", "shuffle")
MIN_NEG = 50
SEED = 42


def load_corpus(path: Path = CORPUS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _median_chars(bodies: list[str]) -> int:
    lens = [len((b or "").strip()) for b in bodies if (b or "").strip()]
    return int(statistics.median(lens)) if lens else 400


def _length_match(text: str, target: int, *, tol: float = 0.2) -> str:
    t = (text or "").strip()
    if target <= 0:
        return t
    lo = max(1, int(target * (1.0 - tol)))
    hi = max(lo, int(target * (1.0 + tol)))
    if len(t) < lo:
        lines = [ln for ln in t.splitlines() if ln.strip()] or [t]
        out, i = t, 0
        while len(out) < lo and i < 500:
            out = (out + "\n" + lines[i % len(lines)]).strip()
            i += 1
        t = out
    if len(t) > hi:
        cut = t[:hi]
        if "\n" in cut:
            cut = cut.rsplit("\n", 1)[0]
        t = cut.strip()
    return t


def _shuffle_lines(text: str, rng: random.Random) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return text
    rng.shuffle(lines)
    return "\n".join(lines)


def _pad(text: str, target: int) -> str:
    filler = "The soft wind moves across the quiet field again."
    t = (text or "").strip()
    while len(t) < target:
        t = (t + "\n" + filler).strip()
        if len(t) > target * 2:
            break
    return _length_match(t, target)


def build_negatives(
    train_texts: list[str],
    probe_texts: list[str],
    *,
    target_chars: int,
    n_per_type: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Return (construct_type, text) negatives — no class labels."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    pool = [t for t in train_texts if (t or "").strip()]
    if not pool:
        pool = [t for t in probe_texts if (t or "").strip()]
    if not pool:
        return out

    # exact_h_member
    members = list(pool)
    rng.shuffle(members)
    for t in members[:n_per_type]:
        out.append(("exact_h_member", _length_match(t, target_chars)))

    # shuffle
    for t in members[:n_per_type]:
        out.append(("shuffle", _length_match(_shuffle_lines(t, rng), target_chars)))

    # pad
    for t in members[:n_per_type]:
        out.append(("pad", _pad(t, target_chars)))

    return out


def save_soft_ctx(path: Path, ctx: SoftDomainDeformationContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "domain": ctx.domain,
            "tau": ctx.tau,
            "vocab_size": ctx.vocab_size,
            "order": ctx.order,
            "counts": ctx.base_lm.counts.detach().cpu(),
            "eps": ctx.base_lm.eps,
            "probe_texts": ctx.probe_texts,
            "probe_soft_seqs": [s.detach().cpu() for s in ctx.probe_soft_seqs],
        },
        path,
    )


def load_soft_ctx(path: Path) -> SoftDomainDeformationContext:
    from creativegainbench.metrics.soft_count_ngram import SoftCountLM

    blob = torch.load(path, map_location="cpu", weights_only=False)
    lm = SoftCountLM(
        vocab_size=int(blob["vocab_size"]),
        counts=blob["counts"],
        eps=float(blob.get("eps", 1e-6)),
    )
    return SoftDomainDeformationContext(
        domain=int(blob["domain"]),
        base_lm=lm,
        probe_texts=list(blob["probe_texts"]),
        probe_soft_seqs=list(blob["probe_soft_seqs"]),
        vocab_size=int(blob["vocab_size"]),
        tau=float(blob["tau"]),
        order=int(blob.get("order", 1)),
    )


def _ideas_from_span_embeds(
    spans: list[str],
    span_embeds: torch.Tensor,
    boundary,
    *,
    boundary_threshold: float = 0.5,
) -> torch.Tensor:
    """Mirror extract_ideas boundary merge; return (n_ideas, d) or empty."""
    if not spans:
        dim = int(span_embeds.size(-1)) if span_embeds.ndim == 2 else 384
        return torch.zeros(0, dim, dtype=torch.float32)
    with torch.no_grad():
        bd_device = next(boundary.parameters()).device
        embeds = span_embeds.to(bd_device)
        probs = boundary(embeds)
    buffer_embeds: list[torch.Tensor] = []
    ideas: list[torch.Tensor] = []
    for embed, p in zip(embeds, probs):
        buffer_embeds.append(embed)
        if float(p.item()) > boundary_threshold:
            ideas.append(torch.stack(buffer_embeds).mean(dim=0).detach().cpu().float())
            buffer_embeds = []
    if buffer_embeds:
        ideas.append(torch.stack(buffer_embeds).mean(dim=0).detach().cpu().float())
    return torch.stack(ideas, dim=0) if ideas else torch.zeros(
        0, embeds.size(-1), dtype=torch.float32
    )


def encode_embeddings_list(
    texts: list[str],
    *,
    encoder,
    codebook,
    boundary,
    batch_size: int = 256,
) -> list[torch.Tensor]:
    """
    Batch MiniLM over all poetry-line spans, then apply the frozen boundary
    detector per text. Much faster than one ST.encode call per poem.
    """
    _ = codebook  # API parity with callers
    span_lists = [poetry_line_splitter(t) for t in texts]
    flat: list[str] = []
    owners: list[int] = []
    for i, spans in enumerate(span_lists):
        for s in spans:
            flat.append(s)
            owners.append(i)

    dim = int(getattr(encoder, "embedding_dim", 384))
    if not flat:
        return [torch.zeros(0, dim, dtype=torch.float32) for _ in texts]

    all_emb = torch.zeros(len(flat), dim, dtype=torch.float32)
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        with torch.no_grad():
            emb = encoder(chunk).detach().cpu().float()
        all_emb[start : start + len(chunk)] = emb
        if (start // batch_size) % 5 == 0:
            print(
                f"    span-batch {min(start + batch_size, len(flat))}/{len(flat)}",
                flush=True,
            )

    per_text_spans: list[list[torch.Tensor]] = [[] for _ in texts]
    for row_i, text_i in enumerate(owners):
        per_text_spans[text_i].append(all_emb[row_i])

    out: list[torch.Tensor] = []
    for i, spans in enumerate(span_lists):
        if not spans:
            out.append(torch.zeros(0, dim, dtype=torch.float32))
            continue
        span_embeds = torch.stack(per_text_spans[i], dim=0)
        out.append(_ideas_from_span_embeds(spans, span_embeds, boundary))
        if (i + 1) % 200 == 0:
            print(f"    boundary {i+1}/{len(texts)}", flush=True)
    return out


def emb_cache_path(cache_dir: Path, domain: int, kind: str) -> Path:
    return cache_dir / f"domain_{domain}_{kind}_embs.pt"


def save_emb_cache(path: Path, texts: list[str], embs: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"texts": texts, "embs": [e.detach().cpu() for e in embs]}, path)


def load_emb_cache(path: Path, texts: list[str]) -> list[torch.Tensor] | None:
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("texts") != texts:
        return None
    return list(blob["embs"])


def soft_ctx_from_embeddings(
    *,
    domain: int,
    train_embs: list[torch.Tensor],
    probe_embs: list[torch.Tensor],
    probe_texts: list[str],
    codebook: IdeaCodebook,
    tau: float,
) -> SoftDomainDeformationContext:
    from creativegainbench.metrics.soft_count_ngram import soft_assign, train_soft_count_lm

    cents = codebook.centroids.detach().cpu().float()
    train_soft = [
        soft_assign(e, cents, tau=tau) for e in train_embs if e.size(0) > 0
    ]
    probe_pairs = []
    for t, e in zip(probe_texts, probe_embs):
        if e.size(0) == 0:
            continue
        probe_pairs.append((t, soft_assign(e, cents, tau=tau)))
    if not train_soft or not probe_pairs:
        raise ValueError(f"Domain {domain}: empty soft sequences")
    base_lm = train_soft_count_lm(train_soft, codebook.vocab_size)
    return SoftDomainDeformationContext(
        domain=domain,
        base_lm=base_lm,
        probe_texts=[t for t, _ in probe_pairs],
        probe_soft_seqs=[s for _, s in probe_pairs],
        vocab_size=codebook.vocab_size,
        tau=float(tau),
        order=1,
    )


def score_from_emb(
    emb: torch.Tensor,
    ctx: SoftDomainDeformationContext,
    codebook: IdeaCodebook,
) -> float:
    from creativegainbench.metrics.count_ngram import lambda_d_normalize
    from creativegainbench.metrics.soft_count_ngram import soft_assign, soft_deformation_gain

    if emb.size(0) == 0:
        return 0.0
    y_soft = soft_assign(emb, codebook.centroids.detach().cpu().float(), tau=ctx.tau)
    raw = soft_deformation_gain(y_soft, ctx.probe_soft_seqs, ctx.base_lm)
    return float(
        lambda_d_normalize(
            raw,
            n_probes=ctx.n_probes,
            vocab_size=ctx.vocab_size,
            mean_probe_len=ctx.mean_probe_len,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=None, help="Fix τ (skip grid)")
    parser.add_argument("--n-neg-per-type", type=int, default=40)
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated domain ids (default: all with train+probe)",
    )
    args = parser.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"Missing {CORPUS}; run extract_corpus.py first")

    rows = load_corpus()
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
    encoder = build_span_encoder("minilm", device=args.device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(ARTIFACTS / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()

    usable = [d for d in domain_ids if by_dom[d]["train"] and by_dom[d]["probe"]]
    emb_disk = ARTIFACTS / "soft_emb_cache"
    emb_disk.mkdir(parents=True, exist_ok=True)

    # Precompute train/probe embeddings once per domain (MiniLM is the bottleneck).
    emb_cache: dict[int, dict] = {}
    for d in usable:
        train_texts = [r["body"] for r in by_dom[d]["train"]]
        probe_texts = [r["body"] for r in by_dom[d]["probe"]]
        print(f"Encoding domain {d} train={len(train_texts)} probe={len(probe_texts)}...", flush=True)
        train_path = emb_cache_path(emb_disk, d, "train")
        probe_path = emb_cache_path(emb_disk, d, "probe")
        train_embs = load_emb_cache(train_path, train_texts)
        if train_embs is None:
            train_embs = encode_embeddings_list(
                train_texts, encoder=encoder, codebook=codebook, boundary=boundary
            )
            save_emb_cache(train_path, train_texts, train_embs)
            print(f"  cached {train_path.name}", flush=True)
        else:
            print(f"  loaded {train_path.name}", flush=True)
        probe_embs = load_emb_cache(probe_path, probe_texts)
        if probe_embs is None:
            probe_embs = encode_embeddings_list(
                probe_texts, encoder=encoder, codebook=codebook, boundary=boundary
            )
            save_emb_cache(probe_path, probe_texts, probe_embs)
            print(f"  cached {probe_path.name}", flush=True)
        else:
            print(f"  loaded {probe_path.name}", flush=True)
        emb_cache[d] = {
            "train_texts": train_texts,
            "probe_texts": probe_texts,
            "train_embs": train_embs,
            "probe_embs": probe_embs,
        }

    # --- choose τ on negatives from first few domains (label-free) ---
    if args.tau is not None:
        best_tau = float(args.tau)
        tau_report: dict = {"fixed": best_tau}
    else:
        print("Calibrating τ on negatives (cached embeddings)...", flush=True)
        cal_domains = usable[:4]
        tau_scores: dict[float, list[float]] = {t: [] for t in TAU_GRID}
        for d in cal_domains:
            cache = emb_cache[d]
            eval_bodies = [r["body"] for r in by_dom[d]["eval"]] or cache["train_texts"]
            target = max(200, min(4000, _median_chars(eval_bodies)))
            negs = build_negatives(
                cache["train_texts"],
                cache["probe_texts"],
                target_chars=target,
                n_per_type=min(args.n_neg_per_type, 25),
                seed=SEED + d,
            )
            neg_texts = [t for ctype, t in negs if ctype in NEG_TYPES and feasibility_bit(t)]
            print(f"  domain {d}: encoding {len(neg_texts)} negatives...", flush=True)
            neg_embs = encode_embeddings_list(
                neg_texts, encoder=encoder, codebook=codebook, boundary=boundary
            )
            for tau in TAU_GRID:
                ctx = soft_ctx_from_embeddings(
                    domain=d,
                    train_embs=cache["train_embs"],
                    probe_embs=cache["probe_embs"],
                    probe_texts=cache["probe_texts"],
                    codebook=codebook,
                    tau=tau,
                )
                for emb in neg_embs:
                    tau_scores[tau].append(score_from_emb(emb, ctx, codebook))

        best_tau, best_gap = float(TAU_GRID[0]), 1e9
        tau_report = {}
        for tau, vals in tau_scores.items():
            if len(vals) < 20:
                tau_report[str(tau)] = {"n": len(vals), "skip": True}
                continue
            thr = quantile(vals, 0.95) + 1e-6
            rate = sum(1 for v in vals if v > thr) / len(vals)
            gap = abs(rate - 0.05)
            tau_report[str(tau)] = {
                "n": len(vals),
                "delta_d_95": thr,
                "neg_pass_rate": rate,
                "gap_to_0.05": gap,
            }
            if gap < best_gap:
                best_gap, best_tau = gap, float(tau)
        print(f"Selected τ={best_tau} (report={tau_report})", flush=True)

    thresholds: dict[str, dict] = {}
    soft_meta = {
        "tau": best_tau,
        "order": 1,
        "vocab_size": codebook.vocab_size,
        "backend": "soft_count",
        "tau_calibration": tau_report,
        "domains": {},
    }

    for d in usable:
        cache = emb_cache[d]
        eval_bodies = [r["body"] for r in by_dom[d]["eval"]] or cache["train_texts"]
        target = max(200, min(4000, _median_chars(eval_bodies)))
        print(f"Building soft ctx domain {d} at τ={best_tau}...", flush=True)
        ctx = soft_ctx_from_embeddings(
            domain=d,
            train_embs=cache["train_embs"],
            probe_embs=cache["probe_embs"],
            probe_texts=cache["probe_texts"],
            codebook=codebook,
            tau=best_tau,
        )
        out_path = ARTIFACTS / f"domain_{d}_soft_ctx.pt"
        save_soft_ctx(out_path, ctx)

        negs = build_negatives(
            cache["train_texts"],
            cache["probe_texts"],
            target_chars=target,
            n_per_type=args.n_neg_per_type,
            seed=SEED + 100 + d,
        )
        neg_texts = [t for ctype, t in negs if ctype in NEG_TYPES and feasibility_bit(t)]
        neg_embs = encode_embeddings_list(
            neg_texts, encoder=encoder, codebook=codebook, boundary=boundary
        )
        neg_vals = [score_from_emb(e, ctx, codebook) for e in neg_embs]
        if len(neg_vals) >= MIN_NEG:
            thr = quantile(neg_vals, 0.95) + 1e-6
            rate = sum(1 for v in neg_vals if v > thr) / len(neg_vals)
            thresholds[str(d)] = {
                "delta_d_95": thr,
                "n_neg": len(neg_vals),
                "quantile": 0.95,
                "neg_pass_rate": rate,
            }
            soft_meta["domains"][str(d)] = {
                "n_train": len(cache["train_texts"]),
                "n_probe": len(cache["probe_texts"]),
                "n_eval": len(by_dom[d]["eval"]),
                "target_chars": target,
                "soft_ctx": str(out_path.name),
            }
            print(f"  δ_D={thr:.6g} n_neg={len(neg_vals)} pass_rate={rate:.3f}", flush=True)
        else:
            print(f"  only {len(neg_vals)} negs — skip δ_D", flush=True)

    thr_path = ARTIFACTS / "soft_delta_d_thresholds.json"
    write_delta_d_thresholds(thr_path, thresholds)
    meta_path = ARTIFACTS / "soft_meta.json"
    meta_path.write_text(json.dumps(soft_meta, indent=2) + "\n")
    print(f"Wrote {thr_path} and {meta_path}")


if __name__ == "__main__":
    main()
