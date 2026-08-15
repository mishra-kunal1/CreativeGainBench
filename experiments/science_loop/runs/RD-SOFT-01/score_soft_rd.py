#!/usr/bin/env python3
"""
Score soft R_D on poetry_v2 eval (human + model generations).

No Ollama: uses dump texts + frozen MiniLM/codebook + soft domain contexts.
Writes paired_eval_soft.jsonl under RD-SOFT-01.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector  # noqa: E402
from creativegainbench.ideas.idea_ngram import IdeaCodebook  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.count_ngram import lambda_d_normalize  # noqa: E402
from creativegainbench.metrics.feasibility import feasibility_bit  # noqa: E402
from creativegainbench.metrics.soft_count_ngram import (  # noqa: E402
    soft_assign,
    soft_deformation_gain,
)

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus_by_domain.jsonl"
ARTIFACTS = REPO / "src/creativegainbench/artifacts/poetry_v2"
OUT = HERE / "paired_eval_soft.jsonl"
MODELS = ("gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b")


def _load_build_helpers():
    spec = importlib.util.spec_from_file_location(
        "build_soft_poetry_contexts",
        REPO / "scripts" / "build_soft_poetry_contexts.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _clip(text: str, n: int | None) -> str:
    t = (text or "").strip()
    if not n or n <= 0 or len(t) <= n:
        return t
    cut = t[:n]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return cut.strip()


def _median_chars(bodies: list[str]) -> int:
    lens = [len((b or "").strip()) for b in bodies if (b or "").strip()]
    return int(statistics.median(lens)) if lens else 400


def load_soft_ctx(path: Path):
    from creativegainbench.metrics.deformation import SoftDomainDeformationContext
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


def score_from_emb(emb: torch.Tensor, ctx, codebook, delta_d: float, feasible: bool) -> dict:
    if not feasible or emb.size(0) == 0:
        return {
            "r_d_raw": 0.0,
            "r_d_norm": 0.0,
            "r_d_gate": 0.0,
            "feasible": False,
            "y_n_symbols": 0,
        }
    y_soft = soft_assign(emb, codebook.centroids.detach().cpu().float(), tau=ctx.tau)
    raw = soft_deformation_gain(y_soft, ctx.probe_soft_seqs, ctx.base_lm)
    norm = float(
        lambda_d_normalize(
            raw,
            n_probes=ctx.n_probes,
            vocab_size=ctx.vocab_size,
            mean_probe_len=ctx.mean_probe_len,
        )
    )
    return {
        "r_d_raw": float(raw),
        "r_d_norm": norm,
        "r_d_gate": 1.0 if norm > delta_d else 0.0,
        "feasible": True,
        "y_n_symbols": int(emb.size(0)),
        "n_probes": ctx.n_probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated model keys (default: all four)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap eval poems (0=all)")
    args = parser.parse_args()
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    helpers = _load_build_helpers()

    meta = json.loads((ARTIFACTS / "soft_meta.json").read_text())
    thr = json.loads((ARTIFACTS / "soft_delta_d_thresholds.json").read_text())
    codebook = IdeaCodebook(
        centroids=torch.load(
            ARTIFACTS / "idea_codebook.pt", map_location="cpu", weights_only=True
        )["centroids"]
    )
    encoder = build_span_encoder("minilm", device=args.device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(ARTIFACTS / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    soft_domains = {int(k) for k in meta.get("domains", {})}
    eval_rows = [
        r
        for r in rows
        if r.get("split") == "eval"
        and r.get("domain") is not None
        and int(r["domain"]) in soft_domains
    ]
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    ctx_by_d = {
        d: load_soft_ctx(ARTIFACTS / f"domain_{d}_soft_ctx.pt")
        for d in sorted(soft_domains)
        if (ARTIFACTS / f"domain_{d}_soft_ctx.pt").exists()
    }

    by_dom_bodies: dict[int, list[str]] = {}
    for r in rows:
        if r.get("split") == "eval" and r.get("domain") is not None:
            by_dom_bodies.setdefault(int(r["domain"]), []).append(r.get("body") or "")
    clip_by_d = {
        d: max(200, min(4000, _median_chars(bodies)))
        for d, bodies in by_dom_bodies.items()
    }

    # Flatten all texts to encode once (batched).
    jobs: list[tuple[int, str, str]] = []  # (eval_idx, role, text)
    for i, r in enumerate(eval_rows):
        d = int(r["domain"])
        if d not in ctx_by_d:
            continue
        clip = clip_by_d.get(d, 800)
        jobs.append((i, "human", _clip(r.get("body") or "", clip)))
        gens = r.get("generations") or {}
        for m in models:
            if m in gens:
                jobs.append((i, m, _clip(gens[m], clip)))

    print(f"Encoding {len(jobs)} texts for {len(eval_rows)} eval poems...", flush=True)
    texts = [t for _, _, t in jobs]
    embs = helpers.encode_embeddings_list(
        texts, encoder=encoder, codebook=codebook, boundary=boundary
    )

    # Assemble records
    recs: dict[int, dict] = {}
    for (i, role, _text), emb in zip(jobs, embs):
        r = eval_rows[i]
        d = int(r["domain"])
        ctx = ctx_by_d[d]
        delta = float(thr.get(str(d), {}).get("delta_d_95", 0.0))
        if i not in recs:
            recs[i] = {
                "poem_id": r["poem_id"],
                "domain": d,
                "source": r.get("source"),
                "tau": meta["tau"],
                "delta_d": delta,
                "length_clip_chars": clip_by_d.get(d, 800),
                "human": None,
                "models": {},
            }
        scored = score_from_emb(
            emb, ctx, codebook, delta, bool(feasibility_bit(_text))
        )
        if role == "human":
            recs[i]["human"] = scored
        else:
            recs[i]["models"][role] = scored

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with OUT.open("w") as fh:
        for i in sorted(recs):
            rec = recs[i]
            if rec["human"] is None:
                continue
            fh.write(json.dumps(rec) + "\n")
            n_out += 1
    print(f"Wrote {n_out} rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
