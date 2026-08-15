"""
Phase 0–1: freeze poetry experiment splits and build poetry-v2 idea artifacts.

1. Embed human poem bodies (MiniLM), k-means into domain clusters.
2. Author-disjoint train / probe / eval splits per cluster.
3. Write domain_cluster + split columns to Postgres.
4. Fit idea codebook + CountNgram background corpora on train texts
   (poetry line splitter), persist under artifacts/poetry_v2/.

Usage:
  python scripts/prepare_poetry_experiment.py [--k 12] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import psycopg
import torch
import torch.nn as nn

from creativegainbench.ideas.idea_extractor import (
    IdeaBoundaryDetector,
    poetry_line_splitter,
)
from creativegainbench.ideas.span_encoder import build_span_encoder
from creativegainbench.metrics.deformation import build_domain_context

DEFAULT_DB = "postgres://poems:poems@localhost:5432/poems"
ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "creativegainbench"
    / "artifacts"
    / "poetry_v2"
)


def _migrate(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        ALTER TABLE poems ADD COLUMN IF NOT EXISTS domain_cluster INT;
        ALTER TABLE poems ADD COLUMN IF NOT EXISTS split TEXT;
        CREATE TABLE IF NOT EXISTS scores (
            poem_id UUID NOT NULL,
            side TEXT NOT NULL,
            metric_version TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (poem_id, side, metric_version)
        );
        """
    )
    conn.commit()


def _kmeans(points: torch.Tensor, k: int, seed: int, iters: int = 30) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = points.size(0)
    idx = torch.randperm(n, generator=g)[:k]
    centroids = points[idx].clone()
    for _ in range(iters):
        assign = torch.argmin(torch.cdist(points, centroids), dim=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centroids[j] = points[mask].mean(dim=0)
            else:
                centroids[j] = points[torch.randint(0, n, (1,), generator=g).item()]
    norms = torch.linalg.vector_norm(centroids, dim=1, keepdim=True).clamp_min(1e-8)
    return centroids / norms


def _author_disjoint_split(
    rows: list[tuple],
    *,
    probe_n: int,
    eval_frac: float,
    seed: int,
    max_probe_author_poems: int = 3,
) -> dict[str, str]:
    """
    rows: (id, author, body) for one domain.
    Returns id -> split in {train, probe, eval}.

    Whole-author assignment, preferring small authors for the probe set so a
    single prolific author cannot consume the entire probe budget. Caps poems
    taken per probe author; remainder of that author goes to train (still
    author-disjoint from eval).
    """
    rng = random.Random(seed)
    by_author: dict[str, list[tuple]] = defaultdict(list)
    for r in rows:
        by_author[r[1]].append(r)

    # Small authors first for probes (diversity, controlled size).
    authors_asc = sorted(by_author.keys(), key=lambda a: (len(by_author[a]), a))
    assignment: dict[str, str] = {}
    n_probe = 0
    probe_authors: set[str] = set()

    for author in authors_asc:
        if n_probe >= probe_n:
            break
        poems = list(by_author[author])
        rng.shuffle(poems)
        take = min(max_probe_author_poems, len(poems), probe_n - n_probe)
        for pid, *_ in poems[:take]:
            assignment[str(pid)] = "probe"
            n_probe += 1
        for pid, *_ in poems[take:]:
            assignment[str(pid)] = "train"
        probe_authors.add(author)

    # Remaining authors → eval then train (never share with probe authors).
    remaining = [a for a in by_author if a not in probe_authors]
    rng.shuffle(remaining)
    n_eval = 0
    target_eval = max(1, int(round(eval_frac * len(rows))))
    for author in remaining:
        poems = by_author[author]
        if n_eval < target_eval:
            for pid, *_ in poems:
                assignment[str(pid)] = "eval"
            n_eval += len(poems)
        else:
            for pid, *_ in poems:
                assignment[str(pid)] = "train"

    for r in rows:
        assignment.setdefault(str(r[0]), "train")
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_DB))
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-per-domain", type=int, default=18)
    parser.add_argument("--eval-frac", type=float, default=0.25)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(args.db) as conn:
        _migrate(conn)
        rows = conn.execute(
            "SELECT id, author, body, title FROM poems "
            "WHERE body IS NOT NULL AND btrim(body) <> '' "
            "AND llm_output IS NOT NULL AND btrim(llm_output) <> '' "
            "ORDER BY id"
        ).fetchall()
    print(f"loaded {len(rows)} paired poems", flush=True)

    encoder = build_span_encoder("minilm", device=args.device)
    print(f"encoding bodies on {args.device}...", flush=True)
    bodies = [r[2][:4000] for r in rows]
    # Batch encode whole poems as document embeddings for clustering.
    embeds_list = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(bodies), bs):
            batch = bodies[i : i + bs]
            # Use span encoder on truncated body as one span each.
            e = encoder(batch)
            embeds_list.append(e.cpu())
            if (i // bs) % 20 == 0:
                print(f"  encoded {min(i+bs, len(bodies))}/{len(bodies)}", flush=True)
    points = torch.cat(embeds_list, dim=0)
    norms = torch.linalg.vector_norm(points, dim=1, keepdim=True).clamp_min(1e-8)
    points = points / norms

    centroids = _kmeans(points, k=args.k, seed=args.seed)
    assign = torch.argmin(torch.cdist(points, centroids), dim=1).tolist()
    print("cluster sizes:", sorted(Counter(assign).items()), flush=True)

    # Per-domain author-disjoint splits.
    by_domain: dict[int, list[tuple]] = defaultdict(list)
    for (pid, author, body, title), d in zip(rows, assign):
        by_domain[d].append((pid, author, body, title))

    id_to_domain: dict[str, int] = {}
    id_to_split: dict[str, str] = {}
    for d, drows in by_domain.items():
        splits = _author_disjoint_split(
            [(r[0], r[1], r[2]) for r in drows],
            probe_n=args.probe_per_domain,
            eval_frac=args.eval_frac,
            seed=args.seed + d,
        )
        for pid, author, body, title in drows:
            sid = str(pid)
            id_to_domain[sid] = d
            id_to_split[sid] = splits[sid]

    # Write back to DB.
    with psycopg.connect(args.db) as conn:
        with conn.cursor() as cur:
            for pid, *_ in rows:
                sid = str(pid)
                cur.execute(
                    "UPDATE poems SET domain_cluster = %s, split = %s WHERE id = %s",
                    (id_to_domain[sid], id_to_split[sid], pid),
                )
        conn.commit()
        counts = conn.execute(
            "SELECT split, COUNT(*) FROM poems WHERE split IS NOT NULL GROUP BY split ORDER BY 1"
        ).fetchall()
        print("split counts:", counts, flush=True)
        dcounts = conn.execute(
            "SELECT domain_cluster, split, COUNT(*) FROM poems "
            "WHERE domain_cluster IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
        for row in dcounts:
            print(" ", row, flush=True)

    # Fit idea codebook on TRAIN texts only (poetry line splitter).
    print("fitting idea codebook on train split...", flush=True)
    train_texts = [
        r[2][:4000]
        for r in rows
        if id_to_split[str(r[0])] == "train"
    ]
    boundary = IdeaBoundaryDetector(hidden_dim=int(encoder.embedding_dim))
    with torch.no_grad():
        nn.init.zeros_(boundary.boundary_head.weight)
        # Mild bias so lines often become their own ideas but can merge.
        nn.init.constant_(boundary.boundary_head.bias, 0.5)
    boundary.eval()
    boundary = boundary.to(args.device)

    idea_embeds: list[torch.Tensor] = []
    for i, text in enumerate(train_texts):
        spans = poetry_line_splitter(text)
        if not spans:
            continue
        with torch.no_grad():
            e = encoder(spans)
            idea_embeds.append(e.cpu())
        if i % 200 == 0:
            print(f"  train ideas {i}/{len(train_texts)}", flush=True)
    all_ideas = torch.cat(idea_embeds, dim=0)
    inorms = torch.linalg.vector_norm(all_ideas, dim=1, keepdim=True).clamp_min(1e-8)
    all_ideas = all_ideas / inorms
    k = min(args.vocab_size, all_ideas.size(0))
    cb_centroids = _kmeans(all_ideas, k=k, seed=args.seed)
    print(f"codebook K={k}, idea embeddings={all_ideas.size(0)}", flush=True)

    codebook_path = ARTIFACTS / "idea_codebook.pt"
    torch.save({"centroids": cb_centroids}, codebook_path)
    torch.save(boundary.state_dict(), ARTIFACTS / "idea_boundary.pt")

    from creativegainbench.ideas.idea_ngram import IdeaCodebook

    codebook = IdeaCodebook(centroids=cb_centroids)

    # Build per-domain deformation contexts (train H, probe P).
    domains_meta = {}
    for d, drows in sorted(by_domain.items()):
        train_t = [
            body[:4000]
            for pid, author, body, title in drows
            if id_to_split[str(pid)] == "train"
        ]
        probe_t = [
            body[:4000]
            for pid, author, body, title in drows
            if id_to_split[str(pid)] == "probe"
        ]
        eval_n = sum(1 for pid, *_ in drows if id_to_split[str(pid)] == "eval")
        if len(probe_t) < 5 or len(train_t) < 10:
            print(
                f"  skip domain {d}: train={len(train_t)} probe={len(probe_t)} eval={eval_n}",
                flush=True,
            )
            continue
        print(
            f"  domain {d}: train={len(train_t)} probe={len(probe_t)} eval={eval_n}",
            flush=True,
        )
        ctx = build_domain_context(
            domain=d,
            train_texts=train_t,
            probe_texts=probe_t,
            span_encoder=encoder,
            codebook=codebook,
            boundary_detector=boundary,
            sentence_splitter=poetry_line_splitter,
            boundary_threshold=0.5,
            order=3,
        )
        with open(ARTIFACTS / f"domain_{d}_ctx.pkl", "wb") as f:
            pickle.dump(ctx, f)
        domains_meta[str(d)] = {
            "n_train": len(train_t),
            "n_probe": ctx.n_probes,
            "n_eval": eval_n,
            "mean_probe_len": ctx.mean_probe_len,
            "mean_probe_symbols": ctx.mean_probe_len,
        }

    # Calibrate δ_D per domain: corpus copies / paraphrases (neg) vs eval poems (pos).
    from creativegainbench.metrics.deformation import compute_deformation

    delta_d = {}
    for d_str in domains_meta:
        d = int(d_str)
        with open(ARTIFACTS / f"domain_{d}_ctx.pkl", "rb") as f:
            ctx = pickle.load(f)
        drows = by_domain[d]
        train_bodies = [
            body[:4000]
            for pid, author, body, title in drows
            if id_to_split[str(pid)] == "train"
        ]
        probe_bodies = [
            body[:4000]
            for pid, author, body, title in drows
            if id_to_split[str(pid)] == "probe"
        ]
        eval_bodies = [
            body[:4000]
            for pid, author, body, title in drows
            if id_to_split[str(pid)] == "eval"
        ][:8]

        negatives: list[float] = []
        positives: list[float] = []
        for body in train_bodies[:8]:
            r = compute_deformation(
                body,
                ctx,
                span_encoder=encoder,
                codebook=codebook,
                boundary_detector=boundary,
                sentence_splitter=poetry_line_splitter,
            )
            negatives.append(r.r_d_norm)
        for body in probe_bodies[:5]:
            para = "Here is the same poem restated: " + body
            r = compute_deformation(
                para,
                ctx,
                span_encoder=encoder,
                codebook=codebook,
                boundary_detector=boundary,
                sentence_splitter=poetry_line_splitter,
            )
            negatives.append(r.r_d_norm)
        for body in eval_bodies:
            r = compute_deformation(
                body,
                ctx,
                span_encoder=encoder,
                codebook=codebook,
                boundary_detector=boundary,
                sentence_splitter=poetry_line_splitter,
            )
            positives.append(r.r_d_norm)

        if positives and negatives:
            neg_hi = max(negatives)
            pos_lo = min(positives)
            if pos_lo > neg_hi:
                thr = 0.5 * (pos_lo + neg_hi)
            else:
                thr = sorted(negatives)[int(0.9 * (len(negatives) - 1))] + 1e-6
        else:
            thr = 0.0
        delta_d[d_str] = {
            "delta_d_norm": thr,
            "positives": positives,
            "negatives": negatives,
            "separable": bool(
                positives and negatives and min(positives) > max(negatives)
            ),
        }
        print(
            f"  δ_D[{d}]={thr:.6f} separable={delta_d[d_str]['separable']} "
            f"pos_mean={sum(positives)/max(len(positives),1):.4f} "
            f"neg_mean={sum(negatives)/max(len(negatives),1):.4f}",
            flush=True,
        )

    meta = {
        "version": "poetry_v2",
        "k": args.k,
        "seed": args.seed,
        "vocab_size": k,
        "embedding_dim": int(encoder.embedding_dim),
        "span_encoder": "minilm",
        "sentence_splitter": "poetry_line_splitter",
        "domains": domains_meta,
        "delta_d": delta_d,
        "n_poems": len(rows),
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"DONE → {ARTIFACTS}", flush=True)


if __name__ == "__main__":
    main()
