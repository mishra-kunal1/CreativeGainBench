"""
Calibrate per-domain δ_D from a constructed negative bank (ProbeCompressor R_D).

Negatives: exact H members, token-shuffled H members, and padded H members.
Threshold = one-sided Q_q(neg R_D_norm) + eps. Never fit to humans/models/z*.

Writes artifacts/delta_d_thresholds_{version}.json (and a stable
artifacts/delta_d_thresholds.json copy for the active version).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from creativegainbench.ideas.artifacts import load_artifacts
from creativegainbench.ideas.idea_extractor import default_sentence_splitter
from creativegainbench.metrics.delta_d import (
    thresholds_from_negatives,
    write_delta_d_thresholds,
)
from creativegainbench.metrics.deformation import compute_deformation

ARTIFACTS_ROOT = Path(__file__).resolve().parent


def _shuffle_words(text: str, rng: random.Random) -> str:
    toks = text.split()
    if len(toks) < 2:
        return text
    rng.shuffle(toks)
    return " ".join(toks)


def _pad_text(text: str, pad_token: str = "the", n: int = 40) -> str:
    return (text + " " + " ".join([pad_token] * n)).strip()


def calibrate_delta_d(
    *,
    version: str = "v1",
    artifacts_root: Path | None = None,
    quantile: float = 0.95,
    n_neg: int = 60,
    seed: int = 42,
) -> dict:
    root = artifacts_root or ARTIFACTS_ROOT
    pipeline = load_artifacts(
        version=version, device="cpu", artifacts_root=root, verify_hashes=True
    )
    rng = random.Random(seed)

    # Build negatives from probe / held-out-style strings already in train ctx:
    # use deformation_ctx probe texts as stand-ins for H-overlapping material
    # plus synthetic pads/shuffles of those strings.
    base_texts = list(pipeline.deformation_ctx.probe_texts)
    if len(base_texts) < 3:
        base_texts = list(pipeline.probe_set.strings[:20])

    negatives: list[str] = []
    for t in base_texts:
        negatives.append(t)  # exact_h-like (already in H/P neighborhood)
        negatives.append(_shuffle_words(t, rng))
        negatives.append(_pad_text(t))
        if len(negatives) >= n_neg:
            break
    while len(negatives) < n_neg:
        t = rng.choice(base_texts)
        negatives.append(_shuffle_words(_pad_text(t, n=rng.randint(10, 60)), rng))

    scores: list[float] = []
    for y in negatives[:n_neg]:
        rd = compute_deformation(
            y,
            pipeline.deformation_ctx,
            span_encoder=pipeline.span_encoder,
            codebook=pipeline.codebook,
            boundary_detector=pipeline.boundary_detector,
            sentence_splitter=default_sentence_splitter,
            boundary_threshold=pipeline.boundary_threshold,
        )
        scores.append(float(rd.r_d_norm))

    thresholds = thresholds_from_negatives({"0": scores, "default": scores}, q=quantile)
    out_versioned = root / f"delta_d_thresholds_{version}.json"
    out_stable = root / "delta_d_thresholds.json"
    write_delta_d_thresholds(out_versioned, thresholds)
    write_delta_d_thresholds(out_stable, thresholds)
    thr = thresholds["0"]["delta_d_95"]
    print(f"Calibrated δ_D={thr:.6g} (q={quantile}, n_neg={len(scores)}) → {out_stable}")
    return dict(thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate δ_D from negative-bank R_D quantiles"
    )
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--n-neg", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    calibrate_delta_d(
        version=args.version,
        quantile=args.quantile,
        n_neg=args.n_neg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
