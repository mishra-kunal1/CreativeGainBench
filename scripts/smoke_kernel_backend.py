#!/usr/bin/env python3
"""Smoke test: live kernel_parzen backend through compute_r_creativity."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from creativegainbench.ideas.artifacts import load_kernel_backend  # noqa: E402
from creativegainbench.ideas.idea_extractor import (  # noqa: E402
    IdeaBoundaryDetector,
    poetry_line_splitter,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.deformation import compute_kernel_deformation  # noqa: E402

ART = REPO / "src/creativegainbench/artifacts/poetry_v2"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = load_kernel_backend(verify_hashes=True)
    print(f"loaded kernel backend σ={backend.sigma} domains={sorted(backend.contexts)}")

    codebook = IdeaCodebook(
        centroids=torch.load(ART / "idea_codebook.pt", map_location="cpu", weights_only=True)[
            "centroids"
        ]
    )
    encoder = build_span_encoder("minilm", device=device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(ART / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()

    domain = sorted(backend.contexts)[0]
    ctx = backend.context_for(domain)
    delta = backend.delta_d_for(domain)

    samples = {
        "in_distribution_ish": "The quiet field lies still beneath the evening light.\n"
        "A slow wind moves the grass and then is gone.",
        "off_manifold": "SYNERGIZE the blockchain paradigm!!! 1010101 KPI KPI KPI "
        "quarterly deliverables optimize the funnel now now now.",
    }
    for name, text in samples.items():
        res = compute_kernel_deformation(
            text,
            ctx,
            span_encoder=encoder,
            boundary_detector=boundary,
            sentence_splitter=poetry_line_splitter,
        )
        gate = 1.0 if res.r_d_norm > delta else 0.0
        print(
            f"[{name}] domain={domain} r_d_norm={res.r_d_norm:.6g} "
            f"raw={res.r_d_raw:.4g} δ_D={delta:.6g} gate={gate} "
            f"n_ideas={res.y_n_symbols}"
        )

    # Hash-verification negative check: tamper -> should raise.
    print("live kernel backend OK")


if __name__ == "__main__":
    main()
