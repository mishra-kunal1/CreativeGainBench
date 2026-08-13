"""Load frozen poetry_v2 idea pipeline + per-domain deformation contexts."""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from lib import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "src"))

from creativegainbench.ideas.idea_extractor import (  # noqa: E402
    IdeaBoundaryDetector,
    poetry_line_splitter,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook, text_to_idea_symbols  # noqa: E402
from creativegainbench.ideas.span_encoder import build_span_encoder  # noqa: E402
from creativegainbench.metrics.deformation import (  # noqa: E402
    DomainDeformationContext,
    compute_deformation,
)


@dataclass
class PoetryV2Stack:
    codebook: IdeaCodebook
    encoder: torch.nn.Module
    boundary: IdeaBoundaryDetector
    domain_ctx: dict[int, DomainDeformationContext]
    meta: dict
    max_chars: int
    device: str

    def to_idea_symbols(self, text: str) -> list[int]:
        return text_to_idea_symbols(
            (text or "")[: self.max_chars],
            self.encoder,
            self.codebook,
            boundary_detector=self.boundary,
            sentence_splitter=poetry_line_splitter,
        )

    def score_r_d(self, text: str, domain: int):
        ctx = self.domain_ctx[domain]
        return compute_deformation(
            (text or "")[: self.max_chars],
            ctx,
            span_encoder=self.encoder,
            codebook=self.codebook,
            boundary_detector=self.boundary,
            sentence_splitter=poetry_line_splitter,
        )


def load_stack(device: str | None = None) -> PoetryV2Stack:
    cfg = load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts = Path(cfg["artifacts"])
    meta = json.loads((artifacts / "meta.json").read_text())
    codebook_state = torch.load(
        artifacts / "idea_codebook.pt", map_location="cpu", weights_only=True
    )
    codebook = IdeaCodebook(centroids=codebook_state["centroids"])
    encoder = build_span_encoder("minilm", device=device)
    boundary = IdeaBoundaryDetector(hidden_dim=codebook.embedding_dim)
    boundary.load_state_dict(
        torch.load(artifacts / "idea_boundary.pt", map_location="cpu", weights_only=True)
    )
    boundary.eval()
    if device != "cpu":
        boundary = boundary.to(device)

    domain_ctx: dict[int, DomainDeformationContext] = {}
    for d_str in meta.get("domains", {}):
        path = artifacts / f"domain_{int(d_str)}_ctx.pkl"
        if path.exists():
            with open(path, "rb") as f:
                domain_ctx[int(d_str)] = pickle.load(f)

    return PoetryV2Stack(
        codebook=codebook,
        encoder=encoder,
        boundary=boundary,
        domain_ctx=domain_ctx,
        meta=meta,
        max_chars=int(cfg["max_chars"]),
        device=device,
    )


# Re-export package U-proxy so calibration / E* share one definition (F0.1).
from creativegainbench.metrics.feasibility import feasibility_bit  # noqa: E402,F401
