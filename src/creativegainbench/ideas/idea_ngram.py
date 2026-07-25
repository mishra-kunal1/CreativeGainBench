"""
Builds n-grams over the SEQUENCE OF IDEAS (not tokens/words).

Each idea is mapped to a discrete cluster ID via a frozen codebook
(vector-quantization over idea embeddings), so idea n-grams become sequences
of discrete idea-cluster symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn as nn

from creativegainbench.ideas.idea_extractor import (
    Idea,
    IdeaBoundaryDetector,
    extract_ideas,
)


@dataclass
class IdeaCodebook:
    """Frozen VQ codebook mapping idea embeddings -> discrete idea-cluster IDs."""

    centroids: torch.Tensor  # (K, d), FROZEN across benchmark

    @property
    def vocab_size(self) -> int:
        return int(self.centroids.size(0))

    @property
    def embedding_dim(self) -> int:
        return int(self.centroids.size(1))


def quantize_idea(idea: Idea, codebook: IdeaCodebook) -> int:
    dists = torch.cdist(idea.embedding.unsqueeze(0), codebook.centroids)  # (1, K)
    return int(torch.argmin(dists, dim=-1).item())


def ideas_to_symbol_sequence(ideas: List[Idea], codebook: IdeaCodebook) -> List[int]:
    return [quantize_idea(idea, codebook) for idea in ideas]


def idea_ngrams(symbol_seq: List[int], n: int) -> List[Tuple[int, ...]]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(symbol_seq) < n:
        return []
    return [tuple(symbol_seq[i : i + n]) for i in range(len(symbol_seq) - n + 1)]


def build_idea_ngram_sequence(
    text: str,
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    n: int = 3,
    boundary_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, ...]], List[int]]:
    ideas = extract_ideas(
        text,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    symbols = ideas_to_symbol_sequence(ideas, codebook)
    return idea_ngrams(symbols, n), symbols


def text_to_idea_symbols(
    text: str,
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> List[int]:
    """Canonical interface: text -> discrete idea-cluster symbol sequence."""
    ideas = extract_ideas(
        text,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    return ideas_to_symbol_sequence(ideas, codebook)


def mean_pool_idea_embeddings(
    text: str,
    span_encoder: nn.Module,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> torch.Tensor:
    """Mean idea embedding for a text (zeros if empty)."""
    ideas = extract_ideas(
        text,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    if not ideas:
        # Infer dim from encoder if possible.
        dim = getattr(span_encoder, "embedding_dim", None)
        if dim is None:
            probe = span_encoder(["_"])
            dim = int(probe.size(-1))
        return torch.zeros(dim, dtype=torch.float32)
    return torch.stack([idea.embedding for idea in ideas], dim=0).mean(dim=0)
