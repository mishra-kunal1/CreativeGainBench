"""
Deterministic ReceiverAgent for offline / CI evaluation.

Samples are synthetic variations of the conditioned input; embeddings come from
the shared frozen span encoder so they live in the idea-codebook space.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import mean_pool_idea_embeddings


class HashReceiverAgent:
    """Offline receiver that does not call external APIs."""

    def __init__(
        self,
        span_encoder: nn.Module,
        boundary_detector: IdeaBoundaryDetector | None = None,
        boundary_threshold: float = 0.5,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        seed: int = 42,
    ):
        self.span_encoder = span_encoder
        self.boundary_detector = boundary_detector
        self.boundary_threshold = boundary_threshold
        self.sentence_splitter = sentence_splitter
        self.seed = seed

    def condition(self, task_input: str, *, context: str) -> str:
        # Compact conditioning string used as the generation seed.
        ctx = (context or "").strip()
        if len(ctx) > 400:
            ctx = ctx[:400]
        return f"Context:\n{ctx}\n\nTask:\n{task_input.strip()}"

    def sample_with_embeddings(
        self, conditioned_input: str, n: int
    ) -> tuple[list[str], torch.Tensor]:
        samples: list[str] = []
        embeds: list[torch.Tensor] = []
        for i in range(n):
            # Deterministic lexical variants — enough diversity for soft clustering.
            sample = f"{conditioned_input}\nVariant-{i}: focus angle {i}."
            samples.append(sample)
            emb = mean_pool_idea_embeddings(
                sample,
                span_encoder=self.span_encoder,
                boundary_detector=self.boundary_detector,
                sentence_splitter=self.sentence_splitter,
                boundary_threshold=self.boundary_threshold,
            )
            embeds.append(emb)
        if not embeds:
            dim = getattr(self.span_encoder, "embedding_dim", 64)
            return [], torch.zeros(0, dim)
        return samples, torch.stack(embeds, dim=0)
