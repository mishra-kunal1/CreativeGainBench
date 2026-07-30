"""
OpenAI-backed ReceiverAgent for live R_B^{→A} estimation.

Optional: only used when OPENAI_API_KEY is set and --receiver openai is chosen.
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
from dotenv import load_dotenv

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import mean_pool_idea_embeddings


class OpenAIReceiverAgent:
    def __init__(
        self,
        span_encoder: nn.Module,
        boundary_detector: IdeaBoundaryDetector | None = None,
        boundary_threshold: float = 0.5,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
    ):
        load_dotenv()
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for OpenAIReceiverAgent")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.span_encoder = span_encoder
        self.boundary_detector = boundary_detector
        self.boundary_threshold = boundary_threshold
        self.sentence_splitter = sentence_splitter

    def condition(self, task_input: str, *, context: str) -> str:
        ctx = (context or "").strip()
        if len(ctx) > 1500:
            ctx = ctx[:1500]
        return (
            "You are a receiver agent. Use the creative context below to answer "
            "the task briefly.\n\n"
            f"Context:\n{ctx}\n\nTask:\n{task_input.strip()}"
        )

    def sample_with_embeddings(
        self, conditioned_input: str, n: int
    ) -> tuple[list[str], torch.Tensor]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": conditioned_input}],
            temperature=self.temperature,
            n=n,
        )
        samples = [c.message.content or "" for c in completion.choices]
        embeds = [
            mean_pool_idea_embeddings(
                s,
                span_encoder=self.span_encoder,
                boundary_detector=self.boundary_detector,
                sentence_splitter=self.sentence_splitter,
                boundary_threshold=self.boundary_threshold,
            )
            for s in samples
        ]
        if not embeds:
            dim = getattr(self.span_encoder, "embedding_dim", 64)
            return samples, torch.zeros(0, dim)
        return samples, torch.stack(embeds, dim=0)
