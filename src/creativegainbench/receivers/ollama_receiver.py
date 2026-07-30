"""
Ollama-backed ReceiverAgent for live R_B^{→A} estimation.
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
from openai import OpenAI

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import mean_pool_idea_embeddings

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"


class OllamaReceiverAgent:
    def __init__(
        self,
        span_encoder: nn.Module,
        boundary_detector: IdeaBoundaryDetector | None = None,
        boundary_threshold: float = 0.5,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        model: str = "gemma2:2b",
        base_url: str | None = None,
        temperature: float = 1.0,
    ):
        self.client = OpenAI(
            base_url=base_url or DEFAULT_OLLAMA_BASE,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
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
            "the task in 1-3 short sentences.\n\n"
            f"Context:\n{ctx}\n\nTask:\n{task_input.strip()}"
        )

    def sample_with_embeddings(
        self, conditioned_input: str, n: int
    ) -> tuple[list[str], torch.Tensor]:
        samples: list[str] = []
        for _ in range(n):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": conditioned_input}],
                temperature=self.temperature,
                n=1,
            )
            samples.append(completion.choices[0].message.content or "")
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
            dim = getattr(self.span_encoder, "embedding_dim", 384)
            return samples, torch.zeros(0, dim)
        return samples, torch.stack(embeds, dim=0)
