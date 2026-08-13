"""
Extracts atomic idea units from a text output.

An "idea" is a minimal proposition-level segment — implemented via a torch-based
idea-boundary classifier over span candidates, then encoded into a fixed-dim
embedding. When no trained boundary detector is available, sentence/clause
splits are used as a deterministic fallback.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
import torch.nn as nn


@dataclass
class Idea:
    text: str
    embedding: torch.Tensor  # shape (d,)


class HashSpanEncoder(nn.Module):
    """
    Deterministic bag-of-hashes encoder used for frozen stub artifacts / CI.
    Produces unit-norm embeddings of fixed dimension without external weights.
    """

    def __init__(self, embedding_dim: int = 64, n_hashes: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_hashes = n_hashes

    def encode_one(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.embedding_dim, dtype=torch.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            tokens = ["_empty_"]
        for token in tokens:
            for i in range(self.n_hashes):
                digest = hashlib.sha256(f"{i}:{token}".encode()).digest()
                idx = int.from_bytes(digest[:4], "little") % self.embedding_dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
        norm = torch.linalg.vector_norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def forward(self, spans: Sequence[str]) -> torch.Tensor:
        if not spans:
            return torch.zeros(0, self.embedding_dim, dtype=torch.float32)
        return torch.stack([self.encode_one(s) for s in spans], dim=0)


class IdeaBoundaryDetector(nn.Module):
    """
    Lightweight classifier over span embeddings predicting idea-boundary
    probability. Trained separately or distilled from an LLM teacher.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.boundary_head = nn.Linear(hidden_dim, 1)

    def forward(self, span_embeddings: torch.Tensor) -> torch.Tensor:
        # span_embeddings: (n_spans, hidden_dim)
        logits = self.boundary_head(span_embeddings).squeeze(-1)
        return torch.sigmoid(logits)


def default_sentence_splitter(text: str) -> List[str]:
    """Coarse span candidates: sentences, then clause-ish splits on ';' and '—'."""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    spans: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        clauses = re.split(r"\s*[;—]\s*", part)
        for clause in clauses:
            clause = clause.strip()
            if clause:
                spans.append(clause)
    return spans or [text]


def poetry_line_splitter(text: str) -> List[str]:
    """
    Verse-aware span candidates: prefer line / stanza breaks over prose
    sentence punctuation so poems do not collapse into a handful of spans.
    """
    text = (text or "").strip()
    if not text:
        return []
    # Split stanzas on blank lines, then lines within stanzas.
    stanzas = re.split(r"\n\s*\n+", text)
    spans: List[str] = []
    for stanza in stanzas:
        for line in stanza.splitlines():
            line = line.strip()
            if not line:
                continue
            # Further split long prose-like lines on sentence ends.
            if len(line) > 120 and re.search(r"[.!?]", line):
                parts = re.split(r"(?<=[.!?])\s+", line)
                spans.extend(p.strip() for p in parts if p.strip())
            else:
                spans.append(line)
    return spans or default_sentence_splitter(text)


def verse_line_splitter(text: str, min_span_chars: int = 15) -> List[str]:
    """
    Verse-aware span candidates: poem lines, merging very short lines forward
    so one-word lines don't become standalone ideas. Falls back to the whole
    text if there are no line breaks.
    """
    text = (text or "").strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    spans: List[str] = []
    buf = ""
    for ln in lines:
        buf = f"{buf} {ln}".strip() if buf else ln
        if len(buf) >= min_span_chars:
            spans.append(buf)
            buf = ""
    if buf:
        if spans:
            spans[-1] = f"{spans[-1]} {buf}"
        else:
            spans.append(buf)
    return spans or [text]


def segment_into_idea_spans(
    text: str,
    sentence_splitter: Callable[[str], List[str]] | None = None,
) -> List[str]:
    splitter = sentence_splitter or default_sentence_splitter
    return splitter(text)


def extract_ideas(
    text: str,
    span_encoder: nn.Module,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> List[Idea]:
    spans = segment_into_idea_spans(text, sentence_splitter)
    if not spans:
        return []

    with torch.no_grad():
        span_embeds = span_encoder(spans)  # (n_spans, hidden_dim)
        if boundary_detector is None:
            # Fallback: each span is its own idea (deterministic, frozen-protocol safe).
            return [
                Idea(text=span, embedding=embed.detach().cpu())
                for span, embed in zip(spans, span_embeds)
            ]
        bd_device = next(boundary_detector.parameters()).device
        span_embeds = span_embeds.to(bd_device)
        boundary_probs = boundary_detector(span_embeds)

    ideas: List[Idea] = []
    buffer_spans: List[str] = []
    buffer_embeds: List[torch.Tensor] = []
    for span, embed, p in zip(spans, span_embeds, boundary_probs):
        buffer_spans.append(span)
        buffer_embeds.append(embed)
        if float(p.item()) > boundary_threshold:
            idea_text = " ".join(buffer_spans)
            idea_embed = torch.stack(buffer_embeds).mean(dim=0).detach().cpu()
            ideas.append(Idea(text=idea_text, embedding=idea_embed))
            buffer_spans, buffer_embeds = [], []

    if buffer_spans:
        ideas.append(
            Idea(
                text=" ".join(buffer_spans),
                embedding=torch.stack(buffer_embeds).mean(dim=0).detach().cpu(),
            )
        )
    return ideas
