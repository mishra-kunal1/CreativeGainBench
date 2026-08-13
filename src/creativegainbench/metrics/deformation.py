"""
True probe-relative deformation R_D matching bridged_validation ProbeCompressor:

  D(y, H, P) = Σ_{s'∈P} ( N(s'|H) - N(s'|H ∪ {y}) )

Uses CountNgramLM over idea-cluster symbols with a non-empty background
corpus H. Returns raw bits and λ_D-normalized score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch.nn as nn

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import IdeaCodebook, text_to_idea_symbols
from creativegainbench.metrics.count_ngram import (
    CountNgramLM,
    deformation_gain,
    lambda_d_normalize,
)


@dataclass
class DomainDeformationContext:
    """Frozen per-domain compressor state for R_D evaluation."""

    domain: int
    base_lm: CountNgramLM
    probe_texts: List[str]
    probe_symbol_seqs: List[List[int]]
    vocab_size: int
    order: int

    @property
    def n_probes(self) -> int:
        return len(self.probe_symbol_seqs)

    @property
    def mean_probe_len(self) -> float:
        if not self.probe_symbol_seqs:
            return 1.0
        return sum(len(s) for s in self.probe_symbol_seqs) / len(self.probe_symbol_seqs)


@dataclass
class DeformationResult:
    r_d_raw: float
    r_d_norm: float
    n_probes: int
    mean_probe_len: float
    y_n_symbols: int


def encode_text(
    text: str,
    *,
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> List[int]:
    return text_to_idea_symbols(
        text,
        span_encoder,
        codebook,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )


def build_domain_context(
    *,
    domain: int,
    train_texts: Sequence[str],
    probe_texts: Sequence[str],
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
    order: int = 3,
) -> DomainDeformationContext:
    from creativegainbench.metrics.count_ngram import train_count_lm

    train_seqs = [
        encode_text(
            t,
            span_encoder=span_encoder,
            codebook=codebook,
            boundary_detector=boundary_detector,
            sentence_splitter=sentence_splitter,
            boundary_threshold=boundary_threshold,
        )
        for t in train_texts
    ]
    train_seqs = [s for s in train_seqs if s]
    probe_seqs = [
        encode_text(
            t,
            span_encoder=span_encoder,
            codebook=codebook,
            boundary_detector=boundary_detector,
            sentence_splitter=sentence_splitter,
            boundary_threshold=boundary_threshold,
        )
        for t in probe_texts
    ]
    # Keep probes even if short; drop empty.
    kept_probes = [(t, s) for t, s in zip(probe_texts, probe_seqs) if s]
    if not kept_probes:
        raise ValueError(f"Domain {domain}: no non-empty probe symbol sequences")
    base_lm = train_count_lm(train_seqs, order=order)
    return DomainDeformationContext(
        domain=domain,
        base_lm=base_lm,
        probe_texts=[t for t, _ in kept_probes],
        probe_symbol_seqs=[s for _, s in kept_probes],
        vocab_size=codebook.vocab_size,
        order=order,
    )


def compute_deformation(
    y: str,
    ctx: DomainDeformationContext,
    *,
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> DeformationResult:
    y_syms = encode_text(
        y,
        span_encoder=span_encoder,
        codebook=codebook,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    raw = deformation_gain(y_syms, ctx.probe_symbol_seqs, ctx.base_lm)
    norm = lambda_d_normalize(
        raw,
        n_probes=ctx.n_probes,
        vocab_size=ctx.vocab_size,
        mean_probe_len=ctx.mean_probe_len,
    )
    return DeformationResult(
        r_d_raw=raw,
        r_d_norm=norm,
        n_probes=ctx.n_probes,
        mean_probe_len=ctx.mean_probe_len,
        y_n_symbols=len(y_syms),
    )
