"""
True probe-relative deformation R_D matching bridged_validation ProbeCompressor:

  D(y, H, P) = Σ_{s'∈P} ( N(s'|H) - N(s'|H ∪ {y}) )

Backends:
  * hard CountNgramLM over idea-cluster symbols (default)
  * soft SoftCountLM over soft codebook assignments (geometry-aware)
  * kernel Parzen CE over raw MiniLM embeddings (geometry-native)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
import torch.nn as nn

from creativegainbench.ideas.idea_extractor import (
    IdeaBoundaryDetector,
    extract_ideas,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook, text_to_idea_symbols
from creativegainbench.metrics.count_ngram import (
    CountNgramLM,
    deformation_gain,
    lambda_d_normalize,
)
from creativegainbench.metrics.kernel_probe_ce import (
    KernelProbeLM,
    kernel_deformation_gain,
    train_kernel_probe_lm,
)
from creativegainbench.metrics.soft_count_ngram import (
    SoftCountLM,
    soft_assign,
    soft_deformation_gain,
    train_soft_count_lm,
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


def encode_text_embeddings(
    text: str,
    *,
    span_encoder: nn.Module,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> torch.Tensor:
    """Idea embeddings for a text → (n_ideas, d), empty (0, d) if none."""
    ideas = extract_ideas(
        text,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    if not ideas:
        dim = getattr(span_encoder, "embedding_dim", None)
        if dim is None:
            probe = span_encoder(["_"])
            dim = int(probe.size(-1))
        return torch.zeros(0, int(dim), dtype=torch.float32)
    return torch.stack([idea.embedding.detach().cpu().float() for idea in ideas], dim=0)


def encode_text_soft(
    text: str,
    *,
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    tau: float = 1.0,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> torch.Tensor:
    """Soft assignment rows (n_ideas, K) for a text."""
    emb = encode_text_embeddings(
        text,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    return soft_assign(emb, codebook.centroids.detach().cpu().float(), tau=tau)


@dataclass
class SoftDomainDeformationContext:
    """Frozen per-domain soft compressor state for R_D^soft evaluation."""

    domain: int
    base_lm: SoftCountLM
    probe_texts: List[str]
    probe_soft_seqs: List[torch.Tensor]
    vocab_size: int
    tau: float
    order: int = 1

    @property
    def n_probes(self) -> int:
        return len(self.probe_soft_seqs)

    @property
    def mean_probe_len(self) -> float:
        if not self.probe_soft_seqs:
            return 1.0
        return sum(int(s.size(0)) for s in self.probe_soft_seqs) / len(
            self.probe_soft_seqs
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


def build_soft_domain_context(
    *,
    domain: int,
    train_texts: Sequence[str],
    probe_texts: Sequence[str],
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    tau: float = 1.0,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> SoftDomainDeformationContext:
    train_soft = [
        encode_text_soft(
            t,
            span_encoder=span_encoder,
            codebook=codebook,
            tau=tau,
            boundary_detector=boundary_detector,
            sentence_splitter=sentence_splitter,
            boundary_threshold=boundary_threshold,
        )
        for t in train_texts
    ]
    train_soft = [s for s in train_soft if s.size(0) > 0]
    probe_pairs = []
    for t in probe_texts:
        s = encode_text_soft(
            t,
            span_encoder=span_encoder,
            codebook=codebook,
            tau=tau,
            boundary_detector=boundary_detector,
            sentence_splitter=sentence_splitter,
            boundary_threshold=boundary_threshold,
        )
        if s.size(0) > 0:
            probe_pairs.append((t, s))
    if not probe_pairs:
        raise ValueError(f"Domain {domain}: no non-empty soft probe sequences")
    if not train_soft:
        raise ValueError(f"Domain {domain}: no non-empty soft train sequences")
    base_lm = train_soft_count_lm(train_soft, codebook.vocab_size)
    return SoftDomainDeformationContext(
        domain=domain,
        base_lm=base_lm,
        probe_texts=[t for t, _ in probe_pairs],
        probe_soft_seqs=[s for _, s in probe_pairs],
        vocab_size=codebook.vocab_size,
        tau=float(tau),
        order=1,
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


def compute_soft_deformation(
    y: str,
    ctx: SoftDomainDeformationContext,
    *,
    span_encoder: nn.Module,
    codebook: IdeaCodebook,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
    tau: float | None = None,
) -> DeformationResult:
    t = float(ctx.tau if tau is None else tau)
    y_soft = encode_text_soft(
        y,
        span_encoder=span_encoder,
        codebook=codebook,
        tau=t,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    raw = soft_deformation_gain(y_soft, ctx.probe_soft_seqs, ctx.base_lm)
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
        y_n_symbols=int(y_soft.size(0)),
    )


@dataclass
class KernelDomainDeformationContext:
    """Frozen per-domain Parzen bank + probe embeddings for R_D^ker."""

    domain: int
    base_lm: KernelProbeLM
    probe_texts: List[str]
    probe_emb_seqs: List[torch.Tensor]
    embedding_dim: int
    sigma: float
    max_bank: int | None = 4096

    @property
    def n_probes(self) -> int:
        return len(self.probe_emb_seqs)

    @property
    def mean_probe_len(self) -> float:
        if not self.probe_emb_seqs:
            return 1.0
        return sum(int(s.size(0)) for s in self.probe_emb_seqs) / len(
            self.probe_emb_seqs
        )

    @property
    def vocab_size(self) -> int:
        # Continuous alphabet proxy for λ_D (embedding dim).
        return max(int(self.embedding_dim), 2)


def build_kernel_domain_context(
    *,
    domain: int,
    train_embs: Sequence[torch.Tensor],
    probe_embs: Sequence[torch.Tensor],
    probe_texts: Sequence[str],
    sigma: float,
    max_bank: int | None = 4096,
    seed: int = 42,
) -> KernelDomainDeformationContext:
    train_kept = [e for e in train_embs if e is not None and e.numel() > 0]
    probe_pairs = [
        (t, e)
        for t, e in zip(probe_texts, probe_embs)
        if e is not None and e.numel() > 0
    ]
    if not train_kept:
        raise ValueError(f"Domain {domain}: no non-empty train embeddings")
    if not probe_pairs:
        raise ValueError(f"Domain {domain}: no non-empty probe embeddings")
    dim = int(train_kept[0].size(-1))
    base_lm = train_kernel_probe_lm(
        list(train_kept), sigma=sigma, max_bank=max_bank, seed=seed
    )
    return KernelDomainDeformationContext(
        domain=domain,
        base_lm=base_lm,
        probe_texts=[t for t, _ in probe_pairs],
        probe_emb_seqs=[e for _, e in probe_pairs],
        embedding_dim=dim,
        sigma=float(sigma),
        max_bank=max_bank,
    )


def load_kernel_domain_context(path) -> KernelDomainDeformationContext:
    """Load a frozen per-domain Parzen context saved by the build script."""
    from pathlib import Path

    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    lm = KernelProbeLM(
        bank=blob["bank"],
        sigma=float(blob["sigma"]),
        eps=float(blob.get("eps", 1e-12)),
    )
    return KernelDomainDeformationContext(
        domain=int(blob["domain"]),
        base_lm=lm,
        probe_texts=list(blob["probe_texts"]),
        probe_emb_seqs=list(blob["probe_emb_seqs"]),
        embedding_dim=int(blob["embedding_dim"]),
        sigma=float(blob["sigma"]),
        max_bank=blob.get("max_bank"),
    )


def compute_kernel_deformation(
    y: str,
    ctx: KernelDomainDeformationContext,
    *,
    span_encoder: nn.Module,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    boundary_threshold: float = 0.5,
) -> DeformationResult:
    y_emb = encode_text_embeddings(
        y,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        boundary_threshold=boundary_threshold,
    )
    raw = kernel_deformation_gain(y_emb, ctx.probe_emb_seqs, ctx.base_lm)
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
        y_n_symbols=int(y_emb.size(0)),
    )
