"""
Soft ProbeCompressor — order-1 soft-count LM over codebook assignments.

Same Lean deformation functional as CountNgram:

  D(y, H, P) = Σ_{s'∈P} ( N(s'|H) - N(s'|H ∪ {y}) )

but symbols are soft assignments

  π_k(e) = softmax_k( -||e - c_k||² / τ )

and N is soft-unigram cross-entropy bits of the probe under fractional counts.
Hard VQ + order-3 CountNgram discarded class-discriminative MiniLM directions;
soft order-1 keeps that geometry inside the compressor alphabet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F

from creativegainbench.metrics.count_ngram import lambda_d_normalize


def soft_assign(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Soft codebook assignment.

    embeddings: (n, d), centroids: (K, d) → (n, K) rows sum to 1.
    Uses squared Euclidean distance (matches plan); falls back to empty (0, K).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {tuple(embeddings.shape)}")
    if embeddings.size(0) == 0:
        return torch.zeros(0, centroids.size(0), dtype=torch.float32)
    # (n, K) squared distances
    dists_sq = torch.cdist(embeddings, centroids).pow(2)
    return F.softmax(-dists_sq / max(float(tau), 1e-8), dim=-1)


def soft_assign_hard_limit(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """τ → 0 limit: one-hot at nearest centroid (hard VQ)."""
    if embeddings.size(0) == 0:
        return torch.zeros(0, centroids.size(0), dtype=torch.float32)
    idx = torch.argmin(torch.cdist(embeddings, centroids), dim=1)
    out = torch.zeros(embeddings.size(0), centroids.size(0), dtype=torch.float32)
    out.scatter_(1, idx.unsqueeze(1), 1.0)
    return out


@dataclass
class SoftCountLM:
    """
    Mutable soft unigram LM over a K-way codebook.

    Counts are fractional: each idea position contributes its soft row π.
    """

    vocab_size: int
    counts: torch.Tensor | None = None  # (K,)
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = torch.zeros(self.vocab_size, dtype=torch.float64)
        else:
            self.counts = self.counts.to(dtype=torch.float64).reshape(self.vocab_size)

    def clone(self) -> "SoftCountLM":
        return SoftCountLM(
            vocab_size=self.vocab_size,
            counts=self.counts.clone(),
            eps=self.eps,
        )

    def add_sequence(self, soft_rows: torch.Tensor | Sequence[Sequence[float]]) -> None:
        """Accumulate soft assignment rows (n, K) into unigram counts."""
        rows = torch.as_tensor(soft_rows, dtype=torch.float64)
        if rows.numel() == 0:
            return
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.size(-1) != self.vocab_size:
            raise ValueError(
                f"soft_rows last dim {rows.size(-1)} != vocab_size {self.vocab_size}"
            )
        self.counts = self.counts + rows.sum(dim=0)

    def add_sequences(self, sequences: Sequence[torch.Tensor]) -> None:
        for seq in sequences:
            self.add_sequence(seq)

    def probs(self) -> torch.Tensor:
        """Smoothed unigram p(k) over codebook."""
        total = float(self.counts.sum().item()) + self.eps * self.vocab_size
        return (self.counts + self.eps) / total

    def sequence_bits(self, soft_rows: torch.Tensor | Sequence[Sequence[float]]) -> float:
        """
        Soft-unigram CE bits of a probe:

          N = -Σ_t log2( Σ_k π_{t,k} p_H(k) )
        """
        rows = torch.as_tensor(soft_rows, dtype=torch.float64)
        if rows.numel() == 0:
            return 0.0
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        p = self.probs()  # (K,)
        # mixture mass per position
        mix = (rows * p.unsqueeze(0)).sum(dim=-1).clamp_min(1e-99)
        return float((-torch.log2(mix)).sum().item())


def train_soft_count_lm(
    soft_seqs: List[torch.Tensor],
    vocab_size: int,
    *,
    eps: float = 1e-6,
) -> SoftCountLM:
    lm = SoftCountLM(vocab_size=vocab_size, eps=eps)
    lm.add_sequences(soft_seqs)
    return lm


def soft_deformation_gain(
    y_soft: torch.Tensor,
    probe_softs: List[torch.Tensor],
    base_lm: SoftCountLM,
) -> float:
    """
    D(y, H, P) = Σ (bits_H(s') - bits_{H∪{y}}(s')).

    Positive when incorporating y makes probes cheaper under the soft LM.
    """
    baseline = sum(base_lm.sequence_bits(p) for p in probe_softs)
    updated = base_lm.clone()
    updated.add_sequence(y_soft)
    conditional = sum(updated.sequence_bits(p) for p in probe_softs)
    return float(baseline - conditional)


__all__ = [
    "SoftCountLM",
    "lambda_d_normalize",
    "soft_assign",
    "soft_assign_hard_limit",
    "soft_deformation_gain",
    "train_soft_count_lm",
]
