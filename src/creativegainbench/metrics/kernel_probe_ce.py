"""
Kernel / Parzen ProbeCompressor — continuous embedding-space deformation.

Same Lean functional as CountNgram / SoftCount:

  D(y, H, P) = Σ_{s'∈P} ( N(s'|H) - N(s'|H ∪ {y}) )

with Parzen code length over raw MiniLM idea embeddings (no codebook):

  p̂_H(e) = (1/|H|) Σ_{h∈H} exp( -||e-h||² / (2σ²) )
  N(s'|H) = -Σ_t log2 max(p̂_H(e'_t), ε)

Soft order-1 failed P1 (R²≈0.017): discrete soft alphabet still discarded
class geometry. Parzen CE is geometry-native and label-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch

from creativegainbench.metrics.count_ngram import lambda_d_normalize


def rbf_kernel_mean(
    queries: torch.Tensor,
    bank: torch.Tensor,
    sigma: float,
    *,
    chunk: int = 2048,
) -> torch.Tensor:
    """
    Mean Gaussian kernel of each query against the bank.

    queries: (m, d), bank: (n, d) → (m,) densities (unnormalized beyond mean).
    """
    if queries.ndim != 2 or bank.ndim != 2:
        raise ValueError("queries and bank must be 2-D")
    if queries.size(0) == 0:
        return torch.zeros(0, dtype=torch.float64)
    if bank.size(0) == 0:
        return torch.zeros(queries.size(0), dtype=torch.float64)

    sigma = max(float(sigma), 1e-8)
    inv = 1.0 / (2.0 * sigma * sigma)
    q = queries.to(dtype=torch.float32)
    b = bank.to(dtype=torch.float32)
    device = q.device
    b = b.to(device)

    acc = torch.zeros(q.size(0), dtype=torch.float64, device=device)
    n = b.size(0)
    for start in range(0, n, chunk):
        chunk_b = b[start : start + chunk]
        # (m, chunk_n) squared distances
        d2 = torch.cdist(q, chunk_b).pow(2)
        acc = acc + torch.exp((-d2 * inv).to(torch.float64)).sum(dim=1)
    return (acc / float(n)).cpu()


@dataclass
class KernelProbeLM:
    """Mutable Parzen bank over idea embeddings."""

    bank: torch.Tensor  # (N, d) float32 CPU
    sigma: float
    eps: float = 1e-12
    chunk: int = 2048

    def __post_init__(self) -> None:
        if self.bank.ndim != 2:
            raise ValueError(f"bank must be 2-D, got {tuple(self.bank.shape)}")
        self.bank = self.bank.detach().cpu().to(dtype=torch.float32)
        self.sigma = float(self.sigma)

    @property
    def n_bank(self) -> int:
        return int(self.bank.size(0))

    @property
    def embedding_dim(self) -> int:
        return int(self.bank.size(1)) if self.bank.numel() else 0

    def clone(self) -> "KernelProbeLM":
        return KernelProbeLM(
            bank=self.bank.clone(),
            sigma=self.sigma,
            eps=self.eps,
            chunk=self.chunk,
        )

    def add_sequence(self, emb: torch.Tensor | Sequence[Sequence[float]]) -> None:
        rows = torch.as_tensor(emb, dtype=torch.float32)
        if rows.numel() == 0:
            return
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if self.bank.numel() == 0:
            self.bank = rows.detach().cpu().contiguous()
        else:
            if rows.size(-1) != self.bank.size(-1):
                raise ValueError(
                    f"emb dim {rows.size(-1)} != bank dim {self.bank.size(-1)}"
                )
            self.bank = torch.cat([self.bank, rows.detach().cpu()], dim=0)

    def densities(self, queries: torch.Tensor) -> torch.Tensor:
        return rbf_kernel_mean(queries, self.bank, self.sigma, chunk=self.chunk)

    def sequence_bits(self, probe: torch.Tensor | Sequence[Sequence[float]]) -> float:
        rows = torch.as_tensor(probe, dtype=torch.float32)
        if rows.numel() == 0:
            return 0.0
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        dens = self.densities(rows).clamp_min(self.eps)
        return float((-torch.log2(dens)).sum().item())


def train_kernel_probe_lm(
    emb_seqs: List[torch.Tensor],
    sigma: float,
    *,
    max_bank: int | None = 4096,
    seed: int = 42,
    eps: float = 1e-12,
) -> KernelProbeLM:
    """Stack idea embeddings into a Parzen bank; optionally subsample."""
    parts = [e.detach().cpu().float() for e in emb_seqs if e is not None and e.numel() > 0]
    if not parts:
        raise ValueError("empty embedding bank")
    bank = torch.cat(parts, dim=0)
    if max_bank is not None and bank.size(0) > max_bank:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(bank.size(0), generator=rng)[:max_bank]
        bank = bank[idx]
    return KernelProbeLM(bank=bank, sigma=sigma, eps=eps)


def kernel_deformation_gain(
    y_emb: torch.Tensor,
    probe_embs: List[torch.Tensor],
    base_lm: KernelProbeLM,
    *,
    probe_densities: List[torch.Tensor] | None = None,
    baseline_bits: float | None = None,
) -> float:
    """
    D(y, H, P) with efficient H∪{y} density update:

      p_{H∪y}(e) = (|H| p_H(e) + Σ_j K(e, y_j)) / (|H| + |y|)

    Optional probe_densities / baseline_bits avoid recomputing bank kernels
    when scoring many y against a fixed domain context.
    """
    y = torch.as_tensor(y_emb, dtype=torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if y.numel() == 0 or y.size(0) == 0:
        return 0.0

    n_h = base_lm.n_bank
    n_y = int(y.size(0))
    if n_h == 0:
        updated = KernelProbeLM(bank=y, sigma=base_lm.sigma, eps=base_lm.eps)
        baseline = sum(
            float((-torch.log2(torch.full((p.size(0),), base_lm.eps))).sum())
            if p.numel()
            else 0.0
            for p in probe_embs
        )
        conditional = sum(updated.sequence_bits(p) for p in probe_embs)
        return float(baseline - conditional)

    sigma = max(base_lm.sigma, 1e-8)
    inv = 1.0 / (2.0 * sigma * sigma)
    eps = base_lm.eps

    if probe_densities is None:
        probe_densities = []
        for p in probe_embs:
            rows = torch.as_tensor(p, dtype=torch.float32)
            if rows.numel() == 0:
                probe_densities.append(torch.zeros(0, dtype=torch.float64))
            else:
                if rows.ndim == 1:
                    rows = rows.unsqueeze(0)
                probe_densities.append(base_lm.densities(rows).clamp_min(eps))

    if baseline_bits is None:
        baseline_bits = 0.0
        for dens_h in probe_densities:
            if dens_h.numel():
                baseline_bits += float((-torch.log2(dens_h.clamp_min(eps))).sum().item())

    conditional = 0.0
    for p, dens_h in zip(probe_embs, probe_densities):
        rows = torch.as_tensor(p, dtype=torch.float32)
        if rows.numel() == 0:
            continue
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        dens_h = dens_h.clamp_min(eps)
        d2 = torch.cdist(rows, y).pow(2)
        k_sum = torch.exp((-d2 * inv).to(torch.float64)).sum(dim=1)
        dens_uy = (float(n_h) * dens_h.to(torch.float64) + k_sum) / float(n_h + n_y)
        dens_uy = dens_uy.clamp_min(eps)
        conditional += float((-torch.log2(dens_uy)).sum().item())

    return float(baseline_bits - conditional)


__all__ = [
    "KernelProbeLM",
    "kernel_deformation_gain",
    "lambda_d_normalize",
    "rbf_kernel_mean",
    "train_kernel_probe_lm",
]
