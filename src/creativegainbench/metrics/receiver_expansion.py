"""
R_B^{→A}(y, A) = H(Gen(A | y, T)) / log|W|, via torch soft clustering
over receiver-generation embeddings, using the SAME frozen idea codebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import torch
import torch.nn.functional as F


@dataclass
class GenerationDistribution:
    samples: List[str]
    embeddings: torch.Tensor  # (n_samples, d)


class ReceiverAgent(Protocol):
    def condition(self, task_input: str, *, context: str) -> str: ...

    def sample_with_embeddings(
        self, conditioned_input: str, n: int
    ) -> tuple[list[str], torch.Tensor]: ...


def soft_cluster_entropy(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """
    Soft-assignment entropy of the average assignment distribution over
    frozen idea-cluster centroids.
    """
    if embeddings.numel() == 0:
        return 0.0
    dists = torch.cdist(embeddings, centroids)  # (n_samples, K)
    soft_assign = F.softmax(-dists / max(temperature, 1e-8), dim=-1)
    avg_dist = soft_assign.mean(dim=0)  # (K,)
    entropy = -(avg_dist * torch.log(avg_dist + 1e-12)).sum()
    return float(entropy.item())


def compute_receiver_expansion(
    y: str,
    receiver_agent: ReceiverAgent,
    task_battery: List[dict],
    idea_codebook_centroids: torch.Tensor,
    n_samples: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> float:
    """
    Normalized expansion label in [0, 1]: mean soft-cluster entropy / log(K).
    """
    if not task_battery:
        return 0.0

    centroids = idea_codebook_centroids.to(device)
    entropies: list[float] = []
    for q in task_battery:
        conditioned_input = receiver_agent.condition(q["input"], context=y)
        _samples, embeddings = receiver_agent.sample_with_embeddings(
            conditioned_input, n=n_samples
        )
        h = soft_cluster_entropy(
            embeddings.to(device),
            centroids,
            temperature=temperature,
        )
        entropies.append(h)

    avg_entropy = sum(entropies) / len(entropies)
    max_entropy = torch.log(torch.tensor(float(centroids.size(0)), device=device)).item()
    if max_entropy <= 0:
        return 0.0
    return float(max(0.0, min(1.0, avg_entropy / max_entropy)))
