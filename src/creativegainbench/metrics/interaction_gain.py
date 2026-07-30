"""
G_k interaction gain: joint downstream entropy minus max single-agent entropy.

Aligned with Lean `kInteractionGain`. Entropy can be supplied directly or
estimated via soft clustering over the frozen idea codebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import torch
import torch.nn as nn

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import mean_pool_idea_embeddings
from creativegainbench.metrics.receiver_expansion import soft_cluster_entropy


@dataclass
class MASOutputs:
    """Multi-agent outputs for one query."""

    agent_texts: Sequence[str]
    joint_text: str
    agent_entropies: Sequence[float] | None = None
    joint_entropy: float | None = None


def _length_entropy_proxy(text: str) -> float:
    n = max(len((text or "").split()), 1)
    return math.log(n + 1.0)


def embedding_entropy(
    text: str,
    *,
    span_encoder: nn.Module,
    centroids: torch.Tensor,
    boundary_detector: IdeaBoundaryDetector | None = None,
    boundary_threshold: float = 0.5,
    temperature: float = 1.0,
) -> float:
    """Soft-cluster entropy of a single text in idea-codebook space."""
    emb = mean_pool_idea_embeddings(
        text,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        boundary_threshold=boundary_threshold,
    ).unsqueeze(0)
    return soft_cluster_entropy(emb, centroids, temperature=temperature)


def mas_outputs_from_row(
    row: dict,
    *,
    span_encoder: nn.Module | None = None,
    centroids: torch.Tensor | None = None,
    boundary_detector: IdeaBoundaryDetector | None = None,
    boundary_threshold: float = 0.5,
) -> MASOutputs | None:
    """Build MASOutputs from mas_infer JSONL row."""
    agents = row.get("agent_texts")
    joint = row.get("joint_text")
    if not agents or not joint:
        return None
    if span_encoder is not None and centroids is not None:
        agent_h = [
            embedding_entropy(
                t,
                span_encoder=span_encoder,
                centroids=centroids,
                boundary_detector=boundary_detector,
                boundary_threshold=boundary_threshold,
            )
            for t in agents
        ]
        joint_h = embedding_entropy(
            joint,
            span_encoder=span_encoder,
            centroids=centroids,
            boundary_detector=boundary_detector,
            boundary_threshold=boundary_threshold,
        )
        return MASOutputs(
            agent_texts=list(agents),
            joint_text=joint,
            agent_entropies=agent_h,
            joint_entropy=joint_h,
        )
    return MASOutputs(agent_texts=list(agents), joint_text=joint)


def compute_interaction_gain(mas: MASOutputs | None) -> float:
    """
    G_k = H_joint - max_i H_i  (clipped at 0).

    Returns 0.0 when mas is None (single-agent mode).
    """
    if mas is None or not mas.agent_texts:
        return 0.0

    if mas.agent_entropies is not None and mas.joint_entropy is not None:
        agent_h = list(mas.agent_entropies)
        joint_h = float(mas.joint_entropy)
    else:
        agent_h = [_length_entropy_proxy(t) for t in mas.agent_texts]
        joint_h = _length_entropy_proxy(mas.joint_text)

    if not agent_h:
        return 0.0
    return float(max(0.0, joint_h - max(agent_h)))
