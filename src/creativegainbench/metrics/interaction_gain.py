"""
G_k interaction gain: joint entropy minus max single-agent entropy.

Two estimators (F8 / V8):

* ``compute_interaction_gain`` / ``mas_outputs_from_row`` —
  **G_k_surface**: soft-cluster entropy of the raw agent/joint *texts*
  (or a length proxy). This is a diversity proxy, **not** Lean-certified
  irreducibility under receiver calibration.

* ``compute_interaction_gain_conditioned`` —
  entropy of *downstream-conditioned* receiver samples given each text,
  matching R_B's sampler. Closer to Lean MASBridge / kInteractionGain
  semantics; requires a ReceiverAgent + task battery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import mean_pool_idea_embeddings
from creativegainbench.metrics.receiver_expansion import (
    ReceiverAgent,
    soft_cluster_entropy,
)

# Explicit labels for reporting / validation (F8).
G_K_SURFACE = "G_k_surface"
G_K_CONDITIONED = "G_k_conditioned"


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
    """Build MASOutputs with G_k_surface entropies from raw texts."""
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
    G_k_surface = max(0, H_joint - max_i H_i) on raw-text entropies.

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


def _downstream_entropy(
    y: str,
    *,
    receiver_agent: ReceiverAgent,
    task_battery: list[dict],
    centroids: torch.Tensor,
    n_samples: int,
    temperature: float,
    device: str,
) -> float:
    if not task_battery:
        return 0.0
    entropies: list[float] = []
    for q in task_battery:
        conditioned = receiver_agent.condition(q["input"], context=y)
        _samples, embeddings = receiver_agent.sample_with_embeddings(
            conditioned, n=n_samples
        )
        entropies.append(
            soft_cluster_entropy(
                embeddings.to(device),
                centroids.to(device),
                temperature=temperature,
            )
        )
    return float(sum(entropies) / len(entropies))


def compute_interaction_gain_conditioned(
    mas: MASOutputs | None,
    *,
    receiver_agent: ReceiverAgent,
    task_battery: list[dict],
    idea_codebook_centroids: torch.Tensor,
    n_samples: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> float:
    """
    G_k_conditioned: same H_joint - max H_i shape, but H is the entropy of
    receiver samples conditioned on each agent/joint text (R_B-style).
    """
    if mas is None or not mas.agent_texts:
        return 0.0
    agent_h = [
        _downstream_entropy(
            t,
            receiver_agent=receiver_agent,
            task_battery=task_battery,
            centroids=idea_codebook_centroids,
            n_samples=n_samples,
            temperature=temperature,
            device=device,
        )
        for t in mas.agent_texts
    ]
    joint_h = _downstream_entropy(
        mas.joint_text,
        receiver_agent=receiver_agent,
        task_battery=task_battery,
        centroids=idea_codebook_centroids,
        n_samples=n_samples,
        temperature=temperature,
        device=device,
    )
    if not agent_h:
        return 0.0
    return float(max(0.0, joint_h - max(agent_h)))
