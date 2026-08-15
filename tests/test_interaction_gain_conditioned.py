"""F8: G_k_surface vs conditioned API."""

from creativegainbench.metrics.interaction_gain import (
    G_K_CONDITIONED,
    G_K_SURFACE,
    MASOutputs,
    compute_interaction_gain,
    compute_interaction_gain_conditioned,
)
from creativegainbench.receivers.hash_receiver import HashReceiverAgent
from creativegainbench.ideas.idea_extractor import HashSpanEncoder
import torch


def test_surface_gain_nonneg():
    mas = MASOutputs(
        agent_texts=["short", "a bit longer text here"],
        joint_text="a bit longer text here and more ideas about rivers",
    )
    g = compute_interaction_gain(mas)
    assert g >= 0.0


def test_conditioned_api_runs():
    enc = HashSpanEncoder(embedding_dim=32)
    centroids = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
    rx = HashReceiverAgent(
        span_encoder=enc,
        boundary_detector=None,
        boundary_threshold=0.5,
        seed=0,
    )
    mas = MASOutputs(
        agent_texts=["solo a", "solo b", "solo c"],
        joint_text="joint answer with several idea units",
    )
    battery = [{"input": "task one"}, {"input": "task two"}]
    g = compute_interaction_gain_conditioned(
        mas,
        receiver_agent=rx,
        task_battery=battery,
        idea_codebook_centroids=centroids,
        n_samples=2,
    )
    assert g >= 0.0
    assert G_K_SURFACE == "G_k_surface"
    assert G_K_CONDITIONED == "G_k_conditioned"
