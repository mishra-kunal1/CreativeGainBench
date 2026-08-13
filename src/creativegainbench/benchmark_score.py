"""
Canonical Benchmark Score / R_creativity.

  R_creativity = 1[CUE > 0] · 1[R_D > δ_D]
                 · ( CUE · (1 + α · R_B^{→A}) + λ_G · G_k )

Mirrors Lean `Creativity.CUE.Rcreativity`. Both gates are multiplicative
safety floors on the *entire* score, including λ_G · G_k.

R_D is ProbeCompressor true deformation (CountNgram), never a KenLM proxy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, List

from creativegainbench.ideas.artifacts import IdeaPipeline
from creativegainbench.ideas.idea_extractor import default_sentence_splitter
from creativegainbench.metrics.cue import CUEModel, compute_cue, cue_gate, stub_positive_cue
from creativegainbench.metrics.delta_d import d_gate  # re-exported for callers/tests
from creativegainbench.metrics.deformation import compute_deformation
from creativegainbench.metrics.feasibility import feasibility_bit
from creativegainbench.metrics.interaction_gain import (
    G_K_SURFACE,
    MASOutputs,
    compute_interaction_gain,
)
from creativegainbench.metrics.receiver_expansion import (
    ReceiverAgent,
    compute_receiver_expansion,
)

__all__ = ["BenchmarkResult", "compute_r_creativity", "d_gate", "feasibility_bit"]


@dataclass
class BenchmarkResult:
    score: float
    cue: float
    r_d: float
    r_b: float
    g_k: float
    cue_gate: float
    d_gate: float
    alpha: float
    lambda_g: float
    delta_d: float
    stub_cue: bool = False
    feasible: bool = True
    g_k_kind: str = "G_k_surface"  # F8: surface vs conditioned
    edge_cue_chain: list | None = None
    handoff_gain_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_r_creativity(
    y: str,
    *,
    pipeline: IdeaPipeline,
    receiver: ReceiverAgent,
    cue_model: CUEModel | None = None,
    mas_outputs: MASOutputs | None = None,
    alpha: float = 1.0,
    lambda_g: float = 0.0,
    delta_d: float = 0.0,
    n_samples: int,
    temperature: float = 1.0,
    use_stub_cue: bool = False,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    edge_cue_chain: list | None = None,
    handoff_gain_rate: float | None = None,
    prompt: str | None = None,
) -> BenchmarkResult:
    """
    Compute the gated creativity score for a single output y.

    R_D gate includes the utility/feasibility bit (Lean RewardD).
    R_B is intentionally *ungated* here (BBase-style baseline; see
    receiver_expansion.R_B_FEASIBILITY_GATED).
    """
    stub_cue = False
    if cue_model is not None:
        cue_val = compute_cue(cue_model)
    elif use_stub_cue:
        cue_val, cue_model = stub_positive_cue(y)
        stub_cue = True
    else:
        # Default: no calibrated receiver → CUE gate closed (score 0).
        cue_val = 0.0

    feasible = feasibility_bit(y, prompt)

    splitter = sentence_splitter or default_sentence_splitter
    deform = compute_deformation(
        y,
        pipeline.deformation_ctx,
        span_encoder=pipeline.span_encoder,
        codebook=pipeline.codebook,
        boundary_detector=pipeline.boundary_detector,
        sentence_splitter=splitter,
        boundary_threshold=pipeline.boundary_threshold,
    )
    r_d = deform.r_d_norm

    r_b = compute_receiver_expansion(
        y,
        receiver_agent=receiver,
        task_battery=pipeline.task_battery,
        idea_codebook_centroids=pipeline.codebook.centroids,
        n_samples=n_samples,
        temperature=temperature,
        device=pipeline.device,
    )

    g_k = compute_interaction_gain(mas_outputs)

    cg = cue_gate(cue_val)
    dg = d_gate(r_d, delta_d, feasible=feasible)
    inner = cue_val * (1.0 + alpha * r_b) + lambda_g * g_k
    score = cg * dg * inner

    return BenchmarkResult(
        score=float(score),
        cue=float(cue_val),
        r_d=float(r_d),
        r_b=float(r_b),
        g_k=float(g_k),
        cue_gate=float(cg),
        d_gate=float(dg),
        alpha=float(alpha),
        lambda_g=float(lambda_g),
        delta_d=float(delta_d),
        stub_cue=stub_cue,
        feasible=feasible,
        g_k_kind=G_K_SURFACE,
        edge_cue_chain=edge_cue_chain,
        handoff_gain_rate=handoff_gain_rate,
    )
