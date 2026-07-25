"""
Canonical Benchmark Score / R_creativity.

  R_creativity = 1[CUE > 0] · 1[R_D > δ_D]
                 · ( CUE · (1 + α · R_B^{→A}) + λ_G · G_k )

Mirrors Lean `Creativity.CUE.Rcreativity`. Both gates are multiplicative
safety floors on the *entire* score, including λ_G · G_k.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from creativegainbench.ideas.artifacts import IdeaPipeline
from creativegainbench.metrics.cue import CUEModel, compute_cue, cue_gate, stub_positive_cue
from creativegainbench.metrics.interaction_gain import MASOutputs, compute_interaction_gain
from creativegainbench.metrics.receiver_expansion import (
    ReceiverAgent,
    compute_receiver_expansion,
)
from creativegainbench.metrics.structural_novelty import (
    compute_structural_novelty,
    structural_novelty_gate,
)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def d_gate(r_d: float, delta_d: float) -> float:
    """1[R_D > δ_D]."""
    return 1.0 if structural_novelty_gate(r_d, delta_d) else 0.0


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
    n_samples: int = 8,
    temperature: float = 1.0,
    use_stub_cue: bool = False,
) -> BenchmarkResult:
    """
    Compute the gated creativity score for a single output y.
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

    r_d = compute_structural_novelty(
        y,
        probe_set=pipeline.probe_set,
        compressor=pipeline.compressor,
        codebook=pipeline.codebook,
        span_encoder=pipeline.span_encoder,
        boundary_detector=pipeline.boundary_detector,
        n=pipeline.n,
        boundary_threshold=pipeline.boundary_threshold,
        device=pipeline.device,
    )

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
    dg = d_gate(r_d, delta_d)
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
    )
