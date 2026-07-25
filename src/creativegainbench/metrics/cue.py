"""
Creative Update Efficiency (CUE): receiver-grounded decision-quality gain
normalized by output length in bits.

  CUE(y, A, T) = brierDelta / |y|_bits

Aligned with Lean `Creativity.CUE.cue` / `CUEModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math


@dataclass
class CUEModel:
    """
    Empirical CUE reading for one output.

    Fields mirror Lean `CUEModel` enough for the benchmark score:
    - brier_delta: reduction in Brier loss (prior -> posterior), >= 0
    - bit_length: |y|_bits > 0
    - voi / posterior_entropy are optional diagnostics
    """

    brier_delta: float
    bit_length: float
    voi: float | None = None
    posterior_entropy: float | None = None

    def __post_init__(self) -> None:
        if self.bit_length <= 0:
            raise ValueError(f"bit_length must be > 0, got {self.bit_length}")
        if self.brier_delta < 0:
            raise ValueError(f"brier_delta must be >= 0, got {self.brier_delta}")


def bit_length_utf8(text: str) -> float:
    """Output length in bits via UTF-8 byte length × 8."""
    return float(len((text or "").encode("utf-8")) * 8)


def brier_score(probs: Sequence[float], outcome_index: int) -> float:
    """Multiclass Brier score for a categorical prediction."""
    if not probs:
        raise ValueError("probs must be non-empty")
    if not (0 <= outcome_index < len(probs)):
        raise ValueError(f"outcome_index {outcome_index} out of range for {len(probs)} classes")
    total = 0.0
    for i, p in enumerate(probs):
        target = 1.0 if i == outcome_index else 0.0
        total += (p - target) ** 2
    return total


def brier_delta(
    prior: Sequence[float],
    posterior: Sequence[float],
    outcome_index: int,
) -> float:
    """Nonnegative Brier improvement; clipped at 0 for numerical noise."""
    delta = brier_score(prior, outcome_index) - brier_score(posterior, outcome_index)
    return max(0.0, float(delta))


def compute_cue(model: CUEModel) -> float:
    """CUE = brierDelta / bitLength."""
    return float(model.brier_delta / model.bit_length)


def cue_from_beliefs(
    y: str,
    prior: Sequence[float],
    posterior: Sequence[float],
    outcome_index: int,
) -> tuple[float, CUEModel]:
    """Build a CUEModel from explicit prior/posterior and return (cue, model)."""
    bits = bit_length_utf8(y)
    if bits <= 0:
        bits = 8.0  # single empty-ish token floor for numerical safety
    delta = brier_delta(prior, posterior, outcome_index)
    m = CUEModel(brier_delta=delta, bit_length=bits)
    return compute_cue(m), m


def stub_positive_cue(y: str, magnitude: float = 1.0) -> tuple[float, CUEModel]:
    """
    Pipeline stub: assign a small positive CUE so the CUE gate opens while
    R_D / R_B plumbing is tested. Not a calibrated receiver reading.
    """
    bits = max(bit_length_utf8(y), 8.0)
    # Keep CUE small but strictly positive.
    delta = abs(magnitude) * math.log(2.0)
    m = CUEModel(brier_delta=delta, bit_length=bits)
    return compute_cue(m), m


def cue_gate(cue_value: float) -> float:
    """1[CUE > 0]."""
    return 1.0 if cue_value > 0.0 else 0.0
