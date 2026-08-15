"""
Utility / feasibility proxy U(q, y) for gated rewards.

Lean (bridged_validation): RewardD and RewardB are utility-gated —
infeasible outputs get zero reward even if they deform probes or expand
entropy. This module supplies a minimal poetry-domain U-proxy until a
stronger task oracle exists.

Convention:
  feasibility_bit(y) ≈ 1[U(q,y) ≥ τ] with a domain-minimal τ
  (non-empty, ≥2 non-blank lines). The prompt q is currently unused for
  poetry (form/tone are soft); the signature keeps q for future oracles.
"""

from __future__ import annotations


def feasibility_bit(text: str, prompt: str | None = None) -> bool:
    """
    Minimal U-proxy for poetry: non-empty, ≥2 non-blank lines.

    ``prompt`` is reserved for stronger U(q,y) oracles; ignored for now.
    """
    del prompt  # reserved
    t = (text or "").strip()
    if not t:
        return False
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return len(lines) >= 2


def utility_gate(
    r_value: float,
    threshold: float,
    *,
    feasible: bool,
) -> float:
    """1[feasible] · 1[r_value > threshold] — shared gate shape for D (and optionally B)."""
    return 1.0 if feasible and r_value > threshold else 0.0
