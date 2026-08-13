"""
Edge-CUE: CUEModel instantiated on a multi-agent handoff.

Prior = belief after task + upstream agent text (conditioned prior).
Posterior = belief after downstream agent text.
Formula is unchanged compute_cue / cue_gate — only elicitation differs.

Engineering extension (not a Lean theorem); see math_backing/docs/edge_cue_gap.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from creativegainbench.metrics.cue import (
    CUEModel,
    bit_length_utf8,
    brier_delta,
    compute_cue,
    cue_gate,
)

if TYPE_CHECKING:
    from creativegainbench.metrics.cue_receiver import CUEBeliefReceiver


def compute_edge_cue(
    upstream_output: str,
    downstream_output: str,
    receiver: "CUEBeliefReceiver",
    task_context: str,
    *,
    z_star: int | None = None,
) -> tuple[float, CUEModel, dict]:
    """
    CUE attributable to a specific handoff upstream → downstream.

    Returns (cue_value, CUEModel, diagnostics).
    Bit-length denominator is |downstream| (bits), matching single-output CUE
    on the produced message.
    """
    prior = receiver.elicit_prior_conditioned(task_context, upstream_output)
    posterior = receiver.elicit_posterior(task_context, downstream_output)
    if z_star is not None:
        outcome = int(z_star)
        if not (0 <= outcome < len(receiver.outcomes)):
            raise ValueError(
                f"z_star {outcome} out of range for {len(receiver.outcomes)} outcomes"
            )
        outcome_source = "external"
    else:
        outcome = receiver.classify_outcome(task_context, downstream_output)
        outcome_source = "self_classify"
    bits = max(bit_length_utf8(downstream_output), 8.0)
    delta = brier_delta(prior, posterior, outcome)
    model = CUEModel(brier_delta=delta, bit_length=bits)
    cue_val = compute_cue(model)
    diag = {
        "prior": prior,
        "posterior": posterior,
        "outcome_index": outcome,
        "outcome_label": receiver.outcomes[outcome],
        "outcomes": list(receiver.outcomes),
        "outcome_source": outcome_source,
        "brier_delta": float(delta),
        "bit_length": float(bits),
        "gate": float(cue_gate(cue_val)),
        "cue": float(cue_val),
    }
    return float(cue_val), model, diag


def handoff_gain_rate(edge_cue_chain: list[dict]) -> float:
    """Mean of cue_gate over registered edges (HandoffGain)."""
    if not edge_cue_chain:
        return 0.0
    gates = [float(e.get("gate", 0.0)) for e in edge_cue_chain]
    return float(sum(gates) / len(gates))
