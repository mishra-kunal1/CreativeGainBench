"""Unit tests for Edge-CUE (offline stub receiver)."""

from __future__ import annotations

from creativegainbench.metrics.cue import CUEModel, brier_delta, compute_cue, cue_gate
from creativegainbench.metrics.edge_cue import compute_edge_cue, handoff_gain_rate


class StubReceiver:
    outcomes = ("a", "b", "c", "d")

    def elicit_prior_conditioned(self, task, upstream):
        return [0.4, 0.3, 0.2, 0.1]

    def elicit_posterior(self, task, y):
        return [0.1, 0.1, 0.1, 0.7]

    def classify_outcome(self, task, y):
        return 3


def test_edge_cue_matches_compute_cue():
    prior = [0.4, 0.3, 0.2, 0.1]
    posterior = [0.1, 0.1, 0.1, 0.7]
    outcome = 3
    downstream = "hello world"  # 11 bytes → 88 bits
    expected_delta = brier_delta(prior, posterior, outcome)
    expected_bits = 11 * 8
    expected = compute_cue(CUEModel(brier_delta=expected_delta, bit_length=expected_bits))

    cue_val, model, diag = compute_edge_cue(
        "upstream draft",
        downstream,
        StubReceiver(),  # type: ignore[arg-type]
        "task",
    )
    assert cue_val == expected
    assert model.brier_delta == expected_delta
    assert diag["gate"] == cue_gate(cue_val)


def test_handoff_gain_rate():
    chain = [{"gate": 1.0}, {"gate": 0.0}, {"gate": 1.0}]
    assert handoff_gain_rate(chain) == 2.0 / 3.0
    assert handoff_gain_rate([]) == 0.0
