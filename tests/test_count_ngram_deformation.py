"""Axioms for CountNgramLM / true deformation R_D."""

from __future__ import annotations

from creativegainbench.metrics.count_ngram import (
    deformation_gain,
    lambda_d_normalize,
    train_count_lm,
)


def test_sequence_bits_nonnegative():
    lm = train_count_lm([[0, 1, 2], [1, 2, 0, 3], [0, 0, 1]], order=3)
    assert lm.sequence_bits([0, 1, 2]) >= 0.0
    assert lm.sequence_bits([]) == 0.0


def test_empty_y_zero_deformation():
    corpus = [[0, 1, 2, 3], [1, 2, 3, 0], [0, 1, 1, 2]]
    probes = [[0, 1, 2], [1, 2, 3]]
    lm = train_count_lm(corpus, order=3)
    assert deformation_gain([], probes, lm) == 0.0


def test_reinforcing_probe_is_nonnegative():
    """Incorporating a probe sequence itself should not make probes costlier."""
    corpus = [[0, 1, 2, 3], [1, 2, 3, 0], [0, 1, 1, 2], [3, 2, 1, 0]]
    probes = [[0, 1, 2], [1, 2, 3]]
    lm = train_count_lm(corpus, order=3)
    d = deformation_gain(probes[0], probes, lm)
    assert d >= -1e-6


def test_lambda_d_normalize_scales_with_probe_count():
    raw = 100.0
    a = lambda_d_normalize(raw, n_probes=10, vocab_size=512, mean_probe_len=20)
    b = lambda_d_normalize(raw, n_probes=20, vocab_size=512, mean_probe_len=20)
    assert a > b
    assert a > 0
