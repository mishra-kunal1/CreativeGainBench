"""
Axiom / regression tests for true-deformation R_D (CountNgramLM).

These encode the failure modes of the removed KenLM prefix-conditioning
proxy plus AbstractCompressor axioms from bridged_validation.
"""

import math
import random

import pytest

from creativegainbench.metrics.count_ngram import (
    CountNgramLM,
    deformation_gain,
    lambda_d_normalize,
    train_count_lm,
)

VOCAB = 64


def _corpus_model(seed: int = 0, n_seqs: int = 200) -> CountNgramLM:
    """Corpus H: sequences from a biased Markov-ish process over 0..31."""
    rng = random.Random(seed)
    seqs = []
    for _ in range(n_seqs):
        seq = [rng.randrange(32)]
        for _ in range(rng.randint(8, 20)):
            seq.append((seq[-1] + rng.choice([0, 1, 1, 2])) % 32)
        seqs.append(seq)
    model = train_count_lm(seqs, order=3)
    model._test_seqs = seqs  # type: ignore[attr-defined]
    return model


PROBES = [
    [40, 41, 42, 43, 40, 41, 42, 43],
    [50, 51, 52, 50, 51, 52, 50, 51],
    [45, 46, 47, 48, 45, 46, 47, 48],
]


def test_code_length_nonnegative():
    model = _corpus_model()
    for probe in PROBES:
        assert model.sequence_bits(probe) >= 0.0


def test_copy_of_corpus_deforms_nothing():
    """Copy minimality: y already in H adds ~no information about probes."""
    model = _corpus_model()
    y_copy = model._test_seqs[0]  # type: ignore[attr-defined]
    raw = deformation_gain(y_copy, PROBES, model)
    assert abs(raw) < 1.0


def test_probe_structured_output_deforms_positively():
    """y exhibiting the probes' (novel-to-H) structure compresses them."""
    model = _corpus_model()
    y_structured = [40, 41, 42, 43, 40, 41, 42, 43, 50, 51, 52, 50, 51, 52]
    y_unrelated = model._test_seqs[1]  # type: ignore[attr-defined]
    raw_structured = deformation_gain(y_structured, PROBES, model)
    raw_unrelated = deformation_gain(y_unrelated, PROBES, model)
    assert raw_structured > 5.0
    assert raw_structured > raw_unrelated + 5.0


def test_full_sequence_sensitivity():
    """Regression vs removed KenLM proxy: symbols before last order-1 matter."""
    model = _corpus_model()
    y_a = [40, 41, 42, 43, 40, 41, 0, 1]
    y_b = [9, 9, 9, 9, 9, 9, 0, 1]
    raw_a = deformation_gain(y_a, PROBES, model)
    raw_b = deformation_gain(y_b, PROBES, model)
    assert abs(raw_a - raw_b) > 1.0


def test_padding_does_not_inflate():
    """Repetitive padding must not increase deformation."""
    model = _corpus_model()
    y = [40, 41, 42, 43]
    y_padded = y + [7] * 50
    raw = deformation_gain(y, PROBES, model)
    raw_padded = deformation_gain(y_padded, PROBES, model)
    assert raw_padded <= raw + 1.0


def test_clone_does_not_mutate_base():
    model = _corpus_model()
    before = model.sequence_bits(PROBES[0])
    deformation_gain([40, 41, 42, 43] * 4, PROBES, model)
    after = model.sequence_bits(PROBES[0])
    assert before == pytest.approx(after)


def test_normalization_scale():
    """Normalized value uses |P| * log|Sigma| * mean probe length."""
    model = _corpus_model()
    y = [40, 41, 42, 43, 40, 41, 42, 43]
    raw = deformation_gain(y, PROBES, model)
    mean_len = sum(len(p) for p in PROBES) / len(PROBES)
    norm = lambda_d_normalize(
        raw, n_probes=len(PROBES), vocab_size=VOCAB, mean_probe_len=mean_len
    )
    assert norm == pytest.approx(raw / (len(PROBES) * math.log(VOCAB) * mean_len))
