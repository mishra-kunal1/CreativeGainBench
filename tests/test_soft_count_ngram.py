"""Unit tests for Soft ProbeCompressor (order-1 soft unigram)."""

from __future__ import annotations

import torch

from creativegainbench.metrics.count_ngram import train_count_lm, deformation_gain
from creativegainbench.metrics.soft_count_ngram import (
    SoftCountLM,
    soft_assign,
    soft_assign_hard_limit,
    soft_deformation_gain,
    train_soft_count_lm,
)


def test_soft_assign_rows_sum_to_one():
    emb = torch.randn(5, 8)
    cents = torch.randn(16, 8)
    pi = soft_assign(emb, cents, tau=1.0)
    assert pi.shape == (5, 16)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_soft_assign_hard_limit_is_one_hot():
    emb = torch.randn(4, 8)
    cents = torch.randn(10, 8)
    hard = soft_assign_hard_limit(emb, cents)
    assert hard.shape == (4, 10)
    assert torch.allclose(hard.sum(dim=-1), torch.ones(4))
    assert int((hard == 1).sum().item()) == 4


def test_incorporation_helps_probes():
    """Lean incorporation_helps: adding a probe itself weakly reduces probe CE."""
    K = 8
    # Fake soft rows: nearly one-hot on different codes
    corpus = [
        torch.nn.functional.one_hot(torch.tensor([0, 1, 2]), num_classes=K).float(),
        torch.nn.functional.one_hot(torch.tensor([1, 2, 3]), num_classes=K).float(),
        torch.nn.functional.one_hot(torch.tensor([0, 0, 1]), num_classes=K).float(),
    ]
    probe = torch.nn.functional.one_hot(torch.tensor([0, 1, 2]), num_classes=K).float()
    lm = train_soft_count_lm(corpus, K)
    d = soft_deformation_gain(probe, [probe], lm)
    assert d >= -1e-6


def test_empty_y_zero_deformation():
    K = 6
    corpus = [
        torch.nn.functional.one_hot(torch.tensor([0, 1, 2]), num_classes=K).float(),
    ]
    probe = torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=K).float()
    lm = train_soft_count_lm(corpus, K)
    empty = torch.zeros(0, K)
    assert soft_deformation_gain(empty, [probe], lm) == 0.0


def test_exact_copy_member_near_zero_d():
    """Exact copy of an H member → near-zero deformation (Lean copy check)."""
    K = 10
    member = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2, 3]), num_classes=K
    ).float()
    corpus = [member.clone(), member.clone()]
    probes = [member.clone()]
    lm = train_soft_count_lm(corpus, K)
    d_member = soft_deformation_gain(member, probes, lm)
    # Already in H: incorporating y barely changes probe CE.
    assert abs(d_member) < 1e-3


def test_tau_to_zero_matches_hard_unigram_bits():
    """
    Soft assignments at hard limit + SoftCountLM vs hard CountNgram order=1
    should agree on sequence bits for one-hot sequences (same unigram model).
    """
    symbols = [0, 1, 2, 1, 0]
    corpus_syms = [[0, 1, 2], [1, 2, 0, 3], [0, 0, 1]]
    K = 8
    hard_lm = train_count_lm(corpus_syms, order=1)
    soft_corpus = [
        soft_assign_hard_limit(
            # Use identity-like embeddings: one row per symbol as e_i
            torch.eye(K)[torch.tensor(seq)],
            torch.eye(K),
        )
        for seq in corpus_syms
    ]
    # With identity centroids, nearest centroid of e_i is i.
    soft_lm = train_soft_count_lm(soft_corpus, K)
    y_soft = soft_assign_hard_limit(torch.eye(K)[torch.tensor(symbols)], torch.eye(K))
    soft_bits = soft_lm.sequence_bits(y_soft)
    hard_bits = hard_lm.sequence_bits(symbols)
    # Different smoothers (eps vs Witten–Bell) → allow relative slack.
    assert abs(soft_bits - hard_bits) / max(hard_bits, 1.0) < 0.5


def test_soft_count_lm_clone_independent():
    K = 4
    rows = torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=K).float()
    lm = SoftCountLM(vocab_size=K)
    lm.add_sequence(rows)
    cloned = lm.clone()
    cloned.add_sequence(rows)
    assert float(lm.counts.sum()) < float(cloned.counts.sum())
