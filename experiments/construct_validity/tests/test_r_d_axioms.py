"""Pytest axiom checks (also covered by E2 against live domains)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from creativegainbench.metrics.count_ngram import deformation_gain, train_count_lm


def test_empty_y_zero():
    lm = train_count_lm([[0, 1, 2], [1, 2, 0]], order=3)
    assert deformation_gain([], [[0, 1], [1, 2]], lm) == 0.0


def test_bits_nonnegative():
    lm = train_count_lm([[0, 1, 2, 3], [3, 2, 1, 0]], order=3)
    assert lm.sequence_bits([0, 1, 2]) >= 0.0


def test_reinforce_probe_nonnegative():
    corpus = [[0, 1, 2, 3], [1, 2, 3, 0], [0, 1, 1, 2]]
    probes = [[0, 1, 2]]
    lm = train_count_lm(corpus, order=3)
    assert deformation_gain(probes[0], probes, lm) >= -1e-6
