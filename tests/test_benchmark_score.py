"""Gate closure and R_creativity plumbing tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from creativegainbench.artifacts.prepare import prepare_artifacts
from creativegainbench.benchmark_score import compute_r_creativity, d_gate
from creativegainbench.ideas.artifacts import load_artifacts
from creativegainbench.metrics.cue import CUEModel, cue_gate
from creativegainbench.metrics.structural_novelty import structural_novelty_gate
from creativegainbench.receivers.hash_receiver import HashReceiverAgent


ARTIFACTS = Path(__file__).resolve().parents[1] / "src" / "creativegainbench" / "artifacts"


@pytest.fixture(scope="module")
def pipeline():
    prepare_artifacts(artifacts_root=ARTIFACTS)
    return load_artifacts(version="v1", device="cpu", artifacts_root=ARTIFACTS)


@pytest.fixture
def receiver(pipeline):
    return HashReceiverAgent(
        span_encoder=pipeline.span_encoder,
        boundary_detector=pipeline.boundary_detector,
        boundary_threshold=pipeline.boundary_threshold,
        seed=pipeline.seed,
    )


def test_cue_gate():
    assert cue_gate(0.1) == 1.0
    assert cue_gate(0.0) == 0.0
    assert cue_gate(-1.0) == 0.0


def test_d_gate():
    assert d_gate(1.0, 0.0) == 1.0
    assert d_gate(0.0, 0.0) == 0.0
    assert structural_novelty_gate(0.5, 0.4) is True


def test_cue_gate_closes_score(pipeline, receiver):
    y = "A creative restructuring of the scientific method for urban bees."
    # Explicit nonpositive CUE via zero brier delta.
    cue_model = CUEModel(brier_delta=0.0, bit_length=64.0)
    # compute_cue = 0 → gate closed
    result = compute_r_creativity(
        y,
        pipeline=pipeline,
        receiver=receiver,
        cue_model=cue_model,
        n_samples=2,
        use_stub_cue=False,
    )
    assert result.cue_gate == 0.0
    assert result.score == 0.0


def test_stub_cue_opens_gate(pipeline, receiver):
    y = (
        "We restructure the probe space by introducing a new clustering of "
        "idea units that changes compression of held-out scientific claims. "
        "First, segment propositions. Second, quantize. Third, condition."
    )
    result = compute_r_creativity(
        y,
        pipeline=pipeline,
        receiver=receiver,
        n_samples=2,
        use_stub_cue=True,
        delta_d=-1e9,  # force D-gate open for this plumbing test
    )
    assert result.stub_cue is True
    assert result.cue_gate == 1.0
    assert result.d_gate == 1.0
    assert result.score > 0.0
    assert 0.0 <= result.r_b <= 1.0


def test_receiver_expansion_bounded(pipeline, receiver):
    y = "Invent a myth about a river that remembers broken promises."
    result = compute_r_creativity(
        y,
        pipeline=pipeline,
        receiver=receiver,
        n_samples=3,
        use_stub_cue=True,
        delta_d=-1e9,
    )
    assert 0.0 <= result.r_b <= 1.0
