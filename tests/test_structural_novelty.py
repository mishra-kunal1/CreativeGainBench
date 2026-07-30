"""R_D / KenLM idea-sequence compressor tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from creativegainbench.artifacts.prepare import prepare_artifacts
from creativegainbench.ideas.artifacts import load_artifacts
from creativegainbench.ideas.idea_ngram import text_to_idea_symbols
from creativegainbench.metrics.kenlm_compressor import (
    sequence_bits,
    train_kenlm,
    load_kenlm_compressor,
)
from creativegainbench.metrics.structural_novelty import (
    ProbeSet,
    compute_structural_novelty,
)


ARTIFACTS = Path(__file__).resolve().parents[1] / "src" / "creativegainbench" / "artifacts"


@pytest.fixture(scope="module")
def pipeline():
    prepare_artifacts(artifacts_root=ARTIFACTS)
    return load_artifacts(version="v1", device="cpu", artifacts_root=ARTIFACTS)


def test_sequence_bits_empty(tmp_path):
    arpa = train_kenlm([[0, 1, 2], [1, 2, 0]], tmp_path / "m.arpa", order=3)
    comp = load_kenlm_compressor(arpa, order=3)
    assert sequence_bits(comp, []) == 0.0


def test_kenlm_scores_finite(tmp_path):
    arpa = train_kenlm([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 1]], tmp_path / "m.arpa", order=3)
    comp = load_kenlm_compressor(arpa, order=3)
    bits = sequence_bits(comp, [0, 1, 2])
    assert bits > 0.0
    assert torch.isfinite(torch.tensor(bits))


def test_identical_y_and_probe_is_finite(pipeline):
    probe = pipeline.probe_set.strings[0]
    tiny = ProbeSet(strings=[probe], seed=pipeline.seed, strata=[])
    r_d = compute_structural_novelty(
        probe,
        probe_set=tiny,
        compressor=pipeline.compressor,
        codebook=pipeline.codebook,
        span_encoder=pipeline.span_encoder,
        boundary_detector=pipeline.boundary_detector,
        n=pipeline.n,
        boundary_threshold=pipeline.boundary_threshold,
        device="cpu",
    )
    assert isinstance(r_d, float)
    assert torch.isfinite(torch.tensor(r_d))


def test_symbol_pipeline_stable(pipeline):
    text = pipeline.probe_set.strings[1]
    s1 = text_to_idea_symbols(
        text,
        pipeline.span_encoder,
        pipeline.codebook,
        boundary_detector=pipeline.boundary_detector,
        boundary_threshold=pipeline.boundary_threshold,
    )
    s2 = text_to_idea_symbols(
        text,
        pipeline.span_encoder,
        pipeline.codebook,
        boundary_detector=pipeline.boundary_detector,
        boundary_threshold=pipeline.boundary_threshold,
    )
    assert s1 == s2
