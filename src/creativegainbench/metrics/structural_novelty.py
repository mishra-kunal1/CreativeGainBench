"""
R_D(y, P) = CompressionCost(P) - CompressionCost(P | y)

Compression cost is estimated as the cross-entropy (bits) of an idea-symbol
sequence under a frozen KenLM n-gram language model — NOT gzip / raw text, and
NOT a neural LM over tokens. This measures conceptual/structural novelty:
paraphrases that map to the same idea-cluster sequence compress identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch.nn as nn

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import IdeaCodebook, build_idea_ngram_sequence
from creativegainbench.metrics.kenlm_compressor import (
    KenLMCompressor,
    context_augmented_bits,
    sequence_bits,
)


@dataclass
class ProbeSet:
    strings: List[str]
    seed: int
    strata: List[str]


def compute_structural_novelty(
    y: str,
    probe_set: ProbeSet,
    compressor: KenLMCompressor,
    codebook: IdeaCodebook,
    span_encoder: nn.Module,
    boundary_detector: IdeaBoundaryDetector | None = None,
    sentence_splitter: Callable[[str], List[str]] | None = None,
    n: int = 3,
    boundary_threshold: float = 0.5,
    device: str = "cpu",  # kept for API compatibility; KenLM is CPU
) -> float:
    """
    R_D(y, P): total bits saved on frozen probes when conditioning on y.

    Corresponds to Lean `corpusDeformationGain` with empty background corpus
    and teacher-forced prefix incorporation of y into KenLM state.
    """
    del device  # KenLM scoring is CPU-only.
    _, y_symbols = build_idea_ngram_sequence(
        y,
        span_encoder,
        codebook,
        boundary_detector=boundary_detector,
        sentence_splitter=sentence_splitter,
        n=n,
        boundary_threshold=boundary_threshold,
    )

    baseline_total_bits = 0.0
    conditional_total_bits = 0.0
    for probe in probe_set.strings:
        _, probe_symbols = build_idea_ngram_sequence(
            probe,
            span_encoder,
            codebook,
            boundary_detector=boundary_detector,
            sentence_splitter=sentence_splitter,
            n=n,
            boundary_threshold=boundary_threshold,
        )
        baseline_total_bits += sequence_bits(compressor, probe_symbols)
        conditional_total_bits += context_augmented_bits(
            compressor, y_symbols, probe_symbols
        )

    return float(baseline_total_bits - conditional_total_bits)


def structural_novelty_gate(r_d_value: float, delta_d: float) -> bool:
    """1[R_D > delta_D] — binary pass/fail filter."""
    return r_d_value > delta_d
