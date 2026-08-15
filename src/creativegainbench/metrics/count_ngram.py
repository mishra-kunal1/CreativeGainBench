"""
Count-based n-gram LM over idea-cluster symbols for true corpus deformation.

Implements the Lean ProbeCompressor / AbstractCompressor contract:
  N(s' | H) = L(H ++ s') - L(H)  ≈  cross-entropy bits of s' under LM(H)
  D(y, H, P) = Σ_{s'∈P} ( N(s'|H) - N(s'|H ∪ {y}) )

Adding y updates counts (H ∪ {y}); copies already in H deform ≈ 0 by
copy-minimality.
"""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


def _pad(seq: Sequence[int], order: int) -> List[str]:
    tokens = [f"i{int(s)}" for s in seq]
    return ["<s>"] * (order - 1) + tokens + ["</s>"]


@dataclass
class CountNgramLM:
    """Mutable Witten–Bell n-gram model over idea-symbol tokens."""

    order: int = 3
    counts: list[Counter] = field(default_factory=list)
    context_totals: list[Counter] = field(default_factory=list)
    continuations: list[dict] = field(default_factory=list)
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [Counter() for _ in range(self.order)]
            self.context_totals = [Counter() for _ in range(self.order)]
            self.continuations = [dict() for _ in range(self.order)]

    def clone(self) -> "CountNgramLM":
        return CountNgramLM(
            order=self.order,
            counts=[Counter(c) for c in self.counts],
            context_totals=[Counter(c) for c in self.context_totals],
            continuations=[
                {k: set(v) for k, v in cont.items()} for cont in self.continuations
            ],
            n_tokens=self.n_tokens,
        )

    def add_sequence(self, symbols: Sequence[int]) -> None:
        if not symbols:
            return
        padded = _pad(symbols, self.order)
        for i in range(self.order - 1, len(padded)):
            for n in range(1, self.order + 1):
                if i - n + 1 < 0:
                    continue
                gram = tuple(padded[i - n + 1 : i + 1])
                ctx = gram[:-1]
                word = gram[-1]
                self.counts[n - 1][gram] += 1
                self.context_totals[n - 1][ctx] += 1
                cont = self.continuations[n - 1]
                if ctx not in cont:
                    cont[ctx] = set()
                cont[ctx].add(word)
                if n == 1 and word not in {"<s>"}:
                    self.n_tokens += 1

    def add_sequences(self, sequences: Iterable[Sequence[int]]) -> None:
        for seq in sequences:
            self.add_sequence(seq)

    def _prob(self, gram: tuple[str, ...]) -> float:
        """Witten–Bell recursive probability."""
        n = len(gram)
        if n == 0:
            return 1.0
        if n == 1:
            word = gram[0]
            uni_total = max(self.n_tokens, 1)
            c = self.counts[0].get(gram, 0)
            if c > 0:
                return c / uni_total
            # Uniform mass over unknown; vocab proxy.
            vocab = max(len(self.counts[0]), 2)
            return 1.0 / (uni_total * vocab)

        ctx = gram[:-1]
        word = gram[-1]
        c = self.counts[n - 1].get(gram, 0)
        n_ctx = self.context_totals[n - 1].get(ctx, 0)
        t_ctx = len(self.continuations[n - 1].get(ctx, set()))
        if n_ctx + t_ctx == 0:
            return self._prob(gram[1:])
        if c > 0:
            return c / (n_ctx + t_ctx)
        # Backoff: remaining mass distributed via lower-order.
        remaining = t_ctx / (n_ctx + t_ctx) if t_ctx > 0 else 1.0
        return remaining * self._prob(gram[1:])

    def sequence_bits(self, symbols: Sequence[int]) -> float:
        """Cross-entropy bits of symbol sequence under this LM."""
        if not symbols:
            return 0.0
        padded = _pad(symbols, self.order)
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            gram = tuple(padded[max(0, i - self.order + 1) : i + 1])
            # Use longest available gram ending at i.
            n = min(self.order, i + 1)
            gram = tuple(padded[i - n + 1 : i + 1])
            p = max(self._prob(gram), 1e-99)
            total += -math.log2(p)
        return float(total)


def train_count_lm(
    sequences: List[List[int]], order: int = 3
) -> CountNgramLM:
    lm = CountNgramLM(order=order)
    lm.add_sequences(sequences)
    return lm


def deformation_gain(
    y_symbols: Sequence[int],
    probe_symbol_seqs: List[List[int]],
    base_lm: CountNgramLM,
) -> float:
    """
    D(y, H, P) = Σ (bits_H(s') - bits_{H∪{y}}(s')).

    Positive when incorporating y makes probes cheaper to encode.
    """
    baseline = sum(base_lm.sequence_bits(p) for p in probe_symbol_seqs)
    updated = base_lm.clone()
    updated.add_sequence(y_symbols)
    conditional = sum(updated.sequence_bits(p) for p in probe_symbol_seqs)
    return float(baseline - conditional)


def lambda_d_normalize(
    raw_d: float,
    *,
    n_probes: int,
    vocab_size: int,
    mean_probe_len: float,
) -> float:
    """
    Scale R_D by the Lean λ_D denominator:
      |P| · log|Σ| · E[|s'|]
    so scores are comparable across domain probe sets of different sizes.
    """
    denom = n_probes * math.log(max(vocab_size, 2)) * max(mean_probe_len, 1.0)
    if denom <= 0:
        return 0.0
    return float(raw_d / denom)
