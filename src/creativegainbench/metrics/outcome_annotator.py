"""
External realized-outcome annotator for CUE (z*).

Independent of the belief-elicitation receiver so Brier scoring is not
circular. Uses MiniLM nearest-exemplar over a small frozen label bank.

Phase A (poetry) uses EXEMPLARS. Phase B domains pass ``domain=`` so math /
writing / proposal centroids are fit from DOMAIN_EXEMPLARS without mutating
the poetry bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from creativegainbench.metrics.cue_receiver import DEFAULT_OUTCOMES


# Hand-crafted poetry-relevant exemplars for each outcome label.
EXEMPLARS: dict[str, list[str]] = {
    "novel_structure": [
        "A poem that invents a new stanza engine: each couplet rewrites the previous "
        "metaphor's physics so time runs backward only inside the rhyme.",
        "Verse that reframes grief as a cartographic error — continents of feeling "
        "misdrawn, and the volta is a corrected map legend.",
        "A lyric that braids two incompatible meters until a third rhythm appears "
        "that belongs to neither tradition.",
    ],
    "fluent_paraphrase": [
        "A smooth rewrite of a familiar love poem with synonyms swapped and the "
        "same rhyme scheme and imagery order preserved.",
        "Competent pastiche of Dickinson dashes and capitalization without a new "
        "conceptual turn.",
        "A conventional nature lyric restating sunrise-as-hope in polished language.",
    ],
    "clear_utility": [
        "A clear, well-formed occasional poem that fulfills the prompt's length, "
        "tone, and subject requirements without structural surprise.",
        "A readable commemorative verse that is useful for the occasion but "
        "follows expected tropes.",
        "An on-topic instructional poem that teaches a lesson cleanly and directly.",
    ],
    "low_quality": [
        "Broken lines with no meter, topic drift, and unfinished clauses that "
        "fail the prompt.",
        "Gibberish tokens and repeated filler that do not form a poem.",
        "Off-topic prose that ignores the requested form and subject entirely.",
    ],
}

# Phase B banks — do not write into EXEMPLARS (poetry centroids stay frozen).
MATH_EXEMPLARS: dict[str, list[str]] = {
    "novel_structure": [
        "A proof that recasts induction as well-founded descent on a novel rank "
        "function, making the inductive step a change of measure rather than a "
        "template n→n+1 writeup.",
        "An argument that proves infinitude of primes by constructing a new "
        "multiplicative generating set instead of Euclid's factorial-plus-one.",
        "A derivation that replaces a standard ε-δ chase with a commuting diagram "
        "of estimates that is not in the usual textbook presentation.",
    ],
    "fluent_paraphrase": [
        "A textbook restatement of the usual induction writeup with synonyms "
        "swapped and the same base case / inductive step order preserved.",
        "A fluent rewrite of Euclid's infinitude-of-primes argument without a "
        "new construction.",
        "A polished copy of the standard contradiction proof that √2 is irrational.",
    ],
    "clear_utility": [
        "A complete, correctly structured induction that meets the prompt but "
        "uses only the expected template.",
        "A readable ε-δ proof that is adequate for the exercise without a new idea.",
        "A correctly cited application of a standard lemma that solves the item.",
    ],
    "low_quality": [
        "A list of unjustified equalities with a missing inductive step.",
        "Gibberish symbols and repeated filler that do not form a proof.",
        "An off-topic paragraph that ignores the claimed theorem.",
    ],
}

WRITING_EXEMPLARS: dict[str, list[str]] = {
    "novel_structure": [
        "A story whose plot is told by rearranging the same paragraph under "
        "three incompatible narrators until a fourth voice appears.",
        "A lyric essay that treats time as a misfiled archive, with the volta "
        "as a corrected catalog card.",
        "A scene that braids two incompatible tenses until a third chronology "
        "belongs to neither character.",
    ],
    "fluent_paraphrase": [
        "A smooth rewrite of a familiar quest scene with synonyms swapped and "
        "the same beat order preserved.",
        "Competent pastiche of a famous short-story opening without a new turn.",
        "A conventional love-letter restating absence-as-weather in polished prose.",
    ],
    "clear_utility": [
        "A well-formed occasional piece that hits the prompt's length, tone, "
        "and subject without structural surprise.",
        "Readable commemorative prose that fulfills the brief with expected tropes.",
        "An on-topic instructional vignette that teaches the requested lesson cleanly.",
    ],
    "low_quality": [
        "Broken sentences, topic drift, and unfinished clauses that fail the prompt.",
        "Gibberish tokens and repeated filler that do not form a story.",
        "Off-topic notes that ignore the requested form and subject.",
    ],
}

SCIENCE_EXEMPLARS: dict[str, list[str]] = {
    "novel_structure": [
        "A proposal that reframes a measurement problem as an identification "
        "argument, with a new instrument that is not a restatement of the call.",
        "A design that couples two existing methods so a third estimand appears "
        "that neither method targets alone.",
        "A protocol whose novelty is a change of unit of analysis, not a synonym "
        "swap of the standard RCT template.",
    ],
    "fluent_paraphrase": [
        "A fluent restatement of a standard RCT template with synonyms swapped.",
        "A competent rewrite of an existing sensor-network pitch without a new mechanism.",
        "A polished paraphrase of a typical materials-recycling aims page.",
    ],
    "clear_utility": [
        "A complete aims/methods page that meets the call without a new idea.",
        "A feasible study design that is adequate and conventional.",
        "An on-topic proposal that is useful as a draft but follows expected tropes.",
    ],
    "low_quality": [
        "Vague aims with no measurable endpoint or method.",
        "Gibberish and repeated filler that do not form a proposal.",
        "Off-topic text that ignores the requested scientific question.",
    ],
}

# Domain keys for OutcomeAnnotator(domain=...). Poetry is EXEMPLARS, not this map.
DOMAIN_EXEMPLARS: dict[str, dict[str, list[str]]] = {
    "mathematical_proof": MATH_EXEMPLARS,
    "math": MATH_EXEMPLARS,
    "creative_writing": WRITING_EXEMPLARS,
    "writing": WRITING_EXEMPLARS,
    "scientific_proposal": SCIENCE_EXEMPLARS,
    "science": SCIENCE_EXEMPLARS,
}


def exemplars_for_domain(domain: str | None) -> dict[str, list[str]]:
    """Return the exemplar bank for ``domain`` without mutating poetry EXEMPLARS."""
    if domain is None:
        return EXEMPLARS
    key = str(domain).strip().lower()
    if key in ("", "poetry", "default", "phase_a"):
        return EXEMPLARS
    return DOMAIN_EXEMPLARS.get(str(domain), DOMAIN_EXEMPLARS.get(key, EXEMPLARS))


@dataclass
class OutcomeAnnotator:
    """Nearest-exemplar classifier over DEFAULT_OUTCOMES.

    Pass ``domain=`` for Phase B math/writing/proposal banks. Poetry centroids
    are the default; ``EXEMPLARS`` is never overwritten.
    """

    span_encoder: nn.Module
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
    domain: str = "poetry"
    _label_centroids: torch.Tensor | None = None
    _fitted_domain: str | None = field(default=None, repr=False)

    def fit(self) -> "OutcomeAnnotator":
        bank = exemplars_for_domain(self.domain)
        cents = []
        with torch.no_grad():
            for name in self.outcomes:
                texts = bank.get(name, EXEMPLARS.get(name, [name]))
                emb = self.span_encoder(texts)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                cents.append(emb.mean(dim=0))
        self._label_centroids = torch.stack(cents, dim=0)
        self._label_centroids = (
            self._label_centroids
            / self._label_centroids.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        self._fitted_domain = self.domain
        return self

    def annotate(self, text: str) -> int:
        if self._label_centroids is None or self._fitted_domain != self.domain:
            self.fit()
        assert self._label_centroids is not None
        clip = (text or "").strip()[:2000] or "_"
        with torch.no_grad():
            e = self.span_encoder([clip])[0]
            e = e / e.norm().clamp_min(1e-8)
            sims = torch.matmul(self._label_centroids, e)
        return int(torch.argmax(sims).item())

    def annotate_label(self, text: str) -> str:
        return self.outcomes[self.annotate(text)]
