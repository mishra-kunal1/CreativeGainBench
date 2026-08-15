"""
External realized-outcome annotator for CUE (z*).

Independent of the belief-elicitation receiver so Brier scoring is not
circular. Uses MiniLM nearest-exemplar over a small frozen label bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


@dataclass
class OutcomeAnnotator:
    """Nearest-exemplar classifier over DEFAULT_OUTCOMES."""

    span_encoder: nn.Module
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
    _label_centroids: torch.Tensor | None = None

    def fit(self) -> "OutcomeAnnotator":
        cents = []
        with torch.no_grad():
            for name in self.outcomes:
                texts = EXEMPLARS.get(name, [name])
                emb = self.span_encoder(texts)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                cents.append(emb.mean(dim=0))
        self._label_centroids = torch.stack(cents, dim=0)
        self._label_centroids = (
            self._label_centroids
            / self._label_centroids.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        return self

    def annotate(self, text: str) -> int:
        if self._label_centroids is None:
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
