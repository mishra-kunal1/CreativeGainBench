"""
Calibrated CUE via belief elicitation over a finite outcome space Z.

Prior/posterior are probability vectors from an LLM receiver (Ollama/OpenAI).
The realized outcome is a forced classification of y into Z. CUE is then
brier_delta(prior→posterior, outcome) / |y|_bits.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI

from creativegainbench.metrics.cue import (
    CUEModel,
    bit_length_utf8,
    brier_delta,
    compute_cue,
)

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"

# Finite decision-relevant state space Z (Lean CUEModel over Fin |Z|).
DEFAULT_OUTCOMES: tuple[str, ...] = (
    "novel_structure",       # restructures ideas / approach
    "fluent_paraphrase",     # surface novelty only
    "clear_utility",         # useful but not structurally new
    "low_quality",           # incoherent / off-task / unhelpful
)


@dataclass
class CUEBeliefConfig:
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
    model: str = "gemma2:2b"
    provider: str = "ollama"  # ollama | openai
    base_url: str | None = None
    temperature: float = 0.0


class CUEBeliefReceiver:
    """Elicits categorical beliefs for CUE Brier scoring."""

    def __init__(self, cfg: CUEBeliefConfig | None = None):
        self.cfg = cfg or CUEBeliefConfig()
        if self.cfg.provider == "ollama":
            self.client = OpenAI(
                base_url=self.cfg.base_url or DEFAULT_OLLAMA_BASE,
                api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY required for CUE provider=openai")
            self.client = OpenAI(api_key=api_key, base_url=self.cfg.base_url)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return self.cfg.outcomes

    def _chat(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            n=1,
        )
        return completion.choices[0].message.content or ""

    def _parse_probs(self, text: str) -> list[float]:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return self._uniform()
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return self._uniform()
        raw = []
        for name in self.outcomes:
            v = data.get(name, data.get(name.replace("_", " "), 0.0))
            try:
                raw.append(float(v))
            except (TypeError, ValueError):
                raw.append(0.0)
        s = sum(raw)
        if s <= 0:
            return self._uniform()
        return [x / s for x in raw]

    def _uniform(self) -> list[float]:
        k = len(self.outcomes)
        return [1.0 / k] * k

    def elicit_prior(self, task_prompt: str) -> list[float]:
        labels = ", ".join(self.outcomes)
        prompt = (
            "You are a calibrated decision receiver. Before seeing any solution, "
            "assign probabilities to how a typical answer to the task will look.\n"
            f"Labels: {labels}\n"
            "Return ONLY JSON mapping each label to a probability (sum≈1).\n\n"
            f"Task:\n{task_prompt.strip()}\n"
        )
        return self._parse_probs(self._chat(prompt))

    def elicit_prior_conditioned(
        self, task_prompt: str, upstream_text: str
    ) -> list[float]:
        """
        Belief after seeing an upstream agent's text, before the downstream
        output — used for Edge-CUE so the delta is attributable to the handoff.
        """
        labels = ", ".join(self.outcomes)
        up = upstream_text.strip()
        if len(up) > 2500:
            up = up[:2500]
        prompt = (
            "You are a calibrated decision receiver. You have seen an upstream "
            "agent's contribution but not the next agent's output. Assign "
            "probabilities over outcome labels for how the eventual answer "
            "will look given only this upstream text.\n"
            f"Labels: {labels}\n"
            "Return ONLY JSON mapping each label to a probability (sum≈1).\n\n"
            f"Task:\n{task_prompt.strip()}\n\n"
            f"Upstream agent text:\n{up}\n"
        )
        return self._parse_probs(self._chat(prompt))

    def elicit_posterior(self, task_prompt: str, y: str) -> list[float]:
        labels = ", ".join(self.outcomes)
        y_clip = y.strip()
        if len(y_clip) > 2500:
            y_clip = y_clip[:2500]
        prompt = (
            "You are a calibrated decision receiver. After seeing a candidate "
            "solution, update your probabilities over outcome labels.\n"
            f"Labels: {labels}\n"
            "Return ONLY JSON mapping each label to a probability (sum≈1).\n\n"
            f"Task:\n{task_prompt.strip()}\n\n"
            f"Candidate solution:\n{y_clip}\n"
        )
        return self._parse_probs(self._chat(prompt))

    def classify_outcome(self, task_prompt: str, y: str) -> int:
        labels = ", ".join(self.outcomes)
        y_clip = y.strip()
        if len(y_clip) > 2500:
            y_clip = y_clip[:2500]
        prompt = (
            "Classify the candidate solution into exactly one label.\n"
            f"Labels: {labels}\n"
            'Return ONLY JSON like {"label": "<one_label>"}.\n\n'
            f"Task:\n{task_prompt.strip()}\n\n"
            f"Candidate solution:\n{y_clip}\n"
        )
        text = self._chat(prompt)
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        label = ""
        if m:
            try:
                label = str(json.loads(m.group(0)).get("label", "")).strip()
            except json.JSONDecodeError:
                label = ""
        if label in self.outcomes:
            return self.outcomes.index(label)
        # Fuzzy contains
        for i, name in enumerate(self.outcomes):
            if name in text:
                return i
        return len(self.outcomes) - 1  # default low_quality

    def compute_cue_for_output(
        self,
        task_prompt: str,
        y: str,
        *,
        external_outcome_index: int | None = None,
    ) -> tuple[float, CUEModel, dict]:
        """
        CUE via Brier improvement on a realized outcome.

        If ``external_outcome_index`` is provided, that label is used as z*
        (math-faithful: receiver does not grade its own classification).
        Otherwise falls back to self-classification (legacy / circular).
        """
        prior = self.elicit_prior(task_prompt)
        posterior = self.elicit_posterior(task_prompt, y)
        if external_outcome_index is not None:
            outcome = int(external_outcome_index)
            if not (0 <= outcome < len(self.outcomes)):
                raise ValueError(
                    f"external_outcome_index {outcome} out of range for "
                    f"{len(self.outcomes)} outcomes"
                )
            outcome_source = "external"
            z_star_source = "external"
        else:
            outcome = self.classify_outcome(task_prompt, y)
            outcome_source = "self_classify"
            z_star_source = "self"
        bits = max(bit_length_utf8(y), 8.0)
        delta = brier_delta(prior, posterior, outcome)
        model = CUEModel(brier_delta=delta, bit_length=bits)
        cue_val = compute_cue(model)
        diag = {
            "prior": prior,
            "posterior": posterior,
            "outcome_index": outcome,
            "outcome_label": self.outcomes[outcome],
            "outcomes": list(self.outcomes),
            "outcome_source": outcome_source,
            # Canonical validation field (F0.2): "external" | "self"
            "z_star_source": z_star_source,
            "brier_delta": float(delta),
            "bit_length": float(bits),
        }
        return cue_val, model, diag
