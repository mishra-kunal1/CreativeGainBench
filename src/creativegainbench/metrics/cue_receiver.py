"""
Calibrated CUE via belief elicitation over a finite outcome space Z.

Prior/posterior are probability vectors from an LLM receiver (Ollama/OpenAI).
The realized outcome is a forced classification of y into Z. CUE is then
brier_delta(prior→posterior, outcome) / |y|_bits.

Parse failures are missing (not silent uniform) on the CUE scoring path.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from creativegainbench.metrics.cue import (
    CUEModel,
    bit_length_utf8,
    brier_delta,
    brier_delta_signed,
    compute_cue,
)

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
DEFAULT_MAX_TOKENS = 1024
RAW_PREVIEW_CHARS = 400

# Finite decision-relevant state space Z (Lean CUEModel over Fin |Z|).
DEFAULT_OUTCOMES: tuple[str, ...] = (
    "novel_structure",       # restructures ideas / approach
    "fluent_paraphrase",     # surface novelty only
    "clear_utility",         # useful but not structurally new
    "low_quality",           # incoherent / off-task / unhelpful
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class CUEBeliefConfig:
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
    model: str = "gemma2:2b"
    provider: str = "ollama"  # ollama | openai
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class BeliefParse:
    """Result of turning a chat string into a probability vector."""

    ok: bool
    probs: list[float] | None
    raw_preview: str
    text_source: str  # content | reasoning | empty
    reason: str = ""


def strip_json_fences(text: str) -> str:
    t = (text or "").strip()
    m = _FENCE_RE.search(t)
    if m:
        return m.group(1).strip()
    return t


def extract_json_object(text: str) -> str | None:
    t = strip_json_fences(text)
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return m.group(0) if m else None


def message_text(message: Any) -> tuple[str, str]:
    """Prefer visible content; fall back to reasoning channels (gpt-oss)."""
    content = (getattr(message, "content", None) or "") if message is not None else ""
    if str(content).strip():
        return str(content), "content"
    for attr in ("reasoning_content", "reasoning"):
        extra = getattr(message, attr, None) if message is not None else None
        if extra and str(extra).strip():
            return str(extra), "reasoning"
    if message is not None:
        model_extra = getattr(message, "model_extra", None) or {}
        if isinstance(model_extra, dict):
            for key in ("reasoning_content", "reasoning"):
                extra = model_extra.get(key)
                if extra and str(extra).strip():
                    return str(extra), "reasoning"
    return "", "empty"


class CUEBeliefReceiver:
    """Elicits categorical beliefs for CUE Brier scoring."""

    def __init__(self, cfg: CUEBeliefConfig | None = None):
        load_dotenv()
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

    def _chat(self, prompt: str) -> tuple[str, str]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
            "n": 1,
        }
        if self.cfg.max_tokens:
            kwargs["max_tokens"] = int(self.cfg.max_tokens)
        completion = self.client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        return message_text(message)

    def parse_probs(self, text: str, *, text_source: str = "content") -> BeliefParse:
        preview = (text or "")[:RAW_PREVIEW_CHARS]
        blob = extract_json_object(text or "")
        if not blob:
            return BeliefParse(
                False, None, preview, text_source, reason="no_json_object"
            )
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return BeliefParse(
                False, None, preview, text_source, reason="json_decode"
            )
        if not isinstance(data, dict):
            return BeliefParse(
                False, None, preview, text_source, reason="not_object"
            )
        raw = []
        for name in self.outcomes:
            v = data.get(name, data.get(name.replace("_", " ")))
            if v is None:
                return BeliefParse(
                    False, None, preview, text_source, reason=f"missing_{name}"
                )
            try:
                raw.append(float(v))
            except (TypeError, ValueError):
                return BeliefParse(
                    False, None, preview, text_source, reason=f"bad_{name}"
                )
        s = sum(raw)
        if s <= 0:
            return BeliefParse(
                False, None, preview, text_source, reason="nonpositive_sum"
            )
        return BeliefParse(True, [x / s for x in raw], preview, text_source, reason="ok")

    def _uniform(self) -> list[float]:
        k = len(self.outcomes)
        return [1.0 / k] * k

    def _parse_probs(self, text: str) -> list[float]:
        """Legacy: uniform on parse miss (Edge-CUE / older callers)."""
        parsed = self.parse_probs(text)
        return list(parsed.probs) if parsed.ok and parsed.probs else self._uniform()

    def _belief_prompt(self, task_prompt: str, y: str | None, *, conditioned: str | None = None) -> str:
        labels = ", ".join(self.outcomes)
        if conditioned is not None:
            up = conditioned.strip()
            if len(up) > 2500:
                up = up[:2500]
            return (
                "You are a calibrated decision receiver. You have seen an upstream "
                "agent's contribution but not the next agent's output. Assign "
                "probabilities over outcome labels for how the eventual answer "
                "will look given only this upstream text.\n"
                f"Labels: {labels}\n"
                "Return ONLY JSON mapping each label to a probability (sum≈1).\n\n"
                f"Task:\n{task_prompt.strip()}\n\n"
                f"Upstream agent text:\n{up}\n"
            )
        if y is None:
            return (
                "You are a calibrated decision receiver. Before seeing any solution, "
                "assign probabilities to how a typical answer to the task will look.\n"
                f"Labels: {labels}\n"
                "Return ONLY JSON mapping each label to a probability (sum≈1).\n\n"
                f"Task:\n{task_prompt.strip()}\n"
            )
        y_clip = y.strip()
        if len(y_clip) > 2500:
            y_clip = y_clip[:2500]
        return (
            "You are a calibrated decision receiver. After seeing a candidate "
            "solution, update your probabilities over outcome labels.\n"
            f"Labels: {labels}\n"
            "Return ONLY JSON mapping each label to a probability (sum≈1).\n\n"
            f"Task:\n{task_prompt.strip()}\n\n"
            f"Candidate solution:\n{y_clip}\n"
        )

    def elicit_prior_detailed(self, task_prompt: str) -> BeliefParse:
        text, source = self._chat(self._belief_prompt(task_prompt, None))
        return self.parse_probs(text, text_source=source)

    def elicit_posterior_detailed(self, task_prompt: str, y: str) -> BeliefParse:
        text, source = self._chat(self._belief_prompt(task_prompt, y))
        return self.parse_probs(text, text_source=source)

    def elicit_prior(self, task_prompt: str) -> list[float]:
        parsed = self.elicit_prior_detailed(task_prompt)
        return list(parsed.probs) if parsed.ok and parsed.probs else self._uniform()

    def elicit_prior_conditioned(
        self, task_prompt: str, upstream_text: str
    ) -> list[float]:
        text, source = self._chat(
            self._belief_prompt(task_prompt, None, conditioned=upstream_text)
        )
        parsed = self.parse_probs(text, text_source=source)
        return list(parsed.probs) if parsed.ok and parsed.probs else self._uniform()

    def elicit_posterior(self, task_prompt: str, y: str) -> list[float]:
        parsed = self.elicit_posterior_detailed(task_prompt, y)
        return list(parsed.probs) if parsed.ok and parsed.probs else self._uniform()

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
        text, _source = self._chat(prompt)
        blob = extract_json_object(text)
        label = ""
        if blob:
            try:
                label = str(json.loads(blob).get("label", "")).strip()
            except json.JSONDecodeError:
                label = ""
        if label in self.outcomes:
            return self.outcomes.index(label)
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
        prior: Sequence[float] | None = None,
        prior_parse: BeliefParse | None = None,
    ) -> tuple[float | None, CUEModel | None, dict]:
        """
        CUE via Brier improvement on a realized outcome.

        Parse failures return cue=None (not CUE=0 from a silent uniform).
        """
        prior_cached = prior is not None or prior_parse is not None
        if prior_parse is not None:
            p_parse = prior_parse
        elif prior is None:
            p_parse = self.elicit_prior_detailed(task_prompt)
        else:
            prior_vec = [float(x) for x in prior]
            if len(prior_vec) != len(self.outcomes) or sum(prior_vec) <= 0:
                p_parse = BeliefParse(
                    False, None, "", "cache", reason="invalid_cached_prior"
                )
            else:
                s = sum(prior_vec)
                p_parse = BeliefParse(
                    True,
                    [x / s for x in prior_vec],
                    "",
                    "cache",
                    reason="ok",
                )
        q_parse = self.elicit_posterior_detailed(task_prompt, y)
        parse_ok = bool(p_parse.ok and q_parse.ok and p_parse.probs and q_parse.probs)
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
        diag: dict[str, Any] = {
            "prior": p_parse.probs,
            "posterior": q_parse.probs,
            "parse_ok": parse_ok,
            "parse_ok_prior": p_parse.ok,
            "parse_ok_posterior": q_parse.ok,
            "parse_reason_prior": p_parse.reason,
            "parse_reason_posterior": q_parse.reason,
            "raw_preview_prior": p_parse.raw_preview,
            "raw_preview_posterior": q_parse.raw_preview,
            "text_source_prior": p_parse.text_source,
            "text_source_posterior": q_parse.text_source,
            "outcome_index": outcome,
            "outcome_label": self.outcomes[outcome],
            "outcomes": list(self.outcomes),
            "outcome_source": outcome_source,
            "z_star_source": z_star_source,
            "bit_length": float(bits),
            "prior_cached": prior_cached,
            "brier_delta": None,
            "brier_delta_signed": None,
        }
        if not parse_ok:
            diag["cue_missing_reason"] = "parse_fail"
            return None, None, diag
        prior_vec = list(p_parse.probs or [])
        posterior = list(q_parse.probs or [])
        signed = brier_delta_signed(prior_vec, posterior, outcome)
        delta = brier_delta(prior_vec, posterior, outcome)
        model = CUEModel(brier_delta=delta, bit_length=bits)
        cue_val = compute_cue(model)
        diag["brier_delta"] = float(delta)
        diag["brier_delta_signed"] = float(signed)
        return cue_val, model, diag
