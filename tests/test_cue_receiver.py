"""Offline CUE receiver parse/audit tests — no network."""
from __future__ import annotations

from types import SimpleNamespace

from creativegainbench.metrics.cue import brier_delta, brier_delta_signed
from creativegainbench.metrics.cue_receiver import (
    DEFAULT_OUTCOMES,
    BeliefParse,
    CUEBeliefConfig,
    CUEBeliefReceiver,
    extract_json_object,
    message_text,
    strip_json_fences,
)


def test_strip_json_fences_and_extract():
    raw = 'Sure.\n```json\n{"novel_structure": 0.7, "fluent_paraphrase": 0.1, "clear_utility": 0.1, "low_quality": 0.1}\n```\n'
    assert "novel_structure" in strip_json_fences(raw)
    blob = extract_json_object(raw)
    assert blob and '"novel_structure"' in blob


def test_message_text_falls_back_to_reasoning():
    content_msg = SimpleNamespace(content="hello", reasoning=None, reasoning_content=None)
    assert message_text(content_msg) == ("hello", "content")
    reasoning_msg = SimpleNamespace(
        content="  ",
        reasoning_content=None,
        reasoning='{"novel_structure": 1}',
        model_extra={},
    )
    text, src = message_text(reasoning_msg)
    assert src == "reasoning"
    assert "novel_structure" in text
    empty = SimpleNamespace(content="", reasoning=None, reasoning_content=None, model_extra={})
    assert message_text(empty)[1] == "empty"


def test_parse_probs_ok_and_fail_is_missing_not_uniform():
    rx = CUEBeliefReceiver.__new__(CUEBeliefReceiver)
    rx.cfg = CUEBeliefConfig()
    ok_text = (
        '{"novel_structure": 0.4, "fluent_paraphrase": 0.3, '
        '"clear_utility": 0.2, "low_quality": 0.1}'
    )
    parsed = rx.parse_probs(ok_text)
    assert parsed.ok
    assert parsed.probs is not None
    assert abs(sum(parsed.probs) - 1.0) < 1e-9

    miss = rx.parse_probs("I refuse to assign probabilities.")
    assert miss.ok is False
    assert miss.probs is None
    assert miss.reason == "no_json_object"

    # Legacy list API still uniforms; CUE path must not.
    legacy = rx._parse_probs("not json")
    assert legacy == [0.25, 0.25, 0.25, 0.25]


def test_compute_cue_parse_fail_returns_none_not_zero():
    rx = CUEBeliefReceiver.__new__(CUEBeliefReceiver)
    rx.cfg = CUEBeliefConfig()

    def _fail_chat(_prompt: str) -> tuple[str, str]:
        return "thinking out loud, no json", "content"

    rx._chat = _fail_chat  # type: ignore[method-assign]
    cue, model, diag = rx.compute_cue_for_output("task", "y text", external_outcome_index=0)
    assert cue is None
    assert model is None
    assert diag["parse_ok"] is False
    assert diag["prior"] is None
    assert diag["posterior"] is None
    assert diag["cue_missing_reason"] == "parse_fail"
    assert diag["brier_delta"] is None


def test_compute_cue_success_persists_vectors_and_signed_brier():
    rx = CUEBeliefReceiver.__new__(CUEBeliefReceiver)
    rx.cfg = CUEBeliefConfig()
    prior = (
        '{"novel_structure": 0.1, "fluent_paraphrase": 0.1, '
        '"clear_utility": 0.1, "low_quality": 0.7}'
    )
    posterior = (
        '{"novel_structure": 0.7, "fluent_paraphrase": 0.1, '
        '"clear_utility": 0.1, "low_quality": 0.1}'
    )
    calls = {"n": 0}

    def _chat(_prompt: str) -> tuple[str, str]:
        calls["n"] += 1
        return (prior if calls["n"] == 1 else posterior), "content"

    rx._chat = _chat  # type: ignore[method-assign]
    cue, model, diag = rx.compute_cue_for_output("task", "y text", external_outcome_index=0)
    assert cue is not None and cue > 0
    assert model is not None
    assert diag["parse_ok"] is True
    assert diag["prior"] is not None
    assert diag["posterior"] is not None
    signed = brier_delta_signed(diag["prior"], diag["posterior"], 0)
    clipped = brier_delta(diag["prior"], diag["posterior"], 0)
    assert diag["brier_delta_signed"] == signed
    assert diag["brier_delta"] == clipped
    assert signed == clipped  # this update is nonnegative


def test_brier_delta_signed_can_be_negative():
    prior = [0.1, 0.1, 0.1, 0.7]
    posterior = [0.7, 0.1, 0.1, 0.1]
    # Realized outcome is low_quality (index 3): posterior is worse.
    signed = brier_delta_signed(prior, posterior, 3)
    assert signed < 0
    assert brier_delta(prior, posterior, 3) == 0.0


def test_cached_invalid_prior_is_parse_fail():
    rx = CUEBeliefReceiver.__new__(CUEBeliefReceiver)
    rx.cfg = CUEBeliefConfig()
    rx._chat = lambda _p: (  # type: ignore[method-assign]
        '{"novel_structure": 0.25, "fluent_paraphrase": 0.25, '
        '"clear_utility": 0.25, "low_quality": 0.25}',
        "content",
    )
    cue, _model, diag = rx.compute_cue_for_output(
        "task",
        "y",
        external_outcome_index=0,
        prior=[0.0, 0.0, 0.0, 0.0],
    )
    assert cue is None
    assert diag["parse_ok_prior"] is False
    assert isinstance(diag.get("parse_ok_posterior"), bool)


def test_belief_parse_dataclass():
    p = BeliefParse(False, None, "preview", "content", reason="no_json_object")
    assert p.ok is False
    assert len(DEFAULT_OUTCOMES) == 4
