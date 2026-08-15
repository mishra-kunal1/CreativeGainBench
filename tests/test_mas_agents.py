"""Proposer-Critic-Verifier orchestration tests. No live network calls."""

from __future__ import annotations

import io
import json

import pytest

from creativegainbench.eval import mas_agents
from creativegainbench.metrics.interaction_gain import (
    compute_interaction_gain,
    mas_outputs_from_row,
)

APPROVE_JSON = json.dumps(
    {"task_valid": True, "addresses_critique": True, "unresolved_issues": [], "verdict": "APPROVE", "reason": "looks good"}
)
REJECT_JSON = json.dumps(
    {
        "task_valid": False,
        "addresses_critique": False,
        "unresolved_issues": ["still generic"],
        "verdict": "REJECT",
        "reason": "not good enough",
    }
)


def _fake_complete_approve_immediately(client, model, prompt, *, temperature=1.0):
    if "Solve the following task" in prompt:
        return f"solo:{model}"
    if "You are the Verifier" in prompt:
        return APPROVE_JSON
    if "Revise your draft" in prompt:
        return "revision text"
    if "You are the Critic" in prompt:
        return "critique text"
    return "draft text"


def test_row_schema_and_role_ordering(monkeypatch):
    monkeypatch.setattr(mas_agents, "_complete", _fake_complete_approve_immediately)
    row = mas_agents.run_triad_for_prompt(
        object(),
        object(),
        object(),
        "do the task",
        "prop-model",
        "crit-model",
        "ver-model",
        domain="creative_writing",
        max_revision_rounds=1,
    )
    assert set(row.keys()) == {
        "prompt",
        "domain",
        "agent_models",
        "agent_texts",
        "roles",
        "joint_text",
        "transcript",
        "edge_cue_chain",
        "handoff_gain_rate",
        "responses",
        "verified",
        "revision_rounds",
    }
    assert row["edge_cue_chain"] == []
    assert row["handoff_gain_rate"] is None
    assert row["agent_models"] == ["prop-model", "crit-model", "ver-model"]
    assert row["roles"] == ["proposer", "critic", "verifier"]
    assert len(row["agent_texts"]) == 3
    assert row["agent_texts"] == ["solo:prop-model", "solo:crit-model", "solo:ver-model"]
    assert row["joint_text"] == "revision text"
    assert row["responses"] == [{"response-0": "revision text"}]
    assert row["verified"] is True
    assert row["revision_rounds"] == 1
    steps = [(t["role"], t["step"]) for t in row["transcript"]]
    assert steps == [
        ("proposer", "solo"),
        ("critic", "solo"),
        ("verifier", "solo"),
        ("proposer", "draft"),
        ("critic", "critique"),
        ("proposer", "revision"),
        ("verifier", "verify"),
    ]


def test_max_revision_rounds_is_a_hard_cap_not_extra_rounds(monkeypatch):
    """Regression for the off-by-one: --max-revision-rounds 1 must run
    exactly 1 round when the verifier keeps rejecting, not 2."""

    def fake(client, model, prompt, *, temperature=1.0):
        if "Solve the following task" in prompt:
            return "solo"
        if "You are the Verifier" in prompt:
            return REJECT_JSON
        if "Revise your draft" in prompt:
            return "revision text"
        if "You are the Critic" in prompt:
            return "critique text"
        return "draft text"

    monkeypatch.setattr(mas_agents, "_complete", fake)
    row = mas_agents.run_triad_for_prompt(
        object(), object(), object(), "p", "m1", "m2", "m3", domain=None, max_revision_rounds=1
    )
    assert row["revision_rounds"] == 1
    verify_steps = [t for t in row["transcript"] if t["step"] == "verify"]
    assert len(verify_steps) == 1


def test_bounded_revision_rounds_when_verifier_never_approves(monkeypatch):
    def fake(client, model, prompt, *, temperature=1.0):
        if "Solve the following task" in prompt:
            return "solo"
        if "You are the Verifier" in prompt:
            return REJECT_JSON
        if "Revise your draft" in prompt:
            return "revision text"
        if "You are the Critic" in prompt:
            return "critique text"
        return "draft text"

    monkeypatch.setattr(mas_agents, "_complete", fake)
    row = mas_agents.run_triad_for_prompt(
        object(), object(), object(), "p", "m1", "m2", "m3", domain=None, max_revision_rounds=2
    )
    assert row["verified"] is False
    assert row["revision_rounds"] == 2  # capped at max_revision_rounds, all rejected
    verify_steps = [t for t in row["transcript"] if t["step"] == "verify"]
    assert len(verify_steps) == 2
    assert row["joint_text"] == "revision text"  # best-effort text still used


def test_verifier_rejection_feedback_flows_into_next_critique(monkeypatch):
    seen_critic_prompts = []

    def fake(client, model, prompt, *, temperature=1.0):
        if "Solve the following task" in prompt:
            return "solo"
        if "You are the Verifier" in prompt:
            return REJECT_JSON
        if "Revise your draft" in prompt:
            return "revision text"
        if "You are the Critic" in prompt:
            seen_critic_prompts.append(prompt)
            return "critique text"
        return "draft text"

    monkeypatch.setattr(mas_agents, "_complete", fake)
    mas_agents.run_triad_for_prompt(
        object(), object(), object(), "p", "m1", "m2", "m3", domain=None, max_revision_rounds=2
    )
    assert len(seen_critic_prompts) == 2
    # Round 1's critic has no prior verifier feedback yet.
    assert "still generic" not in seen_critic_prompts[0]
    # Round 2's critic must see round 1's unresolved_issues from the verifier.
    assert "still generic" in seen_critic_prompts[1]


@pytest.mark.parametrize(
    "domain,expected_snippet",
    [
        ("mathematical_proof", "Lean 4"),
        ("creative_writing", "clich"),
        ("scientific_proposal", "feasibility"),
    ],
)
def test_domain_guidance_reaches_prompts(monkeypatch, domain, expected_snippet):
    seen_prompts = []

    def fake(client, model, prompt, *, temperature=1.0):
        seen_prompts.append(prompt)
        if "You are the Verifier" in prompt:
            return APPROVE_JSON
        return "text"

    monkeypatch.setattr(mas_agents, "_complete", fake)
    mas_agents.run_triad_for_prompt(
        object(), object(), object(), "p", "m1", "m2", "m3", domain=domain, max_revision_rounds=0
    )
    assert any(expected_snippet.lower() in p.lower() for p in seen_prompts)


def test_unknown_domain_falls_back_to_default_guidance(monkeypatch):
    monkeypatch.setattr(mas_agents, "_complete", _fake_complete_approve_immediately)
    row = mas_agents.run_triad_for_prompt(
        object(), object(), object(), "p", "m1", "m2", "m3", domain="mixed", max_revision_rounds=0
    )
    assert row["joint_text"]  # no KeyError, row produced


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"task_valid": True, "addresses_critique": True, "verdict": "APPROVE"}, True),
        ({"task_valid": True, "addresses_critique": True, "verdict": "REJECT"}, False),
        # Exact-match regression: "NOT APPROVED" must not be treated as APPROVE.
        ({"task_valid": True, "addresses_critique": True, "verdict": "NOT APPROVED"}, False),
        # Independent-check regression: verdict says APPROVE but a sub-check
        # is false -> still not approved (defense in depth against the model
        # mislabeling verdict while its own sub-checks disagree).
        ({"task_valid": False, "addresses_critique": True, "verdict": "APPROVE"}, False),
        ({"task_valid": True, "addresses_critique": False, "verdict": "APPROVE"}, False),
    ],
)
def test_parse_verifier_response_verdict(data, expected):
    result = mas_agents._parse_verifier_response(json.dumps(data))
    assert result["approved"] is expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "garbled nonsense with no JSON",
        "VERDICT: APPROVE\nREASON: old plaintext format",
        '{"verdict": "APPROVE"}',  # missing task_valid/addresses_critique -> both default False
    ],
)
def test_parse_verifier_response_malformed_defaults_to_not_approved(text):
    result = mas_agents._parse_verifier_response(text)
    assert result["approved"] is False


def test_parse_verifier_response_extracts_unresolved_issues():
    result = mas_agents._parse_verifier_response(REJECT_JSON)
    assert result["unresolved_issues"] == ["still generic"]
    assert result["reason"] == "not good enough"


def test_roles_can_use_distinct_clients(monkeypatch):
    """Each role's calls must go through its own client (needed when e.g.
    the critic is on a different provider than the proposer/verifier)."""
    proposer_client, critic_client, verifier_client = object(), object(), object()
    seen_clients = []

    def fake(client, model, prompt, *, temperature=1.0):
        seen_clients.append(client)
        if "You are the Verifier" in prompt:
            return APPROVE_JSON
        return "text"

    monkeypatch.setattr(mas_agents, "_complete", fake)
    mas_agents.run_triad_for_prompt(
        proposer_client,
        critic_client,
        verifier_client,
        "p",
        "prop-model",
        "crit-model",
        "ver-model",
        domain="creative_writing",
        max_revision_rounds=0,
    )
    # 3 solo (one per client) + draft(proposer) + critique(critic) + revision(proposer) + verify(verifier)
    assert seen_clients == [
        proposer_client,
        critic_client,
        verifier_client,
        proposer_client,
        critic_client,
        proposer_client,
        verifier_client,
    ]


@pytest.mark.parametrize(
    "provider,base_url_contains",
    [
        ("openai", None),
        ("gemini", "generativelanguage.googleapis.com"),
        ("ollama", "127.0.0.1:11434"),
    ],
)
def test_make_client_gemini_base_url(monkeypatch, provider, base_url_contains):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    client = mas_agents._make_client(provider, None)
    if base_url_contains:
        assert base_url_contains in str(client.base_url)


def test_make_client_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        mas_agents._make_client("gemini", None)


def test_make_client_ollama_needs_no_api_key(monkeypatch):
    """Local ollama must work without any API key env vars set."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = mas_agents._make_client("ollama", None)
    assert "127.0.0.1:11434" in str(client.base_url)


def test_make_client_ollama_respects_base_url_override(monkeypatch):
    client = mas_agents._make_client("ollama", "http://gpu-box:11434/v1")
    assert "gpu-box:11434" in str(client.base_url)


def test_provider_default_model_covers_all_providers():
    assert set(mas_agents.PROVIDER_DEFAULT_MODEL) == {"openai", "gemini", "ollama"}


def test_run_triad_batch_streaming_parallel_workers(monkeypatch):
    monkeypatch.setattr(mas_agents, "_complete", _fake_complete_approve_immediately)
    records = [{"prompt": f"p{i}", "domain": "creative_writing"} for i in range(3)]
    out = io.StringIO()
    n_written = mas_agents.run_triad_batch_streaming(
        records,
        out,
        proposer_client=object(),
        critic_client=object(),
        verifier_client=object(),
        proposer_model="m1",
        critic_model="m2",
        verifier_model="m3",
        max_revision_rounds=1,
        workers=2,
    )
    assert n_written == 3
    rows = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert len(rows) == 3
    for row in rows:
        assert len(row["agent_texts"]) == 3
        assert row["joint_text"]


def test_output_consumable_by_existing_interaction_gain_machinery(monkeypatch):
    """Regression check: new generator output round-trips through the
    UNTOUCHED metrics/interaction_gain.py path with zero changes there."""

    def fake(client, model, prompt, *, temperature=1.0):
        if "Solve the following task" in prompt:
            return "short solo answer"
        if "You are the Verifier" in prompt:
            return APPROVE_JSON
        if "Revise your draft" in prompt:
            return "a substantially longer joint revised answer with many more words than any solo baseline"
        if "You are the Critic" in prompt:
            return "critique"
        return "draft"

    monkeypatch.setattr(mas_agents, "_complete", fake)
    row = mas_agents.run_triad_for_prompt(
        object(), object(), object(), "task", "m1", "m2", "m3", domain="scientific_proposal", max_revision_rounds=0
    )
    mas = mas_outputs_from_row(row)  # no span_encoder/centroids -> length-proxy fallback
    assert mas is not None
    assert mas.agent_texts == row["agent_texts"]
    assert mas.joint_text == row["joint_text"]
    gain = compute_interaction_gain(mas)
    assert isinstance(gain, float)
    assert gain >= 0.0
