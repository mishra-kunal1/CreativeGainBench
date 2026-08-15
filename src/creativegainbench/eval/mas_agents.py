"""
Proposer-Critic-Verifier multi-agent generation with Edge-CUE instrumentation.

Fixed 3-role triad: each role solves the task alone first (-> agent_texts),
then roles interact (Proposer drafts -> Critic critiques -> Proposer revises
-> Verifier checks) to produce joint_text.

Pre-registered handoffs (do not redefine post-hoc):
  1. proposer_draft → critic_feedback → proposer_revision
     (Edge-CUE: prior conditioned on draft+critique context via upstream=
     critique? Plan: upstream=draft/critique chain — we score
     upstream=draft, downstream=revision after critique, AND
     upstream=critique text is part of the revision conditioning.
     Registered edge: from draft to revision with critic as intermediary —
     Edge-CUE(upstream=draft, downstream=revision).)
  2. proposer_revision → verifier_decision (diagnostic only; does not gate
     creativity score)
  3. Ablation arm (without critic) lives in run-mas-ablation, not here.

When ``receiver`` is provided, Edge-CUE is computed at each registered handoff
and stored in ``edge_cue_chain`` plus per-step transcript fields.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "gpt-4o"
PROVIDER_DEFAULT_MODEL = {
    "openai": "gpt-4o",
    "gemini": "gemini-3.5-flash",
    "ollama": "gemma2:2b",
}
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10
ROLES = ["proposer", "critic", "verifier"]

# Pre-registered Edge-CUE edges (Goodhart-resistant registry).
REGISTERED_EDGES = (
    "proposer_draft_to_revision",
    "proposer_revision_to_verifier",
)

DEFAULT_GUIDANCE = {
    "proposer": "Produce a substantive, specific response to the task — avoid generic, one-size-fits-all framing.",
    "critic": "Identify the single weakest or most generic part of the draft and explain concretely why it falls short.",
    "verifier": "Check both whether the revision is independently correct/adequate for the task on its own merits, and whether it addresses the critic's specific objections, noting anything still unresolved.",
}

DOMAIN_GUIDANCE = {
    "creative_writing": {
        "proposer": "Aim for a distinctive voice, unexpected imagery, and structural choices that avoid predictable arcs. Avoid clichés and stock phrasing.",
        "critic": "Push back hard on clichés, predictable plot beats, stock metaphors, and generic character voice. Name the single weakest, most derivative element and demand it be replaced with something more original.",
        "verifier": "Check that the revision is internally consistent (no contradictions in tone, plot, or characters introduced by the revision) and that it actually addresses the critic's specific objections rather than superficially rephrasing them.",
    },
    "scientific_proposal": {
        "proposer": "Propose a specific, non-obvious mechanism or method — not a restatement of the problem. Be concrete about what is novel relative to standard approaches.",
        "critic": "Challenge the proposal's novelty (is this actually different from existing approaches?) and feasibility (is there a concrete, executable path, or is this hand-waving?). Identify the weakest claim.",
        "verifier": "Check that the revision responds to both the novelty and feasibility objections with specifics, not just reassurances, and that no new unsupported claims were introduced.",
    },
    "mathematical_proof": {
        "proposer": "Produce a complete proof attempt with explicit, numbered steps and a stated justification (axiom/lemma/inference rule) for each step.",
        "critic": "Check each step of the proof for logical validity: does it follow from the previous steps and its stated justification? Flag any step asserted without justification, any invalid inference, or any gap. This stands in for a future automated Lean 4 type-checker — be as rigorous as a human proof-checker would be.",
        "verifier": "Re-check the revised proof step-by-step for logical validity, the same way the critic did. Explicitly confirm each step follows validly, or point out exactly which step still fails. This stands in for a future hard Lean 4 oracle — err on the side of rejecting steps with unresolved gaps rather than approving optimistically.",
    },
}

SOLO_PROMPT = (
    "Solve the following task to the best of your ability. Provide a "
    "complete, self-contained response.\n\nTask:\n{prompt}"
)

PROPOSER_DRAFT_PROMPT = (
    "You are the Proposer in a three-agent team solving a task. {guidance}\n\n"
    "Task:\n{prompt}\n\nWrite your draft response now."
)

CRITIC_PROMPT = (
    "You are the Critic in a three-agent team. Your job is to find real "
    "weaknesses in the Proposer's draft, not to rubber-stamp it. {guidance}\n\n"
    "Task:\n{prompt}\n\nProposer's draft:\n{draft}\n\n"
    "{verifier_feedback_section}"
    "Write a specific, actionable critique (concrete points, not generic praise)."
)

PROPOSER_REVISE_PROMPT = (
    "You are the Proposer. Revise your draft to address the Critic's feedback "
    "below. Keep what works; fix what doesn't. {guidance}\n\n"
    "Task:\n{prompt}\n\nYour draft:\n{draft}\n\nCritic's feedback:\n{critique}\n\n"
    "Write the full revised response now (not a diff)."
)

VERIFIER_PROMPT = (
    "You are the Verifier in a three-agent team. Perform two INDEPENDENT checks "
    "on the revision below -- do not conflate them:\n"
    "1. task_valid: judged purely on the revision's own merits against the "
    "original task, as if the Critic had said nothing -- is it actually "
    "correct/adequate? (For a proof: is the full argument logically sound "
    "end-to-end, not just the step the Critic flagged. For other domains: "
    "does it genuinely solve the task.)\n"
    "2. addresses_critique: does the revision resolve the Critic's specific "
    "objections below, not just superficially rephrase them?\n"
    "{guidance}\n\nTask:\n{prompt}\n\nRevision to verify:\n{revision}\n\n"
    "Critic's feedback that prompted this revision:\n{critique}\n\n"
    "Respond with ONLY a JSON object, no other text, in this exact format:\n"
    '{{"task_valid": true or false, "addresses_critique": true or false, '
    '"unresolved_issues": ["<short phrase>", ...], "verdict": "APPROVE" or "REJECT", '
    '"reason": "<one or two sentences>"}}\n'
    'verdict must be "APPROVE" only if both checks above are true.'
)


def _make_client(provider: str, base_url: str | None) -> OpenAI:
    provider = provider.lower()
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for provider=openai")
        return OpenAI(api_key=api_key, base_url=base_url)
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY required for provider=gemini")
        return OpenAI(base_url=base_url or GEMINI_BASE_URL, api_key=api_key)
    if provider == "ollama":
        return OpenAI(
            base_url=base_url or DEFAULT_OLLAMA_BASE,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
    raise ValueError(f"Unknown provider: {provider!r} (use openai|gemini|ollama)")


def _complete(client: OpenAI, model: str, prompt: str, *, temperature: float = 1.0) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=1,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            is_last = attempt == MAX_RETRIES - 1
            if "429" in str(e) and not is_last:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise
    return ""


def _parse_verifier_response(verifier_text: str) -> dict:
    default = {
        "approved": False,
        "task_valid": False,
        "addresses_critique": False,
        "unresolved_issues": [],
        "reason": "",
    }
    if not verifier_text:
        return default
    match = re.search(r"\{.*\}", verifier_text, flags=re.DOTALL)
    if not match:
        return default
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return default
    task_valid = data.get("task_valid") is True
    addresses_critique = data.get("addresses_critique") is True
    verdict = str(data.get("verdict", "")).strip().upper()
    unresolved = data.get("unresolved_issues", [])
    if not isinstance(unresolved, list):
        unresolved = [unresolved]
    return {
        "approved": verdict == "APPROVE" and task_valid and addresses_critique,
        "task_valid": task_valid,
        "addresses_critique": addresses_critique,
        "unresolved_issues": [str(u) for u in unresolved],
        "reason": str(data.get("reason", "")).strip(),
    }


def _verifier_feedback_section(feedback: str) -> str:
    if not feedback:
        return ""
    return (
        "The Verifier rejected the previous revision and flagged this as "
        f"still unresolved -- make sure your critique accounts for it:\n{feedback}\n\n"
    )


def _edge_record(
    *,
    from_agent: str,
    to_agent: str,
    from_step: str,
    to_step: str,
    edge_id: str,
    cue: float,
    gate: float,
    brier_delta: float,
    bit_length: float,
    diagnostic: bool = False,
) -> dict[str, Any]:
    return {
        "edge_id": edge_id,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "from_step": from_step,
        "to_step": to_step,
        "cue": float(cue),
        "gate": float(gate),
        "brier_delta": float(brier_delta),
        "bit_length": float(bit_length),
        "diagnostic": diagnostic,
    }


def run_triad_for_prompt(
    proposer_client: OpenAI,
    critic_client: OpenAI,
    verifier_client: OpenAI,
    prompt: str,
    proposer_model: str,
    critic_model: str,
    verifier_model: str,
    *,
    domain: str | None = None,
    max_revision_rounds: int = 1,
    receiver: Any | None = None,
    outcome_annotator: Any | None = None,
) -> dict:
    guidance = DOMAIN_GUIDANCE.get(domain, DEFAULT_GUIDANCE)
    agent_models = [proposer_model, critic_model, verifier_model]
    agent_clients = [proposer_client, critic_client, verifier_client]
    transcript: list[dict] = []
    edge_cue_chain: list[dict] = []

    agent_texts: list[str] = []
    for role, model, role_client in zip(ROLES, agent_models, agent_clients):
        text = _complete(role_client, model, SOLO_PROMPT.format(prompt=prompt))
        agent_texts.append(text)
        transcript.append(
            {"role": role, "step": "solo", "round": None, "model": model, "content": text}
        )

    draft = _complete(
        proposer_client,
        proposer_model,
        PROPOSER_DRAFT_PROMPT.format(guidance=guidance["proposer"], prompt=prompt),
    )
    transcript.append(
        {"role": "proposer", "step": "draft", "round": None, "model": proposer_model, "content": draft}
    )

    current_text = draft
    approved = False
    round_idx = 0
    verifier_feedback = ""
    while True:
        round_idx += 1
        critique = _complete(
            critic_client,
            critic_model,
            CRITIC_PROMPT.format(
                guidance=guidance["critic"],
                prompt=prompt,
                draft=current_text,
                verifier_feedback_section=_verifier_feedback_section(verifier_feedback),
            ),
        )
        transcript.append(
            {"role": "critic", "step": "critique", "round": round_idx, "model": critic_model, "content": critique}
        )

        pre_revision = current_text
        current_text = _complete(
            proposer_client,
            proposer_model,
            PROPOSER_REVISE_PROMPT.format(
                guidance=guidance["proposer"], prompt=prompt, draft=current_text, critique=critique
            ),
        )
        rev_entry: dict[str, Any] = {
            "role": "proposer",
            "step": "revision",
            "round": round_idx,
            "model": proposer_model,
            "content": current_text,
        }

        if receiver is not None:
            from creativegainbench.metrics.cue import cue_gate
            from creativegainbench.metrics.edge_cue import compute_edge_cue

            z_star = None
            if outcome_annotator is not None:
                z_star = int(outcome_annotator.annotate(current_text))
            # Edge 1: draft/pre-revision → revision after critique
            cue_val, _model, diag = compute_edge_cue(
                pre_revision,
                current_text,
                receiver,
                prompt,
                z_star=z_star,
            )
            edge = _edge_record(
                from_agent="proposer",
                to_agent="proposer",
                from_step="draft" if round_idx == 1 else "revision",
                to_step="revision",
                edge_id="proposer_draft_to_revision",
                cue=cue_val,
                gate=float(cue_gate(cue_val)),
                brier_delta=diag["brier_delta"],
                bit_length=diag["bit_length"],
            )
            edge["round"] = round_idx
            edge["via_agent"] = "critic"
            edge_cue_chain.append(edge)
            rev_entry["edge_cue"] = edge

        transcript.append(rev_entry)

        verdict_text = _complete(
            verifier_client,
            verifier_model,
            VERIFIER_PROMPT.format(
                guidance=guidance["verifier"], prompt=prompt, revision=current_text, critique=critique
            ),
            temperature=0.0,
        )
        verdict = _parse_verifier_response(verdict_text)
        ver_entry: dict[str, Any] = {
            "role": "verifier",
            "step": "verify",
            "round": round_idx,
            "model": verifier_model,
            "content": verdict_text,
            "parsed": verdict,
        }

        if receiver is not None:
            from creativegainbench.metrics.cue import cue_gate
            from creativegainbench.metrics.edge_cue import compute_edge_cue

            z_star = None
            if outcome_annotator is not None:
                z_star = int(outcome_annotator.annotate(verdict_text))
            cue_val, _model, diag = compute_edge_cue(
                current_text,
                verdict_text,
                receiver,
                prompt,
                z_star=z_star,
            )
            edge = _edge_record(
                from_agent="proposer",
                to_agent="verifier",
                from_step="revision",
                to_step="verify",
                edge_id="proposer_revision_to_verifier",
                cue=cue_val,
                gate=float(cue_gate(cue_val)),
                brier_delta=diag["brier_delta"],
                bit_length=diag["bit_length"],
                diagnostic=True,
            )
            edge["round"] = round_idx
            edge_cue_chain.append(edge)
            ver_entry["edge_cue"] = edge

        transcript.append(ver_entry)
        approved = verdict["approved"]

        if approved or round_idx >= max_revision_rounds:
            break
        verifier_feedback = "; ".join(verdict["unresolved_issues"]) or verdict["reason"]

    handoff = None
    if edge_cue_chain:
        from creativegainbench.metrics.edge_cue import handoff_gain_rate

        # HandoffGain uses creativity-relevant edges only (exclude diagnostic verifier).
        scored = [e for e in edge_cue_chain if not e.get("diagnostic")]
        handoff = handoff_gain_rate(scored if scored else edge_cue_chain)

    return {
        "prompt": prompt,
        "domain": domain,
        "agent_models": agent_models,
        "agent_texts": agent_texts,
        "roles": ROLES,
        "joint_text": current_text,
        "transcript": transcript,
        "edge_cue_chain": edge_cue_chain,
        "handoff_gain_rate": handoff,
        "responses": [{"response-0": current_text}],
        "verified": approved,
        "revision_rounds": round_idx,
    }


def run_triad_batch_streaming(
    records: list[dict],
    out_f,
    *,
    proposer_client: OpenAI,
    critic_client: OpenAI,
    verifier_client: OpenAI,
    proposer_model: str,
    critic_model: str,
    verifier_model: str,
    max_revision_rounds: int,
    workers: int,
    receiver: Any | None = None,
    outcome_annotator: Any | None = None,
) -> int:
    workers = max(1, min(workers, len(records)))
    n_written = 0

    def _one(record: dict) -> dict:
        return run_triad_for_prompt(
            proposer_client,
            critic_client,
            verifier_client,
            record["prompt"],
            proposer_model,
            critic_model,
            verifier_model,
            domain=record.get("domain"),
            max_revision_rounds=max_revision_rounds,
            receiver=receiver,
            outcome_annotator=outcome_annotator,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_one, record): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                row = future.result()
            except Exception as e:
                print(f"[warn] triad failed on prompt {record.get('prompt', '')[:60]!r}: {e}")
                continue
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_written += 1
    return n_written


def main() -> None:
    provider_choices = ["openai", "gemini", "ollama"]
    parser = argparse.ArgumentParser(description="Proposer-Critic-Verifier multi-agent generation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--provider", choices=provider_choices, default="openai")
    parser.add_argument("--proposer-provider", choices=provider_choices, default=None)
    parser.add_argument("--critic-provider", choices=provider_choices, default=None)
    parser.add_argument("--verifier-provider", choices=provider_choices, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--proposer-model", type=str, default=None)
    parser.add_argument("--critic-model", type=str, default=None)
    parser.add_argument("--verifier-model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-revision-rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--score-edge-cue",
        action="store_true",
        help="Compute Edge-CUE at registered handoffs during generation",
    )
    parser.add_argument("--cue-provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--cue-model", type=str, default="gemma2:2b")
    args = parser.parse_args()

    proposer_provider = args.proposer_provider or args.provider
    critic_provider = args.critic_provider or args.provider
    verifier_provider = args.verifier_provider or args.provider

    if args.workers is None:
        role_providers = {proposer_provider, critic_provider, verifier_provider}
        args.workers = 2 if "ollama" in role_providers else 8

    def _role_model(explicit: str | None, role_provider: str) -> str:
        if explicit:
            return explicit
        if role_provider == args.provider:
            return args.model
        return PROVIDER_DEFAULT_MODEL[role_provider]

    proposer_model = _role_model(args.proposer_model, proposer_provider)
    critic_model = _role_model(args.critic_model, critic_provider)
    verifier_model = _role_model(args.verifier_model, verifier_provider)

    clients: dict[str, OpenAI] = {}

    def _client_for(provider: str) -> OpenAI:
        if provider not in clients:
            clients[provider] = _make_client(provider, args.base_url)
        return clients[provider]

    proposer_client = _client_for(proposer_provider)
    critic_client = _client_for(critic_provider)
    verifier_client = _client_for(verifier_provider)

    receiver = None
    if args.score_edge_cue:
        from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver

        receiver = CUEBeliefReceiver(
            CUEBeliefConfig(provider=args.cue_provider, model=args.cue_model)
        )

    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]

    print(
        f"Running {len(records)} prompts "
        f"(proposer={proposer_provider}/{proposer_model}, "
        f"critic={critic_provider}/{critic_model}, "
        f"verifier={verifier_provider}/{verifier_model}, "
        f"max_revision_rounds={args.max_revision_rounds}, workers={args.workers}, "
        f"score_edge_cue={args.score_edge_cue})"
    )

    def _safe(m: str) -> str:
        return m.replace("/", "_").replace(":", "_")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"mas_agents_{_safe(proposer_model)}_{_safe(critic_model)}_{_safe(verifier_model)}"
    out_dir = args.output_dir / tag / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.data.stem}.jsonl"

    with open(out_path, "w") as f:
        n_written = run_triad_batch_streaming(
            records,
            f,
            proposer_client=proposer_client,
            critic_client=critic_client,
            verifier_client=verifier_client,
            proposer_model=proposer_model,
            critic_model=critic_model,
            verifier_model=verifier_model,
            max_revision_rounds=args.max_revision_rounds,
            workers=args.workers,
            receiver=receiver,
        )

    print(f"Wrote {n_written} triad items to {out_path}")


if __name__ == "__main__":
    main()
