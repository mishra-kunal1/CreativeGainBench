"""
Causal ablation: with-Critic vs without-Critic Edge-CUE on proposer revisions.

Pre-registered acceptance criterion (document before looking at results):
  Critic is judged causally helpful on an eval set if the paired bootstrap CI
  on mean(Δ) = mean(CUE_with − CUE_without) excludes 0.

This is an engineering extension — not licensed by Lean.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from creativegainbench.eval.mas_agents import (
    DEFAULT_GUIDANCE,
    DOMAIN_GUIDANCE,
    PROPOSER_DRAFT_PROMPT,
    PROPOSER_REVISE_PROMPT,
    _complete,
    _make_client,
)
from creativegainbench.metrics.cue import cue_gate
from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
from creativegainbench.metrics.edge_cue import compute_edge_cue

# Pre-registered acceptance (do not change after seeing results).
ACCEPTANCE = (
    "Critic is judged causally helpful iff the paired bootstrap 95% CI on "
    "mean(CUE_with - CUE_without) excludes 0 on the eval set."
)

NO_CRITIC_REVISE = (
    "You are the Proposer. Revise your draft to improve it for the task. "
    "{guidance}\n\n"
    "Task:\n{prompt}\n\nYour draft:\n{draft}\n\n"
    "Write the full revised response now (not a diff)."
)


def paired_bootstrap_ci(
    deltas: list[float], *, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(deltas) / n, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS Critic causal ablation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--cue-provider", default="ollama")
    parser.add_argument("--cue-model", default="gemma2:2b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    print(f"ACCEPTANCE: {ACCEPTANCE}")

    client = _make_client(args.provider, args.base_url)
    receiver = CUEBeliefReceiver(
        CUEBeliefConfig(provider=args.cue_provider, model=args.cue_model)
    )

    records = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    rows = []
    for rec in records:
        prompt = rec["prompt"]
        domain = rec.get("domain")
        guidance = DOMAIN_GUIDANCE.get(domain, DEFAULT_GUIDANCE)
        draft = _complete(
            client,
            args.model,
            PROPOSER_DRAFT_PROMPT.format(guidance=guidance["proposer"], prompt=prompt),
        )
        # With critic: get a critique then revise (inline minimal critic call)
        critique = _complete(
            client,
            args.model,
            (
                "Critique this draft; name the weakest element.\n\n"
                f"Task:\n{prompt}\n\nDraft:\n{draft}"
            ),
        )
        rev_with = _complete(
            client,
            args.model,
            PROPOSER_REVISE_PROMPT.format(
                guidance=guidance["proposer"],
                prompt=prompt,
                draft=draft,
                critique=critique,
            ),
        )
        rev_without = _complete(
            client,
            args.model,
            NO_CRITIC_REVISE.format(
                guidance=guidance["proposer"], prompt=prompt, draft=draft
            ),
        )
        cue_with, _m1, d1 = compute_edge_cue(draft, rev_with, receiver, prompt)
        cue_without, _m2, d2 = compute_edge_cue(draft, rev_without, receiver, prompt)
        delta = float(cue_with) - float(cue_without)
        rows.append(
            {
                "prompt": prompt,
                "domain": domain,
                "cue_with": float(cue_with),
                "cue_without": float(cue_without),
                "delta": delta,
                "gate_with": float(cue_gate(cue_with)),
                "gate_without": float(cue_gate(cue_without)),
            }
        )
        print(f"delta={delta:.6g} cue_with={cue_with:.6g} cue_without={cue_without:.6g}")

    deltas = [r["delta"] for r in rows]
    mean, lo, hi = paired_bootstrap_ci(deltas)
    accepts = not (lo <= 0.0 <= hi)
    summary = {
        "acceptance_criterion": ACCEPTANCE,
        "n": len(deltas),
        "mean_delta": mean,
        "ci95": [lo, hi],
        "ci_excludes_zero": accepts,
        "critic_judged_helpful": accepts,
    }
    print(json.dumps(summary, indent=2))

    out = args.output
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("data/evaluation") / f"mas_ablation_{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": summary}) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
