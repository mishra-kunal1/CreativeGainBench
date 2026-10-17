"""
Step-CUE curve (gamma) and Diverge-Converge score for mas_agents.py output.

Post-hoc trajectory diagnostic: re-scores CUE and R_B^{->A} at each
candidate-answer snapshot in a Proposer-Critic-Verifier transcript (the
Proposer's initial draft, then each successive revision), fits the Step-CUE
curve to the running CUE values, and checks the R_B sequence for a
diverge-then-converge peak. Does not touch mas_agents.py generation -- reads
an already-generated results file's `transcript` field.

Example:
  run-trajectory-metrics \\
    --results data/results/mas_agents_gpt-4o_gpt-4o_gpt-4o/<ts>/prompts.jsonl \\
    --cue-provider openai --cue-model gpt-4o-mini \\
    --receiver openai --receiver-model gpt-4o-mini \\
    --output data/evaluation/trajectory_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from creativegainbench.eval.benchmark_eval import _build_receiver
from creativegainbench.ideas.artifacts import load_artifacts, load_config
from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
from creativegainbench.metrics.receiver_expansion import compute_receiver_expansion
from creativegainbench.metrics.trajectory import diverge_converge_score, fit_step_cue_curve


def _extract_trajectory_texts(transcript: list[dict]) -> list[str]:
    """
    Trajectory steps = the Proposer's draft, then each successive revision
    in round order. Critique/verify transcript entries are commentary about
    a candidate answer, not candidate answers themselves, so they're not
    scorable trajectory points.
    """
    draft = next((t for t in transcript if t.get("step") == "draft"), None)
    revisions = sorted(
        (t for t in transcript if t.get("step") == "revision"),
        key=lambda t: t.get("round") or 0,
    )
    texts: list[str] = []
    if draft is not None:
        texts.append(draft["content"])
    texts.extend(t["content"] for t in revisions)
    return texts


def score_trajectory_for_row(
    row: dict,
    *,
    cue_fn: Callable[[str, str], float],
    rb_fn: Callable[[str], float],
) -> dict | None:
    """
    Pure-ish per-row scorer: cue_fn/rb_fn are injected so this is testable
    without live API calls. Returns None for rows with no usable transcript.
    """
    prompt = row.get("prompt")
    transcript = row.get("transcript")
    if not prompt or not transcript:
        return None
    texts = _extract_trajectory_texts(transcript)
    if not texts:
        return None

    t_values = list(range(1, len(texts) + 1))
    step_cue = [cue_fn(prompt, text) for text in texts]
    step_rb = [rb_fn(text) for text in texts]

    fit = fit_step_cue_curve(t_values, step_cue)
    dc = diverge_converge_score(step_rb)

    return {
        "prompt": prompt,
        "domain": row.get("domain"),
        "n_steps": len(texts),
        "t_values": t_values,
        "step_cue": step_cue,
        "step_rb": step_rb,
        "step_cue_fit": fit.to_dict(),
        "diverge_converge": dc.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-CUE / Diverge-Converge trajectory diagnostics")
    parser.add_argument(
        "--results", type=Path, required=True, help="mas_agents.py output JSONL (rows need a 'transcript' field)"
    )
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/trajectory_metrics.jsonl"))
    parser.add_argument("--artifacts", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cue-provider", choices=["ollama", "openai"], default="openai")
    parser.add_argument("--cue-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--receiver", choices=["hash", "openai", "ollama"], default="openai")
    parser.add_argument("--receiver-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    version = args.artifacts or cfg["artifacts"]["version"]
    pipeline = load_artifacts(version=version, device=args.device)
    receiver = _build_receiver(args.receiver, pipeline, args.receiver_model)
    cue_receiver = CUEBeliefReceiver(CUEBeliefConfig(provider=args.cue_provider, model=args.cue_model))
    rx_cfg = cfg["receiver_expansion"]

    def cue_fn(prompt: str, y: str) -> float:
        cue_val, _model, _diag = cue_receiver.compute_cue_for_output(prompt, y)
        if cue_val is None:
            raise RuntimeError("CUE elicitation parse failed")
        return cue_val

    def rb_fn(y: str) -> float:
        return compute_receiver_expansion(
            y,
            receiver_agent=receiver,
            task_battery=pipeline.task_battery,
            idea_codebook_centroids=pipeline.codebook.centroids,
            n_samples=int(rx_cfg["n_samples"]),
            temperature=float(rx_cfg["temperature"]),
            device=pipeline.device,
        )

    with open(args.results) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.output, "w") as out_f:
        for row in rows:
            result = score_trajectory_for_row(row, cue_fn=cue_fn, rb_fn=rb_fn)
            if result is None:
                continue
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_written += 1
            gamma = result["step_cue_fit"]["gamma"]
            dc = result["diverge_converge"]["dc"]
            print(
                f"[{n_written}] {row.get('prompt', '')[:60]!r}: "
                f"{result['n_steps']} steps, gamma={gamma}, DC={dc}"
            )

    print(f"Wrote {n_written} trajectory diagnostics to {args.output}")


if __name__ == "__main__":
    main()
