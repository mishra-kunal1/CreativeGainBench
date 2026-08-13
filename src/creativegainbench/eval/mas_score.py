"""
Required MAS scoring stage: Edge-CUE + Step-CUE + G_k + HandoffGain.

Call graph: run-mas-agents → run-mas-score → run-benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from creativegainbench.eval.trajectory_eval import (
    _extract_trajectory_texts,
    score_trajectory_for_row,
)
from creativegainbench.metrics.cue import cue_gate
from creativegainbench.metrics.edge_cue import compute_edge_cue, handoff_gain_rate
from creativegainbench.metrics.interaction_gain import (
    G_K_SURFACE,
    compute_interaction_gain,
    mas_outputs_from_row,
)


def _rebuild_edge_chain(
    row: dict,
    *,
    receiver,
    cue_fn_unused: Any = None,
) -> list[dict]:
    """Compute Edge-CUE from transcript if chain missing."""
    del cue_fn_unused
    transcript = row.get("transcript") or []
    prompt = row.get("prompt") or ""
    chain: list[dict] = []
    draft = next((t for t in transcript if t.get("step") == "draft"), None)
    revisions = sorted(
        (t for t in transcript if t.get("step") == "revision"),
        key=lambda t: t.get("round") or 0,
    )
    verifies = sorted(
        (t for t in transcript if t.get("step") == "verify"),
        key=lambda t: t.get("round") or 0,
    )
    prev = draft["content"] if draft else None
    for rev in revisions:
        if prev is None:
            prev = rev["content"]
            continue
        cue_val, _m, diag = compute_edge_cue(prev, rev["content"], receiver, prompt)
        chain.append(
            {
                "edge_id": "proposer_draft_to_revision",
                "from_agent": "proposer",
                "to_agent": "proposer",
                "from_step": "draft",
                "to_step": "revision",
                "cue": float(cue_val),
                "gate": float(cue_gate(cue_val)),
                "brier_delta": diag["brier_delta"],
                "bit_length": diag["bit_length"],
                "round": rev.get("round"),
                "via_agent": "critic",
                "diagnostic": False,
            }
        )
        prev = rev["content"]
    for ver in verifies:
        rev = next(
            (r for r in revisions if r.get("round") == ver.get("round")),
            revisions[-1] if revisions else None,
        )
        if rev is None:
            continue
        cue_val, _m, diag = compute_edge_cue(
            rev["content"], ver["content"], receiver, prompt
        )
        chain.append(
            {
                "edge_id": "proposer_revision_to_verifier",
                "from_agent": "proposer",
                "to_agent": "verifier",
                "from_step": "revision",
                "to_step": "verify",
                "cue": float(cue_val),
                "gate": float(cue_gate(cue_val)),
                "brier_delta": diag["brier_delta"],
                "bit_length": diag["bit_length"],
                "round": ver.get("round"),
                "diagnostic": True,
            }
        )
    return chain


def score_mas_row(
    row: dict,
    *,
    cue_fn: Callable[[str, str], float],
    rb_fn: Callable[[str], float],
    span_encoder=None,
    centroids=None,
    boundary_detector=None,
    boundary_threshold: float = 0.5,
    receiver=None,
) -> dict:
    """Score one MAS JSONL row; returns enriched dict."""
    out = dict(row)
    chain = list(row.get("edge_cue_chain") or [])
    if not chain and receiver is not None and row.get("transcript"):
        chain = _rebuild_edge_chain(row, receiver=receiver)
    out["edge_cue_chain"] = chain
    scored = [e for e in chain if not e.get("diagnostic")]
    out["handoff_gain_rate"] = handoff_gain_rate(scored if scored else chain)

    traj = score_trajectory_for_row(row, cue_fn=cue_fn, rb_fn=rb_fn)
    if traj is not None:
        out["step_cue"] = traj

    mas = None
    if span_encoder is not None and centroids is not None:
        mas = mas_outputs_from_row(
            row,
            span_encoder=span_encoder,
            centroids=centroids,
            boundary_detector=boundary_detector,
            boundary_threshold=boundary_threshold,
        )
    if mas is not None:
        out["g_k"] = float(compute_interaction_gain(mas))
        out["g_k_kind"] = G_K_SURFACE
    elif row.get("g_k") is not None:
        out["g_k"] = float(row["g_k"])
        out["g_k_kind"] = row.get("g_k_kind", G_K_SURFACE)
    else:
        out["g_k"] = 0.0
        out["g_k_kind"] = G_K_SURFACE

    texts = _extract_trajectory_texts(row.get("transcript") or [])
    out["n_proposer_snapshots"] = len(texts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MAS JSONL (Edge-CUE/Step-CUE/G_k)")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cue-provider", choices=["none", "ollama", "openai", "stub"], default="stub")
    parser.add_argument("--cue-model", type=str, default="gemma2:2b")
    parser.add_argument("--receiver", choices=["hash", "ollama", "openai"], default="hash")
    parser.add_argument("--receiver-model", type=str, default="gemma2:2b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--recompute-edge-cue",
        action="store_true",
        help="Recompute edge chain even if present (requires non-stub CUE)",
    )
    args = parser.parse_args()

    from creativegainbench.eval.benchmark_eval import _build_receiver
    from creativegainbench.ideas.artifacts import load_artifacts, load_config
    from creativegainbench.metrics.cue import stub_positive_cue
    from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
    from creativegainbench.metrics.receiver_expansion import compute_receiver_expansion

    cfg = load_config()
    pipeline = load_artifacts(device=args.device)
    rx = _build_receiver(args.receiver, pipeline, args.receiver_model)
    rx_cfg = cfg["receiver_expansion"]

    cue_receiver = None
    if args.cue_provider in {"ollama", "openai"}:
        cue_receiver = CUEBeliefReceiver(
            CUEBeliefConfig(provider=args.cue_provider, model=args.cue_model)
        )

    def cue_fn(prompt: str, y: str) -> float:
        if cue_receiver is not None:
            val, _m, _d = cue_receiver.compute_cue_for_output(prompt, y)
            return float(val)
        val, _m = stub_positive_cue(y)
        return float(val)

    def rb_fn(y: str) -> float:
        return float(
            compute_receiver_expansion(
                y,
                receiver_agent=rx,
                task_battery=pipeline.task_battery,
                idea_codebook_centroids=pipeline.codebook.centroids,
                n_samples=int(rx_cfg["n_samples"]),
                temperature=float(rx_cfg["temperature"]),
                device=pipeline.device,
            )
        )

    edge_receiver = cue_receiver if args.recompute_edge_cue or args.cue_provider != "stub" else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.results) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.limit is not None and n >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if args.recompute_edge_cue:
                row["edge_cue_chain"] = []
            scored = score_mas_row(
                row,
                cue_fn=cue_fn,
                rb_fn=rb_fn,
                span_encoder=pipeline.span_encoder,
                centroids=pipeline.codebook.centroids,
                boundary_detector=pipeline.boundary_detector,
                boundary_threshold=pipeline.boundary_threshold,
                receiver=edge_receiver,
            )
            fout.write(json.dumps(scored) + "\n")
            n += 1

    print(f"Wrote {n} MAS-scored rows → {args.output}")


if __name__ == "__main__":
    main()
