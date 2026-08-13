"""
Evaluate inference JSONL with the canonical R_creativity score.

Example (Ollama, calibrated CUE, Ollama receiver):
  run-benchmark \\
    --results data/results/gemma2_2b/<ts>/eval_all_domains.jsonl \\
    --cue-provider ollama --cue-model gemma2:2b \\
    --receiver ollama --receiver-model gemma2:2b \\
    --output data/evaluation/r_creativity_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creativegainbench.benchmark_score import compute_r_creativity
from creativegainbench.ideas.artifacts import load_artifacts, load_config
from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
from creativegainbench.metrics.delta_d import load_delta_d_thresholds, resolve_delta_d
from creativegainbench.metrics.interaction_gain import mas_outputs_from_row
from creativegainbench.receivers.hash_receiver import HashReceiverAgent
from creativegainbench.utils.contamination import (
    assert_no_probe_overlap,
    load_probe_hashes,
)


def _extract_response(item: dict, response_key: str) -> str | None:
    responses = item.get("responses")
    if isinstance(responses, list):
        for entry in responses:
            if isinstance(entry, dict) and response_key in entry:
                return entry[response_key]
        if responses and isinstance(responses[0], dict) and responses[0]:
            return next(iter(responses[0].values()))
    if item.get("joint_text"):
        return item["joint_text"]
    if response_key in item:
        return item[response_key]
    if "response" in item:
        return item["response"]
    return None


def _load_thresholds(artifacts_root: Path, version: str) -> dict:
    versioned = artifacts_root / f"delta_d_thresholds_{version}.json"
    stable = artifacts_root / "delta_d_thresholds.json"
    path = versioned if versioned.exists() else stable
    return load_delta_d_thresholds(path)


def _build_receiver(name: str, pipeline, model: str):
    if name == "openai":
        from creativegainbench.receivers.openai_receiver import OpenAIReceiverAgent

        return OpenAIReceiverAgent(
            span_encoder=pipeline.span_encoder,
            boundary_detector=pipeline.boundary_detector,
            boundary_threshold=pipeline.boundary_threshold,
        )
    if name == "ollama":
        from creativegainbench.receivers.ollama_receiver import OllamaReceiverAgent

        return OllamaReceiverAgent(
            span_encoder=pipeline.span_encoder,
            boundary_detector=pipeline.boundary_detector,
            boundary_threshold=pipeline.boundary_threshold,
            model=model,
        )
    return HashReceiverAgent(
        span_encoder=pipeline.span_encoder,
        boundary_detector=pipeline.boundary_detector,
        boundary_threshold=pipeline.boundary_threshold,
        seed=pipeline.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score outputs with R_creativity")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/r_creativity_scores.jsonl"),
    )
    parser.add_argument("--artifacts", type=str, default=None)
    parser.add_argument("--response-key", type=str, default="response-0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--stub-cue",
        action="store_true",
        help="Legacy stub CUE (discouraged; prefer --cue-provider)",
    )
    parser.add_argument(
        "--cue-provider",
        choices=["none", "ollama", "openai"],
        default="ollama",
        help="Calibrated CUE belief elicitation backend (default: ollama)",
    )
    parser.add_argument("--cue-model", type=str, default="gemma2:2b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--receiver",
        choices=["hash", "openai", "ollama"],
        default="ollama",
        help="Receiver agent backend for R_B^{→A}",
    )
    parser.add_argument("--receiver-model", type=str, default="gemma2:2b")
    parser.add_argument(
        "--skip-probe-check",
        action="store_true",
        help="Disable eval-vs-probe contamination check (not recommended)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Also write a .summary.json aggregate report",
    )
    args = parser.parse_args()

    cfg = load_config()
    version = args.artifacts or cfg["artifacts"]["version"]
    from creativegainbench.ideas.artifacts import ARTIFACTS_ROOT

    pipeline = load_artifacts(version=version, device=args.device)
    thresholds = _load_thresholds(ARTIFACTS_ROOT, version)
    receiver = _build_receiver(args.receiver, pipeline, args.receiver_model)

    cue_receiver: CUEBeliefReceiver | None = None
    if args.cue_provider != "none" and not args.stub_cue:
        cue_receiver = CUEBeliefReceiver(
            CUEBeliefConfig(provider=args.cue_provider, model=args.cue_model)
        )

    if not args.skip_probe_check:
        probes_path = (
            ARTIFACTS_ROOT
            / "probes"
            / f"probes_{version}_seed{pipeline.seed}.json"
        )
        probe_hashes = load_probe_hashes(probes_path)
        prompts = []
        with open(args.results) as f:
            for line in f:
                item = json.loads(line)
                if item.get("prompt"):
                    prompts.append(item["prompt"])
        assert_no_probe_overlap(prompts, probe_hashes)

    score_cfg = cfg["score"]
    rx_cfg = cfg["receiver_expansion"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.results) as fin, open(args.output, "w") as fout:
        for line_i, line in enumerate(fin):
            if args.limit is not None and n_written >= args.limit:
                break
            item = json.loads(line)
            y = _extract_response(item, args.response_key)
            if y is None:
                continue

            cue_model = None
            cue_diag = None
            use_stub = args.stub_cue
            if cue_receiver is not None:
                prompt = item.get("prompt") or ""
                _cue_val, cue_model, cue_diag = cue_receiver.compute_cue_for_output(
                    prompt, y
                )
                use_stub = False

            mas = mas_outputs_from_row(
                item,
                span_encoder=pipeline.span_encoder,
                centroids=pipeline.codebook.centroids,
                boundary_detector=pipeline.boundary_detector,
                boundary_threshold=pipeline.boundary_threshold,
            )
            configured_lambda = float(score_cfg.get("lambda_g", 0.0))
            if mas is None:
                effective_lambda = 0.0
            elif configured_lambda > 0:
                effective_lambda = configured_lambda
            else:
                effective_lambda = float(score_cfg.get("lambda_g_mas", 1.0))

            domain = item.get("domain_cluster", item.get("domain"))
            delta_d = resolve_delta_d(thresholds, domain)
            edge_chain = item.get("edge_cue_chain")
            handoff = item.get("handoff_gain_rate")

            result = compute_r_creativity(
                y,
                pipeline=pipeline,
                receiver=receiver,
                cue_model=cue_model,
                mas_outputs=mas,
                alpha=float(score_cfg["alpha"]),
                lambda_g=effective_lambda,
                delta_d=delta_d,
                n_samples=int(rx_cfg["n_samples"]),
                temperature=float(rx_cfg["temperature"]),
                use_stub_cue=use_stub,
                edge_cue_chain=edge_chain,
                handoff_gain_rate=float(handoff) if handoff is not None else None,
            )
            row = {
                "prompt": item.get("prompt"),
                "domain": item.get("domain"),
                "response_key": args.response_key,
                "response": y,
                **result.to_dict(),
            }
            if cue_diag is not None:
                row["cue_diag"] = cue_diag
            if mas is not None:
                row["mas"] = {
                    "n_agents": len(mas.agent_texts),
                    "agent_models": item.get("agent_models"),
                }
            fout.write(json.dumps(row) + "\n")
            n_written += 1
            if (line_i + 1) % 5 == 0:
                print(f"scored {n_written} items...")

    print(f"Wrote {n_written} scores to {args.output}")

    if args.aggregate and n_written > 0:
        from creativegainbench.eval.report import aggregate

        rows = [
            json.loads(l) for l in args.output.read_text().splitlines() if l.strip()
        ]
        summary = aggregate(rows)
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()
