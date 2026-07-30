"""
Multi-agent inference for G_k.

Runs k agent drafts (same or different Ollama models), then a joint synthesis
pass. Writes JSONL consumable by benchmark_eval --mas.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"


def _client(base_url: str | None) -> OpenAI:
    return OpenAI(
        base_url=base_url or DEFAULT_OLLAMA_BASE,
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )


def _complete(client: OpenAI, model: str, prompt: str, temperature: float = 1.0) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        n=1,
    )
    return completion.choices[0].message.content or ""


def run_mas_for_prompt(
    client: OpenAI,
    prompt: str,
    agent_models: list[str],
    joint_model: str,
    *,
    domain: str | None = None,
) -> dict:
    agent_texts: list[str] = []
    for i, model in enumerate(agent_models):
        agent_prompt = (
            f"You are agent {i+1}/{len(agent_models)}. Produce a distinct "
            f"approach to the task (do not copy a generic template).\n\nTask:\n{prompt}"
        )
        agent_texts.append(_complete(client, model, agent_prompt))

    drafts = "\n\n".join(
        f"### Agent {i+1} ({agent_models[i]})\n{t}" for i, t in enumerate(agent_texts)
    )
    joint_prompt = (
        "You are the joint synthesizer. Combine the agent drafts into one "
        "stronger answer that preserves complementary ideas and resolves conflicts.\n\n"
        f"Task:\n{prompt}\n\nDrafts:\n{drafts}\n\nWrite the joint answer:"
    )
    joint_text = _complete(client, joint_model, joint_prompt, temperature=0.8)
    row = {
        "prompt": prompt,
        "agent_models": agent_models,
        "agent_texts": agent_texts,
        "joint_model": joint_model,
        "joint_text": joint_text,
        # Primary response field for single-score path compatibility.
        "responses": [{"response-0": joint_text}],
    }
    if domain is not None:
        row["domain"] = domain
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent Ollama inference for G_k")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--agents",
        type=str,
        default="gemma2:2b,gemma2:2b",
        help="Comma-separated Ollama models for agents (k = count)",
    )
    parser.add_argument("--joint-model", type=str, default="gemma2:2b")
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    agent_models = [m.strip() for m in args.agents.split(",") if m.strip()]
    if len(agent_models) < 2:
        raise SystemExit("Need at least 2 agents for G_k (got: %s)" % agent_models)

    client = _client(args.base_url)
    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]

    rows = [
        run_mas_for_prompt(
            client,
            record["prompt"],
            agent_models,
            args.joint_model,
            domain=record.get("domain"),
        )
        for record in records
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "mas_" + "_".join(m.replace(":", "_") for m in agent_models[:3])
    out_dir = args.output_dir / tag / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.data.stem}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} MAS items to {out_path}")


if __name__ == "__main__":
    main()
