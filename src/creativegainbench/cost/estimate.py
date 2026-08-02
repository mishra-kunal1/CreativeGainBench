"""CLI: estimate generation + judge costs across providers."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from creativegainbench.cost.models_catalog import (
    DEFAULT_JUDGE_MODELS,
    DOMAIN_SUBSET_FILES,
    FULL_SUBSET_SIZES,
    GENERATION_MODELS,
    CatalogModel,
)
from creativegainbench.cost.ollama_quota import quota_guidance
from creativegainbench.cost.pricing import cost_usd, fetch_openrouter_pricing
from creativegainbench.cost.report_md import render_markdown
from creativegainbench.cost.tokenize import estimate_tokens
from creativegainbench.prompts import extract_prompt
from creativegainbench.providers import ollama_cloud

load_dotenv()

DATA_DIR = Path("data")
SUBSET_DIR = DATA_DIR / "subset"
DEFAULT_EVAL_DIR = DATA_DIR / "evaluation"

# Keep in sync with creativegainbench.eval.llm_as_judge.JUDGE_PROMPT
# (imported separately to avoid requiring OPENROUTER_API_KEY at import time).
JUDGE_PROMPT = """You are an expert judge evaluating the creativity of an AI response to a prompt.

Prompt:
{prompt}

Response:
{response}

Rate the response on each of these criteria, using an integer score from 1 (very poor) to 10 (excellent):
- novelty: how original and non-generic the idea/content is
- surprise: how unexpected the response is compared to a typical/predictable answer
- usefulness: how practically valuable or relevant the response is to the prompt
- coherence: how well-formed, clear, and internally consistent the response is

Respond with ONLY a JSON object, no other text, in this exact format:
{{"novelty": <int>, "surprise": <int>, "usefulness": <int>, "coherence": <int>}}
"""


def _load_prompts(domain: str, sample: int, seed: int) -> list[str]:
    filename = DOMAIN_SUBSET_FILES[domain]
    path = SUBSET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `download-datasets` then `create-subset` first."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    prompts = [extract_prompt(domain, row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:sample]


def _model_available_on_ollama(model_id: str, cloud_names: set[str] | None) -> bool:
    if cloud_names is None:
        return False
    if model_id in cloud_names:
        return True
    base = model_id.split(":", 1)[0]
    return base in cloud_names


def _judge_input_tokens(prompt: str, assumed_response_chars: int) -> int:
    # Approximate a response of assumed_completion_tokens * 4 chars for judge input size.
    fake_response = "x" * max(assumed_response_chars, 1)
    text = JUDGE_PROMPT.format(prompt=prompt, response=fake_response)
    return estimate_tokens(text)


def _estimate_for_model(
    model: CatalogModel,
    *,
    prompts: list[str],
    n: int,
    assumed_completion_tokens: int,
    assumed_judge_completion_tokens: int,
    judge_models: list[str],
    pricing,
    cloud_names: set[str] | None,
) -> dict[str, Any]:
    input_tokens_list = [estimate_tokens(p) for p in prompts]
    total_input = sum(input_tokens_list)
    gen_calls = len(prompts) * n

    # Judge: one call per (prompt, response, judge_model)
    judge_in_total = 0
    for prompt in prompts:
        per = _judge_input_tokens(prompt, assumed_completion_tokens * 4)
        judge_in_total += per * n * len(judge_models)
    judge_out_total = (
        len(prompts) * n * len(judge_models) * assumed_judge_completion_tokens
    )

    gen_usd: float | None = None
    judge_usd = 0.0
    notes: list[str] = []
    available = True
    cross_quote_usd: float | None = None

    # Judge cost always via OpenRouter judge model prices
    for jid in judge_models:
        jp = pricing.get(jid)
        if jp is None:
            notes.append(f"judge price missing for {jid}")
            continue
        # Split judge tokens evenly across judges for pricing (each judge sees full traffic)
        # Actually each judge gets the full judge_in for its share:
        share_in = judge_in_total / max(len(judge_models), 1)
        share_out = judge_out_total / max(len(judge_models), 1)
        judge_usd += cost_usd(jp, input_tokens=share_in, output_tokens=share_out)

    if model.provider == "openrouter":
        price = pricing.get(model.model_id)
        if price is None:
            available = False
            notes.append(f"OpenRouter price missing for {model.model_id}")
        else:
            # One chat request per prompt with n completions (input billed once).
            gen_usd = cost_usd(
                price,
                input_tokens=total_input,
                output_tokens=gen_calls * assumed_completion_tokens,
            )
    else:
        # Ollama Cloud
        available = _model_available_on_ollama(model.model_id, cloud_names)
        if cloud_names is None:
            notes.append("Ollama Cloud tags unavailable (set OLLAMA_API_KEY or network)")
            available = False
        elif not available:
            notes.append(f"not listed on Ollama Cloud tags as `{model.model_id}`")

        notes.append("Ollama Cloud: subscription/GPU-time quota (not $/token)")
        if model.openrouter_cross_quote:
            cq = pricing.get(model.openrouter_cross_quote)
            if cq is None:
                notes.append(f"cross-quote missing on OpenRouter: {model.openrouter_cross_quote}")
            else:
                cross_quote_usd = cost_usd(
                    cq,
                    input_tokens=total_input,
                    output_tokens=gen_calls * assumed_completion_tokens,
                )
                gen_usd = cross_quote_usd
                notes.append(
                    f"USD column = OpenRouter cross-quote `{model.openrouter_cross_quote}`"
                )

    total_usd = None
    if gen_usd is not None:
        total_usd = gen_usd + judge_usd
    elif judge_usd:
        total_usd = judge_usd

    return {
        "family": model.family,
        "display_name": model.display_name,
        "provider": model.provider,
        "model_id": model.model_id,
        "openrouter_cross_quote": model.openrouter_cross_quote,
        "available": available,
        "gen_usd": gen_usd,
        "judge_usd": judge_usd,
        "total_usd": total_usd,
        "gen_calls": gen_calls,
        "notes": "; ".join(notes),
    }


def build_report(
    *,
    domains: list[str],
    sample: int,
    n: int,
    assumed_completion_tokens: int,
    assumed_judge_completion_tokens: int,
    seed: int,
    judge_models: list[str] | None = None,
) -> dict[str, Any]:
    judges = judge_models or list(DEFAULT_JUDGE_MODELS)
    pricing = fetch_openrouter_pricing()

    cloud_names: set[str] | None
    ollama_fetched_at: str | None
    try:
        # Tags endpoint is often public; still try with key when present.
        key = os.environ.get("OLLAMA_API_KEY", "").strip() or None
        if key:
            cloud_names = ollama_cloud.cloud_model_names(api_key=key)
        else:
            # Public tags without auth
            cloud_names = ollama_cloud.cloud_model_names(api_key="")
        ollama_fetched_at = datetime.now(timezone.utc).isoformat()
    except Exception as exc:  # noqa: BLE001 — surface in report, continue
        cloud_names = None
        ollama_fetched_at = None
        ollama_error = str(exc)
    else:
        ollama_error = None

    domain_blocks: list[dict[str, Any]] = []
    ollama_notes: list[str] = []
    family_acc: dict[tuple[str, str], dict[str, Any]] = {}

    sample_gen = 0.0
    sample_judge = 0.0
    projected_gen = 0.0
    projected_judge = 0.0

    for domain in domains:
        prompts = _load_prompts(domain, sample=sample, seed=seed)
        if not prompts:
            raise RuntimeError(f"No prompts loaded for domain {domain}")

        mean_in = sum(estimate_tokens(p) for p in prompts) / len(prompts)
        full_size = FULL_SUBSET_SIZES.get(domain, len(prompts))
        scale = full_size / len(prompts)

        model_rows = []
        for model in GENERATION_MODELS:
            row = _estimate_for_model(
                model,
                prompts=prompts,
                n=n,
                assumed_completion_tokens=assumed_completion_tokens,
                assumed_judge_completion_tokens=assumed_judge_completion_tokens,
                judge_models=judges,
                pricing=pricing,
                cloud_names=cloud_names,
            )
            model_rows.append(row)

            key = (row["family"], row["provider"])
            acc = family_acc.setdefault(
                key,
                {
                    "family": row["family"],
                    "provider": row["provider"],
                    "gen_usd": 0.0,
                    "judge_usd": 0.0,
                    "total_usd": 0.0,
                    "gen_calls": 0,
                    "has_usd": False,
                },
            )
            if row["gen_usd"] is not None:
                acc["gen_usd"] += row["gen_usd"]
                acc["has_usd"] = True
                sample_gen += row["gen_usd"]
                projected_gen += row["gen_usd"] * scale
            acc["judge_usd"] += row["judge_usd"]
            sample_judge += row["judge_usd"]
            projected_judge += row["judge_usd"] * scale
            if row["total_usd"] is not None:
                acc["total_usd"] += row["total_usd"]
            acc["gen_calls"] += row["gen_calls"]

            if model.provider == "ollama_cloud":
                ollama_notes.append(
                    quota_guidance(
                        row["gen_calls"],
                        sample_size=len(prompts),
                        full_size=full_size,
                    )
                    + f" [{domain} / {model.display_name}]"
                )

        domain_blocks.append(
            {
                "domain": domain,
                "sample_prompts": len(prompts),
                "mean_input_tokens": mean_in,
                "full_subset_size": full_size,
                "models": model_rows,
            }
        )

    family_rollup = []
    for acc in family_acc.values():
        family_rollup.append(
            {
                "family": acc["family"],
                "provider": acc["provider"],
                "gen_usd": acc["gen_usd"] if acc["has_usd"] else None,
                "judge_usd": acc["judge_usd"],
                "total_usd": acc["total_usd"] if acc["has_usd"] else acc["judge_usd"],
                "gen_calls": acc["gen_calls"],
            }
        )
    family_rollup.sort(key=lambda x: (x["provider"], x["family"]))

    # Deduplicate nearly-identical ollama notes — keep one summary + per-domain first
    unique_ollama = []
    seen = set()
    for note in ollama_notes:
        # collapse to domain-level one note each
        short = note.split(" [")[0]
        if short not in seen:
            seen.add(short)
            unique_ollama.append(note)

    checklist = [
        "`OPENROUTER_API_KEY` recommended for live $/token (models list often works without it).",
        "`OLLAMA_API_KEY` for Ollama Cloud open-model calls (tags may be public).",
        "Judges currently use OpenRouter free models → judge USD often $0.",
        "Ollama Cloud open-model USD in this report is an OpenRouter cross-quote only.",
        "Install subsets via `download-datasets` then `create-subset` before estimating.",
    ]
    if ollama_error:
        checklist.append(f"Ollama Cloud tags error: {ollama_error}")

    assumptions = [
        "Token counts use ceil(len(text)/4); not provider-native tokenizers.",
        "Generation assumes one chat request per prompt with n completions (input billed once).",
        f"Assumed generation completion tokens per completion: {assumed_completion_tokens}.",
        f"Assumed judge completion tokens per judge call: {assumed_judge_completion_tokens}.",
        "Judge cost = sample prompts × n × number of judge models × (judge input + output prices).",
        "Open-model gen USD uses OpenRouter sibling pricing as a research cross-quote.",
        "Ollama Cloud itself is subscription + GPU-time quota — see plan table below.",
        "No live generation/judge inference is performed for this estimate.",
    ]

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample": sample,
            "n": n,
            "domains": domains,
            "assumed_completion_tokens": assumed_completion_tokens,
            "assumed_judge_completion_tokens": assumed_judge_completion_tokens,
            "token_heuristic": "ceil(len(text)/4)",
            "openrouter_fetched_at": pricing.fetched_at,
            "ollama_tags_fetched_at": ollama_fetched_at,
            "judge_models": judges,
            "assumptions": assumptions,
            "seed": seed,
        },
        "domains": domain_blocks,
        "family_rollup": family_rollup,
        "grand_totals": {
            "sample_gen_usd": sample_gen,
            "sample_judge_usd": sample_judge,
            "sample_total_usd": sample_gen + sample_judge,
            "projected_full_gen_usd": projected_gen,
            "projected_full_judge_usd": projected_judge,
            "projected_full_total_usd": projected_gen + projected_judge,
        },
        "ollama_quota_notes": unique_ollama[:6],
        "checklist": checklist,
        "ollama_cloud_models": sorted(cloud_names) if cloud_names else [],
    }


def _print_console_summary(report: dict[str, Any]) -> None:
    print("\nCost estimate (sample)")
    print("-" * 72)
    for domain_block in report["domains"]:
        print(f"\n[{domain_block['domain']}] prompts={domain_block['sample_prompts']}")
        for row in domain_block["models"]:
            gen = row["gen_usd"]
            jud = row["judge_usd"]
            tot = row["total_usd"]
            gen_s = f"${gen:.4f}" if gen is not None else "—"
            print(
                f"  {row['display_name']:<22} {row['provider']:<14} "
                f"gen={gen_s:>10} judge=${jud:.4f} total="
                f"{('$' + f'{tot:.4f}') if tot is not None else '—':>10}"
            )
    gt = report["grand_totals"]
    print("\nGrand totals (all catalog models × domains, sample):")
    print(f"  gen=${gt['sample_gen_usd']:.4f}  judge=${gt['sample_judge_usd']:.4f}  "
          f"total=${gt['sample_total_usd']:.4f}")
    print(
        f"  projected full ≈ gen=${gt['projected_full_gen_usd']:.4f}  "
        f"judge=${gt['projected_full_judge_usd']:.4f}  "
        f"total=${gt['projected_full_total_usd']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate CreativeGainBench generation + judge costs "
        "(OpenRouter $/token + Ollama Cloud quota notes)."
    )
    parser.add_argument("--sample", type=int, default=10, help="Prompts per domain")
    parser.add_argument("--n", type=int, default=5, help="Completions per prompt")
    parser.add_argument(
        "--domains",
        type=str,
        default="infinity_chat,formalmath,rinobench",
        help="Comma-separated domains",
    )
    parser.add_argument(
        "--assumed-completion-tokens",
        type=int,
        default=800,
        help="Assumed output tokens per generation completion",
    )
    parser.add_argument(
        "--assumed-judge-completion-tokens",
        type=int,
        default=80,
        help="Assumed output tokens per judge call",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Directory for MD/JSON reports",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Override Markdown report path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Override JSON report path",
    )
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    for domain in domains:
        if domain not in DOMAIN_SUBSET_FILES:
            raise SystemExit(f"Unknown domain {domain!r}. Choose from {list(DOMAIN_SUBSET_FILES)}")

    report = build_report(
        domains=domains,
        sample=args.sample,
        n=args.n,
        assumed_completion_tokens=args.assumed_completion_tokens,
        assumed_judge_completion_tokens=args.assumed_judge_completion_tokens,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = args.output_md or (args.output_dir / f"cost_estimate_{stamp}.md")
    json_path = args.output_json or (args.output_dir / f"cost_estimate_{stamp}.json")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_console_summary(report)
    print(f"\nWrote Markdown summary: {md_path}")
    print(f"Wrote JSON report:      {json_path}")


if __name__ == "__main__":
    main()
