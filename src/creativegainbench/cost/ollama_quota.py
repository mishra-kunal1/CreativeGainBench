"""Ollama Cloud subscription / GPU-time quota notes (not $/token)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaPlan:
    name: str
    monthly_usd: float
    concurrent_models: int
    usage_note: str


OLLAMA_PLANS = [
    OllamaPlan(
        name="Free",
        monthly_usd=0.0,
        concurrent_models=1,
        usage_note="Light cloud usage; session/weekly quotas apply (GPU-time, not tokens).",
    ),
    OllamaPlan(
        name="Pro",
        monthly_usd=20.0,
        concurrent_models=3,
        usage_note="~50x Free cloud usage; suitable for day-to-day open-model workloads.",
    ),
    OllamaPlan(
        name="Max",
        monthly_usd=100.0,
        concurrent_models=10,
        usage_note="Highest included usage; heavy concurrent / sustained agent workloads.",
    ),
]


def quota_guidance(generation_calls: int, *, sample_size: int, full_size: int) -> str:
    """Heuristic note for which plan may fit given call volume."""
    full_calls = int(generation_calls * (full_size / max(sample_size, 1)))
    if generation_calls <= 50:
        tier = "Free may cover a tiny smoke sample; Pro is safer for repeated runs."
    elif generation_calls <= 500:
        tier = "Pro ($20/mo) is the usual fit for sample-scale open-model generation."
    else:
        tier = "Pro or Max recommended once call volume is sustained / concurrent."
    return (
        f"Ollama Cloud bills via subscription + GPU-time quota (not $/token). "
        f"Sample generation calls≈{generation_calls}; "
        f"linear full-subset projection≈{full_calls}. {tier} "
        f"See https://ollama.com/pricing"
    )


def plans_markdown_table() -> str:
    lines = [
        "| Plan | Monthly USD | Concurrent models | Notes |",
        "| --- | ---: | ---: | --- |",
    ]
    for plan in OLLAMA_PLANS:
        lines.append(
            f"| {plan.name} | ${plan.monthly_usd:.0f} | {plan.concurrent_models} | {plan.usage_note} |"
        )
    return "\n".join(lines)
