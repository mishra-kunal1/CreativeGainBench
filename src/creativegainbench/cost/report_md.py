"""Render cost estimate results as a Markdown summary."""

from __future__ import annotations

from typing import Any

from creativegainbench.cost.ollama_quota import plans_markdown_table


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) < 0.01 and value != 0:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    leftish = {"model", "provider", "family", "notes", "availability", "available"}
    align = []
    for h in headers:
        align.append("---" if h.lower() in leftish else "---:")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    parts: list[str] = []

    parts.append("# CreativeGainBench cost estimate")
    parts.append("")
    parts.append("## Run metadata")
    parts.append("")
    parts.append(f"- **Generated at (UTC):** {meta['generated_at']}")
    parts.append(f"- **Sample size per domain:** {meta['sample']}")
    parts.append(f"- **Completions per prompt (`n`):** {meta['n']}")
    parts.append(f"- **Domains:** {', '.join(meta['domains'])}")
    parts.append(f"- **Assumed generation completion tokens:** {meta['assumed_completion_tokens']}")
    parts.append(f"- **Assumed judge completion tokens:** {meta['assumed_judge_completion_tokens']}")
    parts.append(f"- **Token heuristic:** `{meta['token_heuristic']}`")
    parts.append(f"- **OpenRouter prices fetched at:** {meta.get('openrouter_fetched_at') or 'n/a'}")
    parts.append(f"- **Ollama Cloud tags fetched at:** {meta.get('ollama_tags_fetched_at') or 'n/a'}")
    parts.append("")

    parts.append("## Assumptions")
    parts.append("")
    for line in meta.get("assumptions", []):
        parts.append(f"- {line}")
    parts.append("")

    parts.append("## Per-domain estimates (sample)")
    parts.append("")

    for domain_block in report["domains"]:
        domain = domain_block["domain"]
        parts.append(f"### {domain}")
        parts.append("")
        parts.append(
            f"- Prompts in sample: **{domain_block['sample_prompts']}** "
            f"(mean input tokens ≈ {domain_block['mean_input_tokens']:.1f})"
        )
        parts.append(
            f"- Full subset size (for scale-up): **{domain_block['full_subset_size']}**"
        )
        parts.append("")

        rows = []
        for row in domain_block["models"]:
            notes = row.get("notes") or ""
            avail = "yes" if row.get("available", True) else "no"
            rows.append(
                [
                    row["display_name"],
                    row["provider"],
                    avail,
                    _fmt_usd(row.get("gen_usd")),
                    _fmt_usd(row.get("judge_usd")),
                    _fmt_usd(row.get("total_usd")),
                    str(row.get("gen_calls", "")),
                    notes.replace("|", "/"),
                ]
            )
        parts.append(
            _md_table(
                [
                    "Model",
                    "Provider",
                    "Available",
                    "Gen USD",
                    "Judge USD",
                    "Total USD",
                    "Gen calls",
                    "Notes",
                ],
                rows,
            )
        )
        parts.append("")

    parts.append("## Provider / family rollup (sample)")
    parts.append("")
    roll_rows = []
    for item in report["family_rollup"]:
        roll_rows.append(
            [
                item["family"],
                item["provider"],
                _fmt_usd(item.get("gen_usd")),
                _fmt_usd(item.get("judge_usd")),
                _fmt_usd(item.get("total_usd")),
                str(item.get("gen_calls", "")),
            ]
        )
    parts.append(
        _md_table(
            ["Family", "Provider", "Gen USD", "Judge USD", "Total USD", "Gen calls"],
            roll_rows,
        )
    )
    parts.append("")

    parts.append("## Grand totals and scale-up")
    parts.append("")
    gt = report["grand_totals"]
    parts.append(f"- **Sample gen USD (OpenRouter-priced rows):** {_fmt_usd(gt.get('sample_gen_usd'))}")
    parts.append(f"- **Sample judge USD:** {_fmt_usd(gt.get('sample_judge_usd'))}")
    parts.append(f"- **Sample total USD:** {_fmt_usd(gt.get('sample_total_usd'))}")
    parts.append(
        f"- **Projected full-subset gen USD** (linear × full/sample): "
        f"{_fmt_usd(gt.get('projected_full_gen_usd'))}"
    )
    parts.append(
        f"- **Projected full-subset judge USD:** {_fmt_usd(gt.get('projected_full_judge_usd'))}"
    )
    parts.append(
        f"- **Projected full-subset total USD:** {_fmt_usd(gt.get('projected_full_total_usd'))}"
    )
    parts.append("")
    parts.append(
        "Scale-up is linear in prompt count and assumes the same mean input length and "
        "output-token assumptions. Ollama Cloud open-model dollars above use OpenRouter "
        "cross-quotes only; actual Ollama Cloud spend is subscription/quota."
    )
    parts.append("")

    parts.append("## Ollama Cloud plans (open models)")
    parts.append("")
    parts.append(plans_markdown_table())
    parts.append("")
    if report.get("ollama_quota_notes"):
        for note in report["ollama_quota_notes"]:
            parts.append(f"- {note}")
        parts.append("")

    parts.append("## API / plan checklist")
    parts.append("")
    for item in report.get("checklist", []):
        parts.append(f"- {item}")
    parts.append("")

    return "\n".join(parts)
