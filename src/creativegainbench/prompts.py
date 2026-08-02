"""Normalize dataset rows into a shared prompt string for inference and costing."""

from __future__ import annotations


def prompt_from_infinity_chat(row: dict) -> str:
    if "prompt" in row and row["prompt"]:
        return str(row["prompt"])
    messages = row.get("messages") or []
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    raise ValueError("infinity_chat row has no prompt or user message")


def prompt_from_formalmath(row: dict) -> str:
    if "prompt" in row and row["prompt"]:
        return str(row["prompt"])
    statement = row.get("refined_statement")
    if not statement:
        raise ValueError("formalmath row missing refined_statement")
    return str(statement)


def prompt_from_rinobench(row: dict) -> str:
    if "prompt" in row and row["prompt"]:
        return str(row["prompt"])
    idea = row.get("research_idea") or {}
    objective = (idea.get("objective") or "").strip()
    problem = (idea.get("problem_statement") or "").strip()
    solution = (idea.get("solution_approach") or "").strip()
    parts = []
    if objective:
        parts.append(f"Objective: {objective}")
    if problem:
        parts.append(f"Problem statement: {problem}")
    if solution:
        parts.append(f"Solution approach: {solution}")
    if not parts:
        raise ValueError("rinobench row has empty research_idea fields")
    return "\n\n".join(parts)


DOMAIN_PROMPT_FN = {
    "infinity_chat": prompt_from_infinity_chat,
    "formalmath": prompt_from_formalmath,
    "rinobench": prompt_from_rinobench,
}


def extract_prompt(domain: str, row: dict) -> str:
    try:
        fn = DOMAIN_PROMPT_FN[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown domain: {domain}") from exc
    return fn(row)
