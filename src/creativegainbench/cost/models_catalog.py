"""Curated model catalog for cost estimation.

Proprietary families use OpenRouter for runtime pricing.
Open families prefer Ollama Cloud (no local downloads) with an OpenRouter
sibling slug for $/token cross-quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provider = Literal["openrouter", "ollama_cloud"]


@dataclass(frozen=True)
class CatalogModel:
    family: str
    display_name: str
    provider: Provider
    # Primary id used for that provider (OpenRouter slug or Ollama Cloud name).
    model_id: str
    # Optional OpenRouter slug for $/token when provider is ollama_cloud.
    openrouter_cross_quote: str | None = None


# Judge models used by creativegainbench.eval.llm_as_judge (OpenRouter).
DEFAULT_JUDGE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "cohere/north-mini-code:free",
]


GENERATION_MODELS: list[CatalogModel] = [
    # Proprietary — OpenRouter
    CatalogModel("gpt", "GPT-4o mini", "openrouter", "openai/gpt-4o-mini"),
    CatalogModel("gpt", "GPT-4o", "openrouter", "openai/gpt-4o"),
    CatalogModel("claude", "Claude Sonnet 4", "openrouter", "anthropic/claude-sonnet-4"),
    CatalogModel("claude", "Claude Opus 4", "openrouter", "anthropic/claude-opus-4"),
    CatalogModel("gemini", "Gemini 2.5 Flash", "openrouter", "google/gemini-2.5-flash"),
    CatalogModel("gemini", "Gemini 2.5 Pro", "openrouter", "google/gemini-2.5-pro"),
    # Open — Ollama Cloud (+ OpenRouter $ cross-quote)
    CatalogModel(
        "llama",
        "Llama 4 Maverick",
        "ollama_cloud",
        "llama4",
        openrouter_cross_quote="meta-llama/llama-4-maverick",
    ),
    CatalogModel(
        "deepseek",
        "DeepSeek V4 Flash",
        "ollama_cloud",
        "deepseek-v4-flash",
        openrouter_cross_quote="deepseek/deepseek-v4-flash",
    ),
    CatalogModel(
        "kimi",
        "Kimi K2.6",
        "ollama_cloud",
        "kimi-k2.6",
        openrouter_cross_quote="moonshotai/kimi-k2.6",
    ),
    CatalogModel(
        "glm",
        "GLM 5.1",
        "ollama_cloud",
        "glm-5.1",
        openrouter_cross_quote="z-ai/glm-5.1",
    ),
    CatalogModel(
        "qwen",
        "Qwen 3.5 397B",
        "ollama_cloud",
        "qwen3.5:397b",
        openrouter_cross_quote="qwen/qwen3.5-397b-a17b",
    ),
]


DOMAIN_SUBSET_FILES = {
    "infinity_chat": "infinity_chat_subset.jsonl",
    "formalmath": "formalmath_subset.jsonl",
    "rinobench": "rinobench_subset.jsonl",
}

FULL_SUBSET_SIZES = {
    "infinity_chat": 300,
    "formalmath": 300,
    "rinobench": 299,
}
