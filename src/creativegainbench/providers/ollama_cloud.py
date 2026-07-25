"""Ollama Cloud client — remote open models without local downloads."""

from __future__ import annotations

import json
import os
from typing import Any

import urllib.error
import urllib.request

from openai import OpenAI

OLLAMA_CLOUD_HOST = "https://ollama.com"
OLLAMA_CLOUD_OPENAI_BASE = "https://ollama.com/v1"
TAGS_URL = f"{OLLAMA_CLOUD_HOST}/api/tags"


def _api_key() -> str:
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OLLAMA_API_KEY is required for Ollama Cloud. "
            "Create a key at https://ollama.com/settings/keys"
        )
    return key


def list_cloud_models(api_key: str | None = None) -> list[dict[str, Any]]:
    """List models available on Ollama Cloud (GET /api/tags).

    Auth is optional for the public tags listing; pass api_key='' to skip the header.
    """
    if api_key is None:
        key = os.environ.get("OLLAMA_API_KEY", "").strip()
    else:
        key = api_key.strip()
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        TAGS_URL,
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama Cloud /api/tags failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama Cloud /api/tags network error: {exc}") from exc

    data = json.loads(payload)
    models = data.get("models") or data.get("data") or []
    if isinstance(models, dict):
        models = list(models.values())
    return models


def cloud_model_names(api_key: str | None = None) -> set[str]:
    """Return a set of model name strings from Ollama Cloud tags."""
    names: set[str] = set()
    for model in list_cloud_models(api_key=api_key):
        name = model.get("name") or model.get("model") or model.get("id")
        if name:
            names.add(str(name))
            # Also index base name without tag (e.g. llama3.3:70b -> llama3.3)
            if ":" in str(name):
                names.add(str(name).split(":", 1)[0])
    return names


def openai_client(api_key: str | None = None) -> OpenAI:
    """OpenAI-compatible client pointed at Ollama Cloud (no local pull)."""
    key = api_key if api_key is not None else _api_key()
    return OpenAI(base_url=OLLAMA_CLOUD_OPENAI_BASE, api_key=key)


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 1.0,
    top_p: float = 0.9,
    n: int = 1,
    api_key: str | None = None,
) -> Any:
    """Chat completion via Ollama Cloud OpenAI-compatible API."""
    client = openai_client(api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n,
    )
