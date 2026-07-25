"""Fetch live OpenRouter model pricing (USD per token)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class ModelPrice:
    model_id: str
    prompt_per_token: float
    completion_per_token: float
    raw: dict[str, Any] | None = None


@dataclass
class PricingSnapshot:
    fetched_at: str
    by_id: dict[str, ModelPrice]

    def get(self, model_id: str) -> ModelPrice | None:
        return self.by_id.get(model_id)


def fetch_openrouter_pricing(api_key: str | None = None) -> PricingSnapshot:
    """GET OpenRouter /models and index prompt/completion prices."""
    key = (api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")).strip()
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter /models failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter /models network error: {exc}") from exc

    by_id: dict[str, ModelPrice] = {}
    for item in payload.get("data") or []:
        mid = item.get("id")
        pricing = item.get("pricing") or {}
        if not mid:
            continue
        try:
            prompt = float(pricing.get("prompt") or 0.0)
            completion = float(pricing.get("completion") or 0.0)
        except (TypeError, ValueError):
            continue
        by_id[mid] = ModelPrice(
            model_id=mid,
            prompt_per_token=prompt,
            completion_per_token=completion,
            raw=item,
        )

    return PricingSnapshot(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        by_id=by_id,
    )


def cost_usd(
    price: ModelPrice,
    *,
    input_tokens: float,
    output_tokens: float,
) -> float:
    return input_tokens * price.prompt_per_token + output_tokens * price.completion_per_token
