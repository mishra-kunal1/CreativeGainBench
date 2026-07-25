"""Lightweight shared token estimator (approximate across providers)."""

from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    """Rough token count: ceil(chars / 4). Good enough for planning estimates."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
