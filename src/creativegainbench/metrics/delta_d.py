"""
Per-domain δ_D thresholds from negative-bank one-sided quantiles.

Canonical gate contract: 1[R_D_norm > δ_D(domain)]. Thresholds are never
fit to human/model scores or z*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values for quantile")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def thresholds_from_negatives(
    by_domain: Mapping[str | int, list[float]],
    *,
    q: float = 0.95,
    eps: float = 1e-6,
) -> dict[str, dict[str, Any]]:
    """
    Build {domain: {delta_d_95, n_neg, quantile}} from negative R_D_norm lists.
    """
    out: dict[str, dict[str, Any]] = {}
    for domain, vals in by_domain.items():
        if not vals:
            continue
        thr = quantile(list(vals), q) + eps
        out[str(domain)] = {
            "delta_d_95": float(thr),
            "n_neg": len(vals),
            "quantile": float(q),
        }
    return out


def load_delta_d_thresholds(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing δ_D thresholds at {path}. Run calibrate-delta-d "
            "(negative-bank calibrator) first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid thresholds file: {path}")
    return data


def resolve_delta_d(
    thresholds: Mapping[str, Mapping[str, Any]],
    domain: str | int | None,
) -> float:
    """
    Look up δ_D for a domain. Falls back to key \"default\" then first entry.
    """
    if domain is not None:
        key = str(domain)
        if key in thresholds:
            return float(thresholds[key]["delta_d_95"])
    if "default" in thresholds:
        return float(thresholds["default"]["delta_d_95"])
    # Single-domain package artifacts often use "0".
    if "0" in thresholds:
        return float(thresholds["0"]["delta_d_95"])
    first = next(iter(thresholds.values()))
    return float(first["delta_d_95"])


def d_gate(r_d: float, delta_d: float, *, feasible: bool = True) -> float:
    """
    1[feasible] · 1[R_D > δ_D].

    Feasibility is the utility gate U(q,y)≥τ proxy. Default ``feasible=True``
    preserves call sites that already applied the check; scorers should pass
    ``feasibility_bit(y)`` explicitly.
    """
    return 1.0 if feasible and r_d > delta_d else 0.0


def write_delta_d_thresholds(path: str | Path, thresholds: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(thresholds), indent=2) + "\n", encoding="utf-8")
    return path
