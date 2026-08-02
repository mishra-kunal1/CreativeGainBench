"""
Trajectory-level CUE diagnostics: Step-CUE curve (gamma) and Diverge-Converge (DC).

Mirrors Lean `Creativity.CUE.Trajectory`:
  * stepCUEFit Cinf mu gamma t = Cinf * (1 - exp(-mu*t)) + gamma*t  (PROOF-14)
  * hasDivergeConverge E tstar T: strictly increasing before tstar, strictly
    decreasing from tstar to T, with 0 < tstar < T (PROOF-05) -- tstar must
    be a strictly interior index, so a monotone trajectory (peak at the
    first or last point) is definitionally DC=0, and DC is only evaluable
    at all with >= 3 points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import curve_fit

# 3 free parameters (C_inf, mu, gamma); a fit needs strictly more points than
# parameters to be meaningfully overdetermined rather than just interpolating.
MIN_POINTS_FOR_RELIABLE_FIT = 4


def step_cue_curve(t: np.ndarray, c_inf: float, mu: float, gamma: float) -> np.ndarray:
    """C_inf * (1 - exp(-mu*t)) + gamma*t -- mirrors Lean `stepCUEFit`."""
    t = np.asarray(t, dtype=float)
    return c_inf * (1.0 - np.exp(-mu * t)) + gamma * t


@dataclass
class StepCUEFitResult:
    c_inf: float | None
    mu: float | None
    gamma: float | None
    n_points: int
    reliable: bool
    residual_rmse: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_step_cue_curve(t_values: Sequence[float], cue_values: Sequence[float]) -> StepCUEFitResult:
    """
    Fit CUE(t) = C_inf*(1 - exp(-mu*t)) + gamma*t via nonlinear least squares.

    Fewer than 3 points can't identify 3 free parameters at all (returns
    None params). Exactly-3-point fits are returned but flagged
    `reliable=False` -- exactly saturated, so the fit reduces to
    interpolation with no residual signal about goodness of fit.
    `reliable=True` only from MIN_POINTS_FOR_RELIABLE_FIT points on.
    """
    n = len(t_values)
    if n < 3 or n != len(cue_values):
        return StepCUEFitResult(None, None, None, n, False, None)

    t = np.asarray(t_values, dtype=float)
    y = np.asarray(cue_values, dtype=float)
    c_inf_guess = float(y[-1]) if y[-1] != 0 else 1.0
    try:
        popt, _ = curve_fit(
            step_cue_curve,
            t,
            y,
            p0=[c_inf_guess, 0.5, 0.0],
            maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return StepCUEFitResult(None, None, None, n, False, None)

    c_inf, mu, gamma = (float(v) for v in popt)
    residual_rmse = float(np.sqrt(np.mean((step_cue_curve(t, c_inf, mu, gamma) - y) ** 2)))
    return StepCUEFitResult(
        c_inf=c_inf,
        mu=mu,
        gamma=gamma,
        n_points=n,
        reliable=n >= MIN_POINTS_FOR_RELIABLE_FIT,
        residual_rmse=residual_rmse,
    )


@dataclass
class DivergeConvergeResult:
    dc: int
    peak_index: int | None
    n_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diverge_converge_score(rb_values: Sequence[float]) -> DivergeConvergeResult:
    """
    DC = 1 iff there is a strictly interior peak index `tstar` (0 < tstar <
    T, i.e. not the first or last point) such that rb_values strictly
    increases up to tstar and strictly decreases from tstar onward --
    mirrors Lean `hasDivergeConverge`. A monotone trajectory (peak at either
    end) is definitionally DC=0, not an edge case of DC=1. Fewer than 3
    points can never satisfy the strict-interior requirement, so DC=0.
    """
    n = len(rb_values)
    if n < 3:
        return DivergeConvergeResult(dc=0, peak_index=None, n_points=n)

    last = n - 1
    for tstar in range(1, last):
        rising = all(rb_values[t] < rb_values[t + 1] for t in range(0, tstar))
        falling = all(rb_values[t + 1] < rb_values[t] for t in range(tstar, last))
        if rising and falling:
            return DivergeConvergeResult(dc=1, peak_index=tstar, n_points=n)
    return DivergeConvergeResult(dc=0, peak_index=None, n_points=n)
