"""Step-CUE curve fit and Diverge-Converge score tests. Pure functions, no network."""

from __future__ import annotations

from creativegainbench.metrics.trajectory import (
    diverge_converge_score,
    fit_step_cue_curve,
    step_cue_curve,
)


def test_fit_step_cue_curve_recovers_known_parameters():
    true_c_inf, true_mu, true_gamma = 1.0, 0.5, 0.1
    t_values = list(range(1, 8))
    cue_values = [float(step_cue_curve(t, true_c_inf, true_mu, true_gamma)) for t in t_values]

    result = fit_step_cue_curve(t_values, cue_values)

    assert result.reliable is True
    assert result.n_points == 7
    assert result.c_inf is not None and abs(result.c_inf - true_c_inf) < 0.05
    assert result.mu is not None and abs(result.mu - true_mu) < 0.05
    assert result.gamma is not None and abs(result.gamma - true_gamma) < 0.05
    assert result.residual_rmse is not None and result.residual_rmse < 1e-4


def test_fit_step_cue_curve_gamma_zero_when_front_loaded():
    # Immediate convergence to a ceiling (front-loaded creative work, gamma ~ 0).
    t_values = [1, 2, 3, 4, 5]
    cue_values = [0.9, 0.99, 0.999, 0.9999, 0.99999]
    result = fit_step_cue_curve(t_values, cue_values)
    assert result.reliable is True
    assert result.gamma is not None and abs(result.gamma) < 0.05


def test_fit_step_cue_curve_too_few_points_returns_none():
    result = fit_step_cue_curve([1, 2], [0.1, 0.2])
    assert result.c_inf is None
    assert result.mu is None
    assert result.gamma is None
    assert result.reliable is False
    assert result.n_points == 2


def test_fit_step_cue_curve_exactly_three_points_flagged_unreliable():
    result = fit_step_cue_curve([1, 2, 3], [0.1, 0.2, 0.3])
    assert result.n_points == 3
    assert result.reliable is False  # fit may succeed, but exactly-saturated


def test_diverge_converge_tent_trajectory_matches_lean_example():
    # Mirrors Lean `tentTrajectory`: E = [-1, 0, -1], peak at index 1.
    result = diverge_converge_score([-1.0, 0.0, -1.0])
    assert result.dc == 1
    assert result.peak_index == 1
    assert result.n_points == 3


def test_diverge_converge_interior_peak_amid_more_points():
    result = diverge_converge_score([0.0, 1.0, 2.0, 1.0, 0.0])
    assert result.dc == 1
    assert result.peak_index == 2


def test_diverge_converge_monotone_increasing_is_zero():
    result = diverge_converge_score([0.0, 1.0, 2.0, 3.0])
    assert result.dc == 0
    assert result.peak_index is None


def test_diverge_converge_monotone_decreasing_is_zero():
    result = diverge_converge_score([3.0, 2.0, 1.0, 0.0])
    assert result.dc == 0
    assert result.peak_index is None


def test_diverge_converge_too_few_points_is_zero():
    result = diverge_converge_score([0.0, 1.0])
    assert result.dc == 0
    assert result.peak_index is None
    assert result.n_points == 2


def test_diverge_converge_flat_trajectory_is_zero():
    result = diverge_converge_score([1.0, 1.0, 1.0, 1.0])
    assert result.dc == 0
