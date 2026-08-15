"""Focused tests for the creativity_stats equivalence framework."""
from __future__ import annotations

import numpy as np
import pytest

from creativegainbench.stats import (
    CRPS,
    ComparisonPipeline,
    ConcordanceCC,
    CreativityMetric,
    EnergyDistance,
    Geometry,
    HedgesG,
    KrippendorffAlpha,
    LogVarianceRatio,
    MeasurementLevel,
    PITUniformity,
    Pairing,
    Resampler,
    Sample,
    StatisticalMeasure,
    Verdict,
    krippendorff_alpha,
    mean_crps,
    pit_ks_distance,
    pit_values,
    report_to_dict,
)


class _ToyMetric(CreativityMetric):
    name = "toy"
    framework = "test"
    level = MeasurementLevel.CONTINUOUS
    bounds = (0.0, 1.0)


class _SignedStub(StatisticalMeasure):
    """Deterministic signed-difference measure for verdict-path tests."""
    pairing = Pairing.UNPAIRED
    geometry = Geometry.SIGNED_DIFFERENCE

    def __init__(self, value: float, margin: float = 0.2, alpha: float = 0.05):
        super().__init__(alpha=alpha, margin=margin)
        self._value = value

    @property
    def name(self):
        return "signed_stub"

    def statistic(self, human, model):
        return self._value

    def default_margin(self, sample):
        return 0.2


class _NonnegStub(StatisticalMeasure):
    pairing = Pairing.UNPAIRED
    geometry = Geometry.NONNEG_DISTANCE

    def __init__(self, value: float, margin: float = 0.2, alpha: float = 0.05):
        super().__init__(alpha=alpha, margin=margin)
        self._value = value

    @property
    def name(self):
        return "nonneg_stub"

    def statistic(self, human, model):
        return self._value

    def default_margin(self, sample):
        return 0.2


class _AgreementStub(StatisticalMeasure):
    pairing = Pairing.PAIRED
    geometry = Geometry.AGREEMENT

    def __init__(self, value: float, margin: float = 0.8, alpha: float = 0.05):
        super().__init__(alpha=alpha, margin=margin)
        self._value = value

    @property
    def name(self):
        return "agreement_stub"

    def statistic(self, human, model):
        return self._value

    def default_margin(self, sample):
        return 0.8


def test_sample_require_paired_rejects_unequal_lengths():
    s = Sample([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="equal-length"):
        s.require_paired()


def test_sample_require_paired_requires_item_ids():
    s = Sample([1.0, 2.0], [1.5, 2.5])  # equal lengths, no item_ids
    with pytest.raises(ValueError, match="item_ids"):
        s.require_paired()


def test_sample_require_paired_rejects_mismatched_item_ids():
    s = Sample([1.0, 2.0], [1.5, 2.5], item_ids=np.array([0]))
    with pytest.raises(ValueError, match="item_ids length"):
        s.require_paired()


def test_sample_require_paired_accepts_equal_lengths():
    s = Sample([1.0, 2.0], [1.5, 2.5], item_ids=np.array([0, 1]))
    s.require_paired()  # no raise


def test_energy_distance_matches_pairwise_definition():
    rng = np.random.default_rng(20)
    h = rng.normal(0, 1, 25)
    m = rng.normal(0.2, 1.2, 30)
    naive = (
        2 * np.abs(h[:, None] - m[None, :]).mean()
        - np.abs(h[:, None] - h[None, :]).mean()
        - np.abs(m[:, None] - m[None, :]).mean()
    )
    assert EnergyDistance().statistic(h, m) == pytest.approx(naive, rel=1e-12)


def test_bca_ignores_nonfinite_replicates():
    rs = Resampler(n_boot=10, n_perm=10, seed=21)
    theta = 0.5
    boot = np.array([0.4, 0.5, np.nan, 0.6, 0.55, np.inf, 0.45, 0.52, 0.48, 0.51])
    jack = np.array([0.49, 0.51, np.nan, 0.50, 0.48])
    est, lo, hi = rs._bca_from_replicates(theta, boot, jack)
    assert est == 0.5
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_pit_values_rejects_bad_shape():
    with pytest.raises(ValueError, match="draws must have shape"):
        pit_values(np.array([0.5]), np.array([0.0, 0.5, 1.0]))  # 1-D


def test_pipeline_alpha_propagates_to_measures():
    m1 = HedgesG(alpha=0.10)
    m2 = EnergyDistance(alpha=0.10)
    pipe = ComparisonPipeline(measures=[m1, m2], alpha=0.01)
    assert m1.alpha == 0.01
    assert m2.alpha == 0.01
    assert pipe.alpha == 0.01


def test_signed_difference_verdict_paths():
    m = _SignedStub(0.0, margin=0.2)
    assert m._verdict(-0.1, 0.1, p=0.5, margin=0.2) is Verdict.EQUIVALENT
    assert m._verdict(-0.1, 0.1, p=0.01, margin=0.2) is Verdict.TRIVIALLY_DIFFERENT
    assert m._verdict(-0.5, -0.3, p=0.01, margin=0.2) is Verdict.DIFFERENT
    assert m._verdict(-0.5, 0.5, p=0.5, margin=0.2) is Verdict.INDETERMINATE


def test_nonneg_distance_verdict_paths():
    m = _NonnegStub(0.0, margin=0.2)
    assert m._verdict(0.0, 0.1, p=0.5, margin=0.2) is Verdict.EQUIVALENT
    assert m._verdict(0.0, 0.1, p=0.01, margin=0.2) is Verdict.TRIVIALLY_DIFFERENT
    assert m._verdict(0.3, 0.5, p=0.01, margin=0.2) is Verdict.DIFFERENT
    assert m._verdict(0.1, 0.3, p=0.5, margin=0.2) is Verdict.INDETERMINATE


def test_agreement_verdict_paths():
    m = _AgreementStub(0.9, margin=0.8)
    assert m._verdict(0.85, 0.95, p=None, margin=0.8) is Verdict.EQUIVALENT
    assert m._verdict(0.5, 0.7, p=None, margin=0.8) is Verdict.DIFFERENT
    assert m._verdict(0.7, 0.9, p=None, margin=0.8) is Verdict.INDETERMINATE


def test_unpaired_sample_skips_paired_measures():
    rng = np.random.default_rng(0)
    human = rng.normal(0.5, 0.1, 40)
    model = rng.normal(0.5, 0.1, 40)
    sample = Sample(human, model)  # no item_ids => unpaired

    pipe = ComparisonPipeline(
        measures=[EnergyDistance(), HedgesG(), ConcordanceCC()],
        resampler=Resampler(n_boot=40, n_perm=40, seed=1),
    )
    rep = pipe.run(_ToyMetric(), sample)
    names = {r.name for r in rep.results}
    assert "lins_ccc" not in names
    assert "energy_distance" in names
    assert "hedges_g" in names


def test_paired_sample_runs_ccc():
    rng = np.random.default_rng(2)
    human = rng.normal(0.5, 0.1, 30)
    model = human + 0.02  # high agreement, tiny bias
    sample = Sample(human, model, item_ids=np.arange(30))

    pipe = ComparisonPipeline(
        measures=[ConcordanceCC(margin=0.5)],
        resampler=Resampler(n_boot=40, n_perm=40, seed=3),
    )
    rep = pipe.run(_ToyMetric(), sample)
    assert len(rep.results) == 1
    assert rep.results[0].name == "lins_ccc"
    assert rep.results[0].estimate > 0.5


def test_log_variance_ratio_detects_mode_collapse():
    rng = np.random.default_rng(4)
    human = rng.normal(0.6, 0.15, 80)
    model = rng.normal(0.6, 0.05, 80)
    sample = Sample(human, model)

    msr = LogVarianceRatio()
    rs = Resampler(n_boot=80, n_perm=80, seed=5)
    result = msr.evaluate(sample, rs)
    assert result.estimate < 0  # model less dispersed
    assert result.verdict in (Verdict.DIFFERENT, Verdict.INDETERMINATE, Verdict.TRIVIALLY_DIFFERENT)


def test_sample_optional_extensions_validated():
    with pytest.raises(ValueError, match="rater_matrix must be 2-D"):
        Sample([1.0], [1.0], rater_matrix=[1.0, 2.0])
    with pytest.raises(ValueError, match="predictive must be 2-D"):
        Sample([1.0], [1.0], predictive=[1.0, 2.0])
    s = Sample(
        [1.0, 2.0],
        [1.0, 2.0],
        rater_matrix=[[1.0, 1.0], [2.0, 2.0]],
        predictive=[[0.0, 1.0], [1.0, 2.0]],
    )
    assert s.rater_matrix.shape == (2, 2)
    assert s.predictive.shape == (2, 2)


def test_krippendorff_alpha_perfect_agreement():
    R = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
    ])
    assert krippendorff_alpha(R, level="interval") == pytest.approx(1.0)
    assert krippendorff_alpha(R, level="nominal") == pytest.approx(1.0)
    assert krippendorff_alpha(R, level="ordinal") == pytest.approx(1.0)


def test_krippendorff_alpha_allows_nans():
    R = np.array([
        [1.0, 1.0, np.nan],
        [2.0, 2.0, 2.0],
        [3.0, np.nan, 3.0],
    ])
    a = krippendorff_alpha(R, level="interval")
    assert a == pytest.approx(1.0)


def test_reliability_runs_when_rater_matrix_present():
    rng = np.random.default_rng(6)
    # Shared latent + rater noise → moderate α
    latent = rng.normal(0.5, 0.2, 40)
    R = np.column_stack([latent + rng.normal(0, 0.05, 40) for _ in range(3)])
    sample = Sample(latent, latent, rater_matrix=R)

    pipe = ComparisonPipeline(
        measures=[EnergyDistance(), KrippendorffAlpha(margin=0.5)],
        resampler=Resampler(n_boot=30, n_perm=30, seed=7),
    )
    rep = pipe.run(_ToyMetric(), sample)
    names = {r.name for r in rep.results}
    assert "krippendorff_alpha" in names
    assert "energy_distance" in names
    alpha_row = next(r for r in rep.results if r.name == "krippendorff_alpha")
    assert alpha_row.estimate > 0.5
    assert alpha_row.extra.get("reliability_level") == "interval"


def test_reliability_skipped_without_rater_matrix():
    sample = Sample([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    pipe = ComparisonPipeline(
        measures=[KrippendorffAlpha()],
        resampler=Resampler(n_boot=20, n_perm=20, seed=8),
    )
    rep = pipe.run(_ToyMetric(), sample)
    assert rep.results == []


def test_mean_crps_perfect_point_mass():
    obs = np.array([1.0, 2.0, 3.0])
    draws = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert mean_crps(obs, draws) == pytest.approx(0.0)


def test_crps_and_pit_run_when_predictive_present():
    rng = np.random.default_rng(9)
    n, m = 35, 20
    human = rng.normal(0.0, 1.0, n)
    # Well-calibrated predictive: draws from N(0,1) independent of obs scale match
    predictive = rng.normal(0.0, 1.0, (n, m))
    # Shifted / miscalibrated arm unused by calibration measures
    model = human + 0.5
    sample = Sample(human, model, item_ids=np.arange(n), predictive=predictive)

    pipe = ComparisonPipeline(
        measures=[ConcordanceCC(margin=0.3), CRPS(), PITUniformity(n_bins=5)],
        resampler=Resampler(n_boot=30, n_perm=30, seed=10),
    )
    rep = pipe.run(_ToyMetric(), sample)
    names = {r.name for r in rep.results}
    assert "crps" in names
    assert "pit_uniformity" in names
    assert "lins_ccc" in names  # paired via item_ids still works

    pit_row = next(r for r in rep.results if r.name == "pit_uniformity")
    assert "pit_histogram" in pit_row.extra
    assert len(pit_row.extra["pit_histogram"]) == 5
    assert "pit_mad" in pit_row.extra
    assert pit_row.estimate >= 0


def test_calibration_skipped_without_predictive():
    sample = Sample([0.1, 0.2], [0.1, 0.2], item_ids=[0, 1])
    pipe = ComparisonPipeline(
        measures=[CRPS(), PITUniformity()],
        resampler=Resampler(n_boot=20, n_perm=20, seed=11),
    )
    rep = pipe.run(_ToyMetric(), sample)
    assert rep.results == []


def test_calibration_skipped_when_predictive_length_mismatch():
    sample = Sample(
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        predictive=np.zeros((2, 5)),  # wrong n_items
    )
    pipe = ComparisonPipeline(
        measures=[CRPS()],
        resampler=Resampler(n_boot=20, n_perm=20, seed=12),
    )
    rep = pipe.run(_ToyMetric(), sample)
    assert rep.results == []


def test_pit_ks_small_for_uniform():
    rng = np.random.default_rng(13)
    u = rng.uniform(0, 1, 200)
    assert pit_ks_distance(u) < 0.15


def test_pit_values_midrank():
    obs = np.array([0.5])
    draws = np.array([[0.0, 0.5, 1.0]])
    # lt=1, eq=1, M=3 → (1 + 0.5)/3 = 0.5
    assert pit_values(obs, draws)[0] == pytest.approx(0.5)


def test_report_to_dict_shape():
    rng = np.random.default_rng(14)
    human = rng.normal(0.5, 0.1, 25)
    model = rng.normal(0.5, 0.1, 25)
    sample = Sample(human, model)
    pipe = ComparisonPipeline(
        measures=[HedgesG()],
        resampler=Resampler(n_boot=25, n_perm=25, seed=15),
    )
    rep = pipe.run(_ToyMetric(), sample)
    payload = report_to_dict(rep, model="gpt-test")
    assert payload["metric"] == "toy"
    assert payload["model"] == "gpt-test"
    assert payload["results"][0]["measure"] == "hedges_g"
    assert payload["results"][0]["model"] == "gpt-test"
    assert "p_adj_by" in payload["results"][0]


def test_unpaired_behavior_unchanged_with_extensions_absent():
    """Existing unpaired path ignores reliability/calibration when fields missing."""
    rng = np.random.default_rng(16)
    sample = Sample(rng.normal(0, 1, 30), rng.normal(0, 1, 28))
    pipe = ComparisonPipeline(
        measures=[EnergyDistance(), HedgesG(), ConcordanceCC(),
                  KrippendorffAlpha(), CRPS(), PITUniformity()],
        resampler=Resampler(n_boot=25, n_perm=25, seed=17),
    )
    rep = pipe.run(_ToyMetric(), sample)
    names = {r.name for r in rep.results}
    assert names == {"energy_distance", "hedges_g"}
