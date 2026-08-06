"""
creativity_stats.py
===================

A pluggable framework for comparing LLM-vs-human creativity scores with
*equivalence-first* inference.

Two extension points:
  * CreativityMetric   -- the domain plug (what score are we comparing?)
  * StatisticalMeasure -- the stats plug (how do we compare two samples?)

All resampling (BCa bootstrap CIs + permutation nulls) lives in one Resampler,
so every measure returns an estimate, a CI, and a four-state equivalence verdict
from a single code path.

Design contracts
----------------
Data           : Sample(human, model, item_ids?, rater_matrix?, predictive?)
Pairing        : UNPAIRED (generator, two populations) | PAIRED (judge, shared items).
Geometry       : SIGNED_DIFFERENCE | NONNEG_DISTANCE | AGREEMENT -- drives the verdict.
Verdict        : EQUIVALENT | DIFFERENT | TRIVIALLY_DIFFERENT | INDETERMINATE.

Optional Sample extensions
--------------------------
rater_matrix   : (n_items, n_raters) human-rater scores; NaNs allowed. Reliability gate.
predictive     : (n_items, n_draws) predictive samples per item. Calibration (CRPS / PIT).
                 Observations for calibration are ``human`` (item-aligned).

Dependencies: numpy only (portable; no scipy required).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Value objects                                                               #
# --------------------------------------------------------------------------- #
class Pairing(Enum):
    UNPAIRED = "unpaired"   # generator: two independent populations of artifacts
    PAIRED = "paired"       # judge: human[i] and model[i] score the same item_ids[i]


class Geometry(Enum):
    SIGNED_DIFFERENCE = "signed_difference"   # null 0, symmetric band (-d, +d)
    NONNEG_DISTANCE = "nonneg_distance"       # >= 0, match iff CI upper bound < d
    AGREEMENT = "agreement"                   # high good, adequate iff CI lower > gamma


class Verdict(Enum):
    EQUIVALENT = "equivalent"                 # provably negligible difference / adequate
    DIFFERENT = "different"                   # provably non-negligible / inadequate
    TRIVIALLY_DIFFERENT = "trivially_diff"    # detectable but negligibly small
    INDETERMINATE = "indeterminate"           # underpowered: cannot conclude either way


class MeasurementLevel(Enum):
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    COUNT = "count"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class Sample:
    """Canonical input for one (metric, framework, model-vs-human) comparison.

    Optional extensions (backward-compatible defaults):
      rater_matrix : (n_items, n_raters) for reliability (Krippendorff α). NaNs = missing.
      predictive   : (n_items, n_draws) predictive draws per item for CRPS / PIT.
                     Convention: row i are draws for item i; observations are ``human``.
    """
    human: np.ndarray
    model: np.ndarray
    item_ids: Optional[np.ndarray] = None   # required for PAIRED analyses
    rater_matrix: Optional[np.ndarray] = None
    predictive: Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, "human", np.asarray(self.human, float).ravel())
        object.__setattr__(self, "model", np.asarray(self.model, float).ravel())
        if self.item_ids is not None:
            object.__setattr__(self, "item_ids", np.asarray(self.item_ids).ravel())
        if self.rater_matrix is not None:
            R = np.asarray(self.rater_matrix, float)
            if R.ndim != 2:
                raise ValueError("rater_matrix must be 2-D with shape (n_items, n_raters).")
            object.__setattr__(self, "rater_matrix", R)
        if self.predictive is not None:
            P = np.asarray(self.predictive, float)
            if P.ndim != 2:
                raise ValueError(
                    "predictive must be 2-D with shape (n_items, n_draws)."
                )
            object.__setattr__(self, "predictive", P)

    def require_paired(self):
        if len(self.human) != len(self.model):
            raise ValueError("Paired analysis requires equal-length human/model arrays.")
        if self.item_ids is None:
            raise ValueError("Paired analysis requires item_ids.")
        if len(self.item_ids) != len(self.human):
            raise ValueError("item_ids length must match human/model arrays.")

    def require_predictive(self):
        if self.predictive is None:
            raise ValueError("Calibration measures require sample.predictive.")
        if self.predictive.shape[0] != len(self.human):
            raise ValueError(
                "predictive rows must match len(human) (one draw vector per item)."
            )

    def require_rater_matrix(self):
        if self.rater_matrix is None:
            raise ValueError("Reliability measures require sample.rater_matrix.")

    @property
    def pooled_sd(self) -> float:
        return float(np.sqrt(0.5 * (self.human.var(ddof=1) + self.model.var(ddof=1))))


@dataclass(frozen=True)
class MeasureResult:
    name: str
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    margin: float
    verdict: Verdict
    p_value: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {
            "measure": self.name, "estimate": self.estimate,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "ci_level": self.ci_level, "margin": self.margin,
            "p_value": self.p_value, "verdict": self.verdict.value,
        }
        d.update(self.extra)
        return d


# --------------------------------------------------------------------------- #
# Resampling engine (BCa bootstrap CIs + permutation nulls)                    #
# --------------------------------------------------------------------------- #
@dataclass
class Resampler:
    n_boot: int = 2000
    n_perm: int = 2000
    ci_level: float = 0.95
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    # --- bootstrap replicates -------------------------------------------- #
    def _boot_replicates(self, stat, h, m, pairing) -> np.ndarray:
        reps = np.empty(self.n_boot)
        n, k = len(h), len(m)
        for b in range(self.n_boot):
            if pairing is Pairing.PAIRED:
                idx = self.rng.integers(0, n, n)
                reps[b] = stat(h[idx], m[idx])
            else:
                reps[b] = stat(h[self.rng.integers(0, n, n)],
                               m[self.rng.integers(0, k, k)])
        return reps

    # --- jackknife for BCa acceleration ---------------------------------- #
    def _jackknife(self, stat, h, m, pairing) -> np.ndarray:
        if pairing is Pairing.PAIRED:
            n = len(h)
            return np.array([stat(np.delete(h, i), np.delete(m, i)) for i in range(n)])
        vals = []
        for i in range(len(h)):
            vals.append(stat(np.delete(h, i), m))
        for j in range(len(m)):
            vals.append(stat(h, np.delete(m, j)))
        return np.array(vals)

    def bca_ci(self, stat, h, m, pairing):
        theta = stat(h, m)
        boot = self._boot_replicates(stat, h, m, pairing)
        jack = self._jackknife(stat, h, m, pairing)
        return self._bca_from_replicates(theta, boot, jack)

    def bca_ci_rows(self, stat, data: np.ndarray):
        """BCa CI resampling the first axis of ``data`` (e.g. items)."""
        data = np.asarray(data)
        n = data.shape[0]
        theta = stat(data)
        boot = np.empty(self.n_boot)
        for b in range(self.n_boot):
            idx = self.rng.integers(0, n, n)
            boot[b] = stat(data[idx])
        jack = np.array([stat(np.delete(data, i, axis=0)) for i in range(n)])
        return self._bca_from_replicates(theta, boot, jack)

    def _bca_from_replicates(self, theta, boot, jack):
        boot = np.asarray(boot, float)
        jack = np.asarray(jack, float)
        boot_f = boot[np.isfinite(boot)]
        jack_f = jack[np.isfinite(jack)]
        if not np.isfinite(theta) or len(boot_f) < 2:
            return float(theta), float("nan"), float("nan")

        prop = np.mean(boot_f < theta)
        prop = min(max(prop, 1e-6), 1 - 1e-6)
        z0 = _norm_ppf(prop)
        if len(jack_f) >= 2:
            jbar = jack_f.mean()
            num = np.sum((jbar - jack_f) ** 3)
            den = 6.0 * (np.sum((jbar - jack_f) ** 2) ** 1.5) + 1e-12
            a = num / den
        else:
            a = 0.0

        alpha = 1 - self.ci_level
        z_lo, z_hi = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)
        lo = _norm_cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
        hi = _norm_cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
        ci_low, ci_high = np.quantile(boot_f, [lo, hi])
        return float(theta), float(ci_low), float(ci_high)

    # --- permutation null (difference test) ------------------------------ #
    def permutation_p(self, stat, h, m, pairing, one_sided=False):
        obs = stat(h, m)
        count = 0
        if pairing is Pairing.PAIRED:
            diff = h - m
            for _ in range(self.n_perm):
                signs = self.rng.choice([-1.0, 1.0], size=len(diff))
                d = signs * diff
                rep = stat(0.5 * (h + m) + d / 2, 0.5 * (h + m) - d / 2)
                count += (rep >= obs) if one_sided else (abs(rep) >= abs(obs))
        else:
            pool = np.concatenate([h, m])
            n = len(h)
            for _ in range(self.n_perm):
                self.rng.shuffle(pool)
                rep = stat(pool[:n], pool[n:])
                count += (rep >= obs) if one_sided else (abs(rep) >= abs(obs))
        return (1 + count) / (1 + self.n_perm)

    def permutation_p_shuffle_obs(self, stat_obs_pred, obs, pred, one_sided=True):
        """Null: shuffle observations against fixed predictive rows (calibration)."""
        observed = stat_obs_pred(obs, pred)
        count = 0
        for _ in range(self.n_perm):
            shuffled = self.rng.permutation(obs)
            rep = stat_obs_pred(shuffled, pred)
            count += (rep >= observed) if one_sided else (abs(rep) >= abs(observed))
        return (1 + count) / (1 + self.n_perm)


# small self-contained normal helpers (avoid scipy) ------------------------- #
def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))

def _norm_ppf(p):
    # Acklam's rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------- #
# ABC 1: the statistics plug                                                   #
# --------------------------------------------------------------------------- #
class StatisticalMeasure(ABC):
    """Subclass to add a comparison. Implement statistic() + default_margin();
    set `pairing` and `geometry`. The verdict/CI machinery is inherited."""

    pairing: Pairing = Pairing.UNPAIRED
    geometry: Geometry = Geometry.SIGNED_DIFFERENCE
    # Pipeline gates for Sample extensions (human/model arms unchanged by default).
    requires_rater_matrix: bool = False
    requires_predictive: bool = False

    def __init__(self, alpha: float = 0.05, margin: Optional[float] = None):
        self.alpha = alpha
        self._margin_override = margin

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def statistic(self, human: np.ndarray, model: np.ndarray) -> float:
        """Point estimate of the measure on one (resampled) sample."""

    @abstractmethod
    def default_margin(self, sample: Sample) -> float:
        """Equivalence margin / adequacy threshold on THIS measure's own scale.
        Override per benchmark on substantive grounds -- do not trust the default."""

    # -- template method: same for every measure -------------------------- #
    def evaluate(self, sample: Sample, rs: Resampler) -> MeasureResult:
        if self.pairing is Pairing.PAIRED:
            sample.require_paired()
        margin = self._margin_override if self._margin_override is not None \
            else self.default_margin(sample)

        est, lo, hi = rs.bca_ci(self.statistic, sample.human, sample.model, self.pairing)
        p = None
        if self.geometry is not Geometry.AGREEMENT:
            one_sided = self.geometry is Geometry.NONNEG_DISTANCE
            p = rs.permutation_p(self.statistic, sample.human, sample.model,
                                 self.pairing, one_sided=one_sided)
        verdict = self._verdict(lo, hi, p, margin)
        return MeasureResult(self.name, est, lo, hi, rs.ci_level, margin, verdict, p)

    def _verdict(self, lo, hi, p, margin) -> Verdict:
        sig = (p is not None) and (p < self.alpha)
        if self.geometry is Geometry.SIGNED_DIFFERENCE:
            equivalent = (lo > -margin) and (hi < margin)
            if equivalent and not sig: return Verdict.EQUIVALENT
            if equivalent and sig:     return Verdict.TRIVIALLY_DIFFERENT
            if (not equivalent) and sig: return Verdict.DIFFERENT
            return Verdict.INDETERMINATE
        if self.geometry is Geometry.NONNEG_DISTANCE:
            if hi < margin and not sig: return Verdict.EQUIVALENT
            if hi < margin and sig:     return Verdict.TRIVIALLY_DIFFERENT
            if lo > margin:             return Verdict.DIFFERENT
            return Verdict.INDETERMINATE
        # AGREEMENT: high good, threshold gamma = margin
        if lo > margin:  return Verdict.EQUIVALENT       # adequate
        if hi < margin:  return Verdict.DIFFERENT        # provably inadequate
        return Verdict.INDETERMINATE


# --------------------------------------------------------------------------- #
# Worked measures -- one per geometry, covering generator + judge              #
# --------------------------------------------------------------------------- #
def _mean_pairwise_abs_1d(x: np.ndarray) -> float:
    """Mean |x_i - x_j| over all pairs including i=j. O(n log n) via sort."""
    x = np.sort(np.asarray(x, float).ravel())
    n = len(x)
    if n == 0:
        return 0.0
    # sum_{i,j} |x_i-x_j| = 2 * sum_i (2*i - n + 1) * x_{(i)}  (0-based i)
    coef = 2 * np.arange(n) - n + 1
    return float(2.0 * np.dot(coef, x) / (n * n))


def _mean_cross_abs_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |a_i - b_j| over all cross pairs. O((n+m) log) via sort + scan."""
    a = np.sort(np.asarray(a, float).ravel())
    b = np.sort(np.asarray(b, float).ravel())
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    pref = np.concatenate([[0.0], np.cumsum(b)])
    j = 0
    total = 0.0
    for i in range(n):
        while j < m and b[j] < a[i]:
            j += 1
        left = pref[j]
        right = pref[m] - pref[j]
        total += j * a[i] - left + right - (m - j) * a[i]
    return float(total / (n * m))


class EnergyDistance(StatisticalMeasure):
    """Marginal-shape default. Ties-safe, honest permutation null. Generator.

    Uses the O(n log n) 1-D energy-distance form (scalar scores); avoids
    materializing full pairwise distance matrices.
    """
    pairing = Pairing.UNPAIRED
    geometry = Geometry.NONNEG_DISTANCE

    @property
    def name(self): return "energy_distance"

    def statistic(self, h, m):
        return float(
            2 * _mean_cross_abs_1d(h, m)
            - _mean_pairwise_abs_1d(h)
            - _mean_pairwise_abs_1d(m)
        )

    def default_margin(self, s: Sample):
        return 0.2 * s.pooled_sd   # "small" in pooled-SD units; OVERRIDE per benchmark


class LogVarianceRatio(StatisticalMeasure):
    """Dispersion / mode-collapse axis. Symmetric on log scale. Generator."""
    pairing = Pairing.UNPAIRED
    geometry = Geometry.SIGNED_DIFFERENCE

    @property
    def name(self): return "log_var_ratio"

    def statistic(self, h, m):
        return float(np.log(m.var(ddof=1) + 1e-12) - np.log(h.var(ddof=1) + 1e-12))

    def default_margin(self, s: Sample):
        return float(np.log(1.5))  # within 1.5x dispersion counts as equivalent


class HedgesG(StatisticalMeasure):
    """Location, standardized + small-sample corrected. Generator."""
    pairing = Pairing.UNPAIRED
    geometry = Geometry.SIGNED_DIFFERENCE

    @property
    def name(self): return "hedges_g"

    def statistic(self, h, m):
        n1, n2 = len(h), len(m)
        sp = np.sqrt(((n1-1)*h.var(ddof=1) + (n2-1)*m.var(ddof=1)) / (n1+n2-2) + 1e-12)
        d = (m.mean() - h.mean()) / sp
        J = 1 - 3 / (4*(n1+n2) - 9)   # Hedges correction
        return float(J * d)

    def default_margin(self, s: Sample):
        return 0.2   # Cohen "small" -- a convention; replace with a domain margin


class ConcordanceCC(StatisticalMeasure):
    """Lin's CCC. Penalizes shift AND correlation loss. Judge (paired)."""
    pairing = Pairing.PAIRED
    geometry = Geometry.AGREEMENT

    @property
    def name(self): return "lins_ccc"

    def statistic(self, h, m):
        mh, mm = h.mean(), m.mean()
        vh, vm = h.var(), m.var()
        cov = np.mean((h - mh) * (m - mm))
        return float(2*cov / (vh + vm + (mh - mm)**2 + 1e-12))

    def default_margin(self, s: Sample):
        return 0.80   # "substantial" agreement threshold gamma


# --------------------------------------------------------------------------- #
# Reliability gate + calibration (Sample extensions)                           #
# --------------------------------------------------------------------------- #
def krippendorff_alpha(rater_matrix: np.ndarray, level: str = "interval") -> float:
    """Krippendorff's α on (n_items, n_raters); NaNs mark missing ratings.

    Levels: ``nominal``, ``ordinal``, ``interval`` (default; continuous scores).
    Numpy-only coincidence-matrix implementation.

    Note: for continuous multi-rater reliability, ICC(2,k) is a common alternative
    (not implemented here; prefer α when missingness / mixed levels matter).
    """
    data = np.asarray(rater_matrix, float)
    if data.ndim != 2:
        raise ValueError("rater_matrix must be 2-D (n_items, n_raters).")
    level = level.lower()
    if level not in ("nominal", "ordinal", "interval"):
        raise ValueError("level must be one of: nominal, ordinal, interval")

    mask = ~np.isnan(data)
    n_per_unit = mask.sum(axis=1)
    if np.all(n_per_unit < 2):
        return float("nan")

    values = np.unique(data[mask])
    value_map = {float(v): i for i, v in enumerate(values)}
    v = len(values)
    o = np.zeros((v, v))
    for u in range(data.shape[0]):
        mu = int(n_per_unit[u])
        if mu < 2:
            continue
        idx = [value_map[float(x)] for x in data[u, mask[u]]]
        inv = 1.0 / (mu - 1)
        for a in range(mu):
            for b in range(mu):
                if a != b:
                    o[idx[a], idx[b]] += inv

    n_c = o.sum(axis=1)
    n = float(n_c.sum())
    if n <= 1:
        return float("nan")

    delta = np.zeros((v, v))
    if level == "nominal":
        delta[:, :] = 1.0
        np.fill_diagonal(delta, 0.0)
    elif level == "ordinal":
        for i in range(v):
            for j in range(v):
                if i == j:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                cum = n_c[lo:hi + 1].sum()
                delta[i, j] = (cum - (n_c[i] + n_c[j]) / 2.0) ** 2
    else:  # interval
        for i in range(v):
            for j in range(v):
                delta[i, j] = (values[i] - values[j]) ** 2

    Do = float((o * delta).sum() / n)
    De = float((np.outer(n_c, n_c) * delta).sum() / (n * (n - 1)))
    if De < 1e-15:
        return 1.0 if Do < 1e-15 else 0.0
    return float(1.0 - Do / De)


def mean_crps(obs: np.ndarray, draws: np.ndarray) -> float:
    """Mean CRPS from predictive draws vs observations (item-aligned).

    For each item: CRPS = E|X - y| - 0.5 E|X - X'| with X, X' iid from the
    empirical predictive (columns of ``draws``).

    E|X - X'| uses the O(n_draws log n_draws) sorted closed form per item.
    """
    obs = np.asarray(obs, float).ravel()
    draws = np.asarray(draws, float)
    if draws.ndim != 2 or draws.shape[0] != len(obs):
        raise ValueError("draws must have shape (n_items, n_draws) matching obs.")
    term1 = np.mean(np.abs(draws - obs[:, None]), axis=1)
    term2 = np.array([_mean_pairwise_abs_1d(draws[i]) for i in range(len(obs))])
    return float(np.mean(term1 - 0.5 * term2))


def pit_values(obs: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Probability integral transform: empirical CDF of draws at each obs."""
    obs = np.asarray(obs, float).ravel()
    draws = np.asarray(draws, float)
    if draws.ndim != 2 or draws.shape[0] != len(obs):
        raise ValueError("draws must have shape (n_items, n_draws) matching obs.")
    # Mid-rank for ties: (#{X < y} + 0.5 #{X = y}) / M
    lt = np.sum(draws < obs[:, None], axis=1)
    eq = np.sum(draws == obs[:, None], axis=1)
    return (lt + 0.5 * eq) / draws.shape[1]


def pit_ks_distance(pit: np.ndarray) -> float:
    """Kolmogorov–Smirnov distance of PIT samples to Uniform[0, 1]."""
    p = np.sort(np.asarray(pit, float).ravel())
    n = len(p)
    if n == 0:
        return float("nan")
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(ecdf_hi - p)), np.max(np.abs(ecdf_lo - p))))


class KrippendorffAlpha(StatisticalMeasure):
    """Inter-rater reliability (human gate) via Krippendorff's α.

    Uses ``sample.rater_matrix`` only — human/model arms are ignored.
    Geometry AGREEMENT: adequate iff CI lower bound > gamma (``margin``).
    """
    pairing = Pairing.PAIRED          # resample items (rows); no model arm needed
    geometry = Geometry.AGREEMENT
    requires_rater_matrix = True

    def __init__(self, alpha: float = 0.05, margin: Optional[float] = None,
                 level: str = "interval"):
        super().__init__(alpha=alpha, margin=margin)
        self.level = level

    @property
    def name(self):
        return "krippendorff_alpha"

    def statistic(self, human, model):
        raise NotImplementedError(
            "KrippendorffAlpha.evaluate uses sample.rater_matrix; "
            "statistic(human, model) is not applicable."
        )

    def default_margin(self, s: Sample):
        return 0.80   # substantial reliability threshold gamma

    def evaluate(self, sample: Sample, rs: Resampler) -> MeasureResult:
        sample.require_rater_matrix()
        R = sample.rater_matrix
        margin = self._margin_override if self._margin_override is not None \
            else self.default_margin(sample)
        level = self.level

        def stat(block):
            return krippendorff_alpha(block, level=level)

        est, lo, hi = rs.bca_ci_rows(stat, R)
        verdict = self._verdict(lo, hi, None, margin)
        return MeasureResult(
            self.name, est, lo, hi, rs.ci_level, margin, verdict, None,
            extra={"reliability_level": level, "n_items": int(R.shape[0]),
                   "n_raters": int(R.shape[1])},
        )


class CRPS(StatisticalMeasure):
    """Continuous Ranked Probability Score (mean over items).

    Observations: ``sample.human``. Predictive draws: ``sample.predictive``
    with shape (n_items, n_draws). Geometry NONNEG_DISTANCE (lower better).
    """
    pairing = Pairing.PAIRED
    geometry = Geometry.NONNEG_DISTANCE
    requires_predictive = True

    @property
    def name(self):
        return "crps"

    def statistic(self, human, model):
        raise NotImplementedError(
            "CRPS.evaluate uses sample.human + sample.predictive."
        )

    def default_margin(self, s: Sample):
        # small absolute error relative to outcome scale
        scale = float(np.std(s.human, ddof=1) + 1e-12)
        return 0.2 * scale

    def evaluate(self, sample: Sample, rs: Resampler) -> MeasureResult:
        sample.require_predictive()
        obs, pred = sample.human, sample.predictive
        margin = self._margin_override if self._margin_override is not None \
            else self.default_margin(sample)

        # Pack obs as column 0 so row-bootstrap keeps item alignment.
        block = np.concatenate([obs[:, None], pred], axis=1)

        def stat_block(b):
            return mean_crps(b[:, 0], b[:, 1:])

        est, lo, hi = rs.bca_ci_rows(stat_block, block)
        p = rs.permutation_p_shuffle_obs(mean_crps, obs, pred, one_sided=True)
        verdict = self._verdict(lo, hi, p, margin)
        return MeasureResult(
            self.name, est, lo, hi, rs.ci_level, margin, verdict, p,
            extra={"n_items": int(len(obs)), "n_draws": int(pred.shape[1])},
        )


class PITUniformity(StatisticalMeasure):
    """PIT uniformity diagnostic: KS distance of PIT values to Uniform[0, 1].

    Observations: ``sample.human``. Predictive draws: ``sample.predictive``.
    Scalar summary suitable for equivalence (NONNEG_DISTANCE). Histogram and
    mean-absolute ECDF deviation live in ``MeasureResult.extra``.
    """
    pairing = Pairing.PAIRED
    geometry = Geometry.NONNEG_DISTANCE
    requires_predictive = True

    def __init__(self, alpha: float = 0.05, margin: Optional[float] = None,
                 n_bins: int = 10):
        super().__init__(alpha=alpha, margin=margin)
        self.n_bins = n_bins

    @property
    def name(self):
        return "pit_uniformity"

    def statistic(self, human, model):
        raise NotImplementedError(
            "PITUniformity.evaluate uses sample.human + sample.predictive."
        )

    def default_margin(self, s: Sample):
        # rough KS critical-value scale for moderate n; OVERRIDE per study
        n = max(len(s.human), 1)
        return 1.36 / np.sqrt(n)

    def evaluate(self, sample: Sample, rs: Resampler) -> MeasureResult:
        sample.require_predictive()
        obs, pred = sample.human, sample.predictive
        margin = self._margin_override if self._margin_override is not None \
            else self.default_margin(sample)

        block = np.concatenate([obs[:, None], pred], axis=1)

        def stat_block(b):
            return pit_ks_distance(pit_values(b[:, 0], b[:, 1:]))

        def stat_obs_pred(o, p):
            return pit_ks_distance(pit_values(o, p))

        est, lo, hi = rs.bca_ci_rows(stat_block, block)
        p = rs.permutation_p_shuffle_obs(stat_obs_pred, obs, pred, one_sided=True)
        verdict = self._verdict(lo, hi, p, margin)

        pit = pit_values(obs, pred)
        hist, edges = np.histogram(pit, bins=self.n_bins, range=(0.0, 1.0), density=False)
        sorted_pit = np.sort(pit)
        n = len(sorted_pit)
        ecdf = np.arange(1, n + 1) / n
        mad = float(np.mean(np.abs(ecdf - sorted_pit))) if n else float("nan")

        return MeasureResult(
            self.name, est, lo, hi, rs.ci_level, margin, verdict, p,
            extra={
                "pit_mad": mad,
                "pit_histogram": hist.astype(int).tolist(),
                "pit_bin_edges": edges.tolist(),
                "n_items": int(n),
                "n_draws": int(pred.shape[1]),
            },
        )


# --------------------------------------------------------------------------- #
# ABC 2: the domain plug                                                       #
# --------------------------------------------------------------------------- #
class CreativityMetric(ABC):
    """Subclass per creativity metric. Mostly a descriptor: name, framework, and
    measurement level (which gates valid measures + margin scales). Implement
    score() only if you want the pipeline to compute scores from raw artifacts;
    otherwise feed a Sample of pre-computed scores directly."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def framework(self) -> str: ...

    @property
    @abstractmethod
    def level(self) -> MeasurementLevel: ...

    @property
    def bounds(self) -> Optional[tuple]:
        return None

    def score(self, artifact, context=None) -> float:
        raise NotImplementedError("Provide scores directly, or implement score().")


# --------------------------------------------------------------------------- #
# Pipeline + multiplicity                                                      #
# --------------------------------------------------------------------------- #
def benjamini_yekutieli(pvals):
    """FDR control valid under ARBITRARY dependence (metrics correlate)."""
    p = np.asarray(pvals, float)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    q = p[mask]
    n = len(q)
    if n == 0:
        return out
    c = np.sum(1.0 / np.arange(1, n + 1))          # harmonic penalty
    order = np.argsort(q)
    ranks = np.arange(1, n + 1)
    adj = q[order] * n * c / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    res = np.empty(n); res[order] = adj
    out[mask] = res
    return out


@dataclass
class ComparisonReport:
    metric: str
    framework: str
    results: list
    def rows(self):
        return [{"metric": self.metric, "framework": self.framework, **r.as_row()}
                for r in self.results]


def report_to_dict(report: ComparisonReport, model: Optional[str] = None) -> dict:
    """Dump a ComparisonReport into the report.json shape (schema: report.schema.json)."""
    rows = report.rows()
    if model is not None:
        for row in rows:
            row["model"] = model
    return {
        "metric": report.metric,
        "framework": report.framework,
        "model": model,
        "results": rows,
    }


class ComparisonPipeline:
    def __init__(self, measures, resampler=None, alpha=0.05):
        self.measures = list(measures)
        self.rs = resampler or Resampler()
        self.alpha = alpha
        # Pipeline-level alpha overrides each measure's significance threshold.
        for msr in self.measures:
            msr.alpha = alpha

    def _applicable(self, metric, sample):
        paired = (sample.item_ids is not None
                  and len(sample.human) == len(sample.model))
        has_raters = sample.rater_matrix is not None
        has_pred = (sample.predictive is not None
                    and sample.predictive.shape[0] == len(sample.human))
        for msr in self.measures:
            if getattr(msr, "requires_rater_matrix", False):
                if has_raters:
                    yield msr
                continue
            if getattr(msr, "requires_predictive", False):
                if has_pred:
                    yield msr
                continue
            if msr.pairing is Pairing.PAIRED and not paired:
                continue
            yield msr

    def run(self, metric: CreativityMetric, sample: Sample) -> ComparisonReport:
        results = [msr.evaluate(sample, self.rs)
                   for msr in self._applicable(metric, sample)]
        # BY correction across this metric's difference-tests
        pvals = [r.p_value if r.p_value is not None else np.nan for r in results]
        padj = benjamini_yekutieli(pvals)
        results = [MeasureResult(**{**r.__dict__,
                                    "extra": {**r.extra, "p_adj_by": None if np.isnan(pa)
                                              else float(pa)}})
                   for r, pa in zip(results, padj)]
        return ComparisonReport(metric.name, metric.framework, results)


if __name__ == "__main__":
    # smoke test: mode-collapsed model, matched mean
    rng = np.random.default_rng(1)
    human = np.clip(rng.normal(0.6, 0.15, 120), 0, 1)
    model = np.clip(rng.normal(0.6, 0.07, 100), 0, 1)   # half the spread

    class Novelty(CreativityMetric):
        name = "novelty"; framework = "SAUCE"
        level = MeasurementLevel.CONTINUOUS
        bounds = (0.0, 1.0)

    pipe = ComparisonPipeline(
        measures=[EnergyDistance(), LogVarianceRatio(), HedgesG()],
        resampler=Resampler(n_boot=800, n_perm=800, seed=7),
    )
    rep = pipe.run(Novelty(), Sample(human, model))
    for row in rep.rows():
        print(f"{row['measure']:16s} est={row['estimate']:+.3f} "
              f"CI=[{row['ci_low']:+.3f},{row['ci_high']:+.3f}] "
              f"margin={row['margin']:.3f} -> {row['verdict']}")
