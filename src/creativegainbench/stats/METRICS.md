# Creativity Metrics — Registry Template

Fill one block per benchmark metric. Margins are on the **StatisticalMeasure**
scale (not necessarily the raw metric scale). See `SPEC.md` for Pairing /
Geometry / Verdict.

Legend for stubs: `[ ]` = not yet wired into `CreativityMetric` subclasses /
pipeline configs.

---

## StatisticalMeasures (worked)

| Measure | `name` | Pairing | Geometry | Default margin (convention) |
|---------|--------|---------|----------|-----------------------------|
| Energy distance | `energy_distance` | UNPAIRED | NONNEG_DISTANCE | `0.2 * pooled_sd` |
| Log variance ratio | `log_var_ratio` | UNPAIRED | SIGNED_DIFFERENCE | `log(1.5)` |
| Hedges' g | `hedges_g` | UNPAIRED | SIGNED_DIFFERENCE | `0.2` |
| Lin's CCC | `lins_ccc` | PAIRED | AGREEMENT | `0.80` |
| Paired mean difference | `paired_mean_diff` | PAIRED | SIGNED_DIFFERENCE | `0.0` (CI excluding 0 ⇒ DIFFERENT) |
| Spearman ρ | `spearman_rho` | PAIRED | AGREEMENT | `0.80` (pass = BCa CI lower > 0.80) |
| Krippendorff α | `krippendorff_alpha` | items (rater matrix) | AGREEMENT | `0.80` |
| CRPS | `crps` | PAIRED by item | NONNEG_DISTANCE | `0.2 * sd(human)` |
| PIT uniformity | `pit_uniformity` | PAIRED by item | NONNEG_DISTANCE | `1.36 / sqrt(n)` |

Reliability / calibration require Sample extensions (`rater_matrix`, `predictive`).

---

## Benchmark metrics (stubs)

Discovered under `src/creativegainbench/metrics/`. Complete level / bounds /
recommended measures / margins when registering each as a `CreativityMetric`.

### CUE — Creative Update Efficiency

- **Module:** `metrics/cue.py` (`compute_cue`, `CUEModel`)
- **Framework:** `[ ]` (Lean-aligned CUE)
- **Level:** `[ ]` continuous (nonnegative efficiency)
- **Bounds:** `[ ]` typically ≥ 0; upper unbound or study-specific
- **Recommended measures:** `[ ]` energy_distance, hedges_g, log_var_ratio; lins_ccc if paired judge; **paired_mean_diff** (E7 yoked CUE contrasts); **spearman_rho** + lins_ccc (E5a receiver pairs)
- **Margins:** `[ ]` override in `config.yaml`
- **Notes:** Brier-delta / bit-length; receiver-grounded

### Structural novelty \(R_D\)

- **Module:** `metrics/structural_novelty.py`
- **Framework:** `[ ]`
- **Level:** `[ ]` continuous
- **Bounds:** `[ ]`
- **Recommended measures:** `[ ]`
- **Margins:** `[ ]`
- **Notes:** KenLM idea-symbol compression cost difference vs probe set

### Receiver expansion \(R_B\)

- **Module:** `metrics/receiver_expansion.py`
- **Framework:** `[ ]`
- **Level:** `[ ]` continuous (normalized entropy-like)
- **Bounds:** `[ ]` often in \([0, 1]\) when divided by `log|W|`
- **Recommended measures:** `[ ]`
- **Margins:** `[ ]`
- **Notes:** Soft clustering over frozen idea codebook

### Interaction gain \(G_k\)

- **Module:** `metrics/interaction_gain.py`
- **Framework:** `[ ]` (MAS / multi-agent)
- **Level:** `[ ]` continuous
- **Bounds:** `[ ]`
- **Recommended measures:** `[ ]`
- **Margins:** `[ ]`
- **Notes:** Joint entropy − max single-agent entropy

### Trajectory diagnostics (Step-CUE / Diverge–Converge)

- **Module:** `metrics/trajectory.py`
- **Framework:** `[ ]`
- **Level:** `[ ]` continuous / curve summaries
- **Bounds:** `[ ]`
- **Recommended measures:** `[ ]`
- **Margins:** `[ ]`
- **Notes:** Not a single scalar by default; register derived summaries explicitly

### CUE belief receiver

- **Module:** `metrics/cue_receiver.py`
- **Framework:** `[ ]`
- **Level:** `[ ]`
- **Bounds:** `[ ]`
- **Recommended measures:** `[ ]`
- **Margins:** `[ ]`
- **Notes:** Receiver-side belief updates supporting CUE

### KenLM compressor (support)

- **Module:** `metrics/kenlm_compressor.py`
- **Framework:** n/a (infrastructure for structural novelty)
- **Level:** n/a
- **Notes:** Not a creativity score by itself

---

## Per-metric registration template

Copy for each filled metric:

```yaml
# config.yaml fragment
metrics:
  <metric_name>:
    framework: <framework>
    level: continuous   # continuous | ordinal | count | categorical
    bounds: [null, null]
    measures:
      - energy_distance
      - log_var_ratio
      - hedges_g
      # - lins_ccc          # if paired judge
      # - krippendorff_alpha  # if rater_matrix
      # - crps                # if predictive
      # - pit_uniformity
    margins:
      energy_distance: null     # null => default_margin(sample)
      log_var_ratio: null
      hedges_g: null
      lins_ccc: null
      spearman_rho: null
      paired_mean_diff: 0.0
      krippendorff_alpha: null
      crps: null
      pit_uniformity: null
```
