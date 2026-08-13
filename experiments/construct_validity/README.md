# Construct validity suite (E0–E5)

Executable checks that the frozen `poetry_v2` metric stack measures creativity-related structure rather than length/fluency/domain artifacts.

Reuses `src/creativegainbench` + `artifacts/poetry_v2` (no codebook retrain).  
δ_D is fit **only** from a constructed negative bank — never from z* or human/model scores.

## Layout

```
experiments/construct_validity/
  config/db.yaml
  db/migrations/          # delta_d_calibration, delta_d_thresholds, experiment_results
  metrics/pipeline.py     # frozen poetry_v2 loader + feasibility_bit
  calibration/            # negative bank + one-sided quantile δ_D
  experiments/e0..e5.py
  experiments/run_all.py
  analysis/report_builder.py
  results/REPORT.md
```

## Run

```bash
# Full suite (builds negative bank — CPU, ~tens of minutes)
python experiments/construct_validity/experiments/run_all.py

# Skip expensive E5 judge calls
python experiments/construct_validity/experiments/run_all.py --skip-e5

# Resume after bank already built
python experiments/construct_validity/experiments/run_all.py --skip-bank --from e0
```

## Experiments

| ID | Purpose |
|----|---------|
| E0 | Orphan scores / domain×model coverage |
| E1 | Known-groups (provenance, domain order, legacy stability) |
| E2 | Exact-copy / anti-padding / monotonicity |
| E3 | δ_D acceptance + length corr + z*-stratified joint gate (report) |
| E4 | corr(CUE, R_D) band |
| E5 | Judge inventiveness vs joint gate (report only) |
