# Science-in-the-loop artifacts

Working directory for experiment diagnostics consumed by the science-in-the-loop
crew (`.cursor/skills/science-in-the-loop/`).

| Path | Purpose |
|------|---------|
| `embedding_pca_latest.md` | **Stable pointer** to latest human-vs-LLM embedding PCA report |
| `runs/EMB-PCA-01/` | PCA / linear-probe separability study (script + metrics + scatter) |
| `EXPERIMENT_LOG.md` | Append-only detailed log of runs and hypotheses |
| `snapshot_latest.md` / `.json` | Latest construct-validity / DB snapshot |
| `runs/<hypothesis-id>/` | Per-hypothesis experimenter artifacts |

## EMB-PCA-01 (repro)

Requires Postgres `poems` with eval `generations` for the target model and
frozen `poetry_v2` MiniLM + boundary artifacts:

```bash
podman start poems-pg
.venv/bin/python experiments/science_loop/runs/EMB-PCA-01/analyze_embedding_pca.py \
  --model gemma2:2b --device cpu --stratified-limit 400
```

Omit `--stratified-limit` (or set `1489`) for the full eval set.
