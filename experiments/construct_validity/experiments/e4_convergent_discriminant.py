"""E4 — corr(R_D, CUE) on frozen experiment1 CUE subsample."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import pearson  # noqa: E402
from db.connection import run_migrations  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402


def main() -> None:
    run_migrations()
    run_id = new_run_id()
    cue_dir = Path(__file__).resolve().parents[2] / "experiment1" / "results"

    xs: list[float] = []
    ys: list[float] = []
    by_model: dict[str, list[tuple[float, float]]] = {}

    for path in sorted(cue_dir.glob("cue_*.jsonl")):
        model = path.stem.replace("cue_", "").replace("_", ":", 1)
        pairs = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for side in ("human", "llm"):
                rd = rec[side].get("r_d_norm")
                cue = rec[side].get("cue")
                if rd is None or cue is None:
                    continue
                xs.append(float(rd))
                ys.append(float(cue))
                pairs.append((float(rd), float(cue)))
        by_model[model] = pairs

    rho = pearson(xs, ys)
    # Pass if moderate band 0.1–0.5 (absolute)
    passed = (not math.isnan(rho)) and (0.1 <= abs(rho) <= 0.5)
    model_rhos = {
        m: pearson([a for a, _ in ps], [b for _, b in ps]) for m, ps in by_model.items() if len(ps) >= 10
    }
    record(
        run_id,
        "E4",
        "cue_rd_correlation",
        None if math.isnan(rho) else float(rho),
        passed,
        {"pooled_r": rho, "n": len(xs), "by_model": model_rhos, "band": [0.1, 0.5]},
    )
    print("DONE E4")


if __name__ == "__main__":
    main()
