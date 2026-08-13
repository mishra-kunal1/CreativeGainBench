"""E0 — Data audit: orphan scores and domain×model coverage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.connection import connect, run_migrations  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402
from lib import load_config  # noqa: E402


def main() -> None:
    run_migrations()
    cfg = load_config()
    run_id = new_run_id()
    models = ["gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b"]

    with connect() as conn:
        # Only flag model-side eval scores missing a matching generations row.
        orphans = conn.execute(
            """
            SELECT COUNT(*) FROM scores s
            JOIN poems p ON p.id = s.poem_id
            WHERE s.metric_version = %s
              AND p.split = 'eval'
              AND s.side = ANY(%s)
              AND NOT EXISTS (
                SELECT 1 FROM generations g
                WHERE g.poem_id = s.poem_id AND g.model = s.side
              )
            """,
            (cfg["metric_version"], models),
        ).fetchone()[0]

        cells = conn.execute(
            """
            SELECT p.domain_cluster, s.side, COUNT(*)
            FROM scores s
            JOIN poems p ON p.id = s.poem_id
            WHERE s.metric_version = %s AND p.split = 'eval' AND s.side = ANY(%s)
            GROUP BY 1, 2
            """,
            (cfg["metric_version"], models),
        ).fetchall()

    min_cell = min((n for _, _, n in cells), default=0)
    cell_map = {(d, m): n for d, m, n in cells}

    record(
        run_id,
        "E0",
        "orphan_scores_count",
        float(orphans),
        orphans == 0,
        {"orphans": orphans},
    )
    record(
        run_id,
        "E0",
        "min_cell_n",
        float(min_cell),
        min_cell >= 30,
        {
            "min_cell": min_cell,
            "cells": {f"{d}|{m}": n for (d, m), n in cell_map.items()},
            "threshold": 30,
        },
    )
    print("DONE E0")


if __name__ == "__main__":
    main()
