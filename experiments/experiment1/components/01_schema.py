"""
Create generations table and migrate existing poems.llm_output → gemma2:2b.
Also ensure scores supports model-keyed rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    with psycopg.connect(cfg["db_url"]) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                poem_id UUID NOT NULL REFERENCES poems(id),
                model TEXT NOT NULL,
                output TEXT NOT NULL,
                max_tokens INT NOT NULL DEFAULT 1024,
                temperature REAL NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (poem_id, model)
            );
            CREATE INDEX IF NOT EXISTS idx_generations_model ON generations (model);

            CREATE TABLE IF NOT EXISTS scores (
                poem_id UUID NOT NULL,
                side TEXT NOT NULL,
                metric_version TEXT NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (poem_id, side, metric_version)
            );
            """
        )
        # Migrate gemma2:2b from poems.llm_output if not already present.
        migrated = conn.execute(
            """
            INSERT INTO generations (poem_id, model, output, max_tokens, temperature)
            SELECT id, 'gemma2:2b', llm_output, 1024, 1.0
            FROM poems
            WHERE llm_output IS NOT NULL AND btrim(llm_output) <> ''
            ON CONFLICT (poem_id, model) DO NOTHING
            """
        ).rowcount
        conn.commit()
        counts = conn.execute(
            "SELECT model, COUNT(*) FROM generations GROUP BY model ORDER BY 1"
        ).fetchall()
        eval_n = conn.execute(
            "SELECT COUNT(*) FROM poems WHERE split = %s", (cfg["split"],)
        ).fetchone()[0]
    print(f"migrated_or_skipped_rows={migrated}")
    print(f"eval_poems={eval_n}")
    for model, n in counts:
        print(f"  generations[{model}]={n}")
    print("DONE 01_schema")


if __name__ == "__main__":
    main()
