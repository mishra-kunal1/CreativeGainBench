from __future__ import annotations

from typing import Any

import psycopg


def fetch_domain_splits(conn: psycopg.Connection) -> list[tuple]:
    return conn.execute(
        """
        SELECT domain_cluster, split, COUNT(*)
        FROM poems
        WHERE domain_cluster IS NOT NULL AND split IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()


def fetch_poems_by_split(
    conn: psycopg.Connection, domain: int, split: str
) -> list[tuple]:
    return conn.execute(
        """
        SELECT id, body, author, source, line_count, prompt
        FROM poems
        WHERE domain_cluster = %s AND split = %s
        ORDER BY id
        """,
        (domain, split),
    ).fetchall()


def fetch_eval_scores(
    conn: psycopg.Connection, metric_version: str, side: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT s.poem_id, p.domain_cluster, p.source, p.split, s.side, s.payload
        FROM scores s
        JOIN poems p ON p.id = s.poem_id
        WHERE s.metric_version = %s AND s.side = %s AND p.split = 'eval'
        """,
        (metric_version, side),
    ).fetchall()
    out = []
    for pid, domain, source, split, side_v, payload in rows:
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        out.append(
            {
                "poem_id": str(pid),
                "domain_cluster": domain,
                "source": source,
                "split": split,
                "side": side_v,
                "payload": payload,
            }
        )
    return out


def insert_experiment_result(
    conn: psycopg.Connection,
    run_id,
    experiment: str,
    check_name: str,
    metric_value: float | None,
    passed: bool | None,
    details: dict,
) -> None:
    import json
    import math

    def _sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    if isinstance(metric_value, float) and (
        math.isnan(metric_value) or math.isinf(metric_value)
    ):
        metric_value = None

    conn.execute(
        """
        INSERT INTO experiment_results
            (run_id, experiment, check_name, metric_value, passed, details)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (run_id, check_name) DO UPDATE SET
            metric_value = EXCLUDED.metric_value,
            passed = EXCLUDED.passed,
            details = EXCLUDED.details,
            created_at = now()
        """,
        (
            run_id,
            experiment,
            check_name,
            metric_value,
            passed,
            json.dumps(_sanitize(details)),
        ),
    )


def clear_calibration(conn: psycopg.Connection, domain: int | None = None) -> None:
    if domain is None:
        conn.execute("DELETE FROM delta_d_calibration")
        conn.execute("DELETE FROM delta_d_thresholds")
    else:
        conn.execute(
            "DELETE FROM delta_d_calibration WHERE domain_cluster = %s", (domain,)
        )
        conn.execute(
            "DELETE FROM delta_d_thresholds WHERE domain_cluster = %s", (domain,)
        )
