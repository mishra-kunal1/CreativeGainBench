from __future__ import annotations

import uuid
from typing import Any

from db.connection import connect
from db.queries import insert_experiment_result


def new_run_id() -> uuid.UUID:
    return uuid.uuid4()


def record(
    run_id,
    experiment: str,
    check_name: str,
    metric_value: float | None,
    passed: bool | None,
    details: dict[str, Any],
) -> None:
    with connect() as conn:
        insert_experiment_result(
            conn, run_id, experiment, check_name, metric_value, passed, details
        )
    status = "PASS" if passed else ("FAIL" if passed is False else "REPORT")
    print(f"[{experiment}] {check_name}: {status} value={metric_value} {details}", flush=True)
