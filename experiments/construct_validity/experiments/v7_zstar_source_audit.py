"""V7 — z*-source audit: external vs self by model/domain."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.connection import connect, run_migrations  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402
from lib import load_config  # noqa: E402


def _source(payload: dict) -> str | None:
    z = payload.get("z_star_source")
    if z in {"external", "self"}:
        return z
    o = payload.get("outcome_source")
    if o == "external":
        return "external"
    if o in {"self_classify", "self"}:
        return "self"
    return None


def main() -> None:
    run_migrations()
    cfg = load_config()
    run_id = new_run_id()
    metric_cue = cfg.get("metric_version", "poetry_v2") + "_cue"

    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"external": 0, "self": 0, "unknown": 0})
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"external": 0, "self": 0, "unknown": 0})
    total = {"external": 0, "self": 0, "unknown": 0}

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.side, s.payload, p.domain_cluster
            FROM scores s
            JOIN poems p ON p.id = s.poem_id
            WHERE s.metric_version = %s
               OR (s.payload ? 'z_star_source')
               OR (s.payload ? 'outcome_source' AND s.payload ? 'cue')
            """,
            (metric_cue,),
        ).fetchall()

    # Also scan experiment1 CUE JSONL if DB empty (pre-F0.2 runs).
    jsonl_rows: list[tuple] = []
    cue_dir = Path(cfg.get("experiment1_results", ""))
    if not cue_dir.is_absolute():
        cue_dir = (
            Path(__file__).resolve().parents[2] / "experiment1" / "results"
        )
    for path in cue_dir.glob("cue_*.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for side_key in ("human", "llm"):
                payload = rec.get(side_key) or {}
                if "cue" not in payload and "outcome_source" not in payload:
                    continue
                jsonl_rows.append(
                    (f"{path.stem}:{side_key}", payload, rec.get("domain_cluster"))
                )

    for side, payload, domain in list(rows) + jsonl_rows:
        if isinstance(payload, str):
            payload = json.loads(payload)
        src = _source(payload) or "unknown"
        total[src] += 1
        by_model[str(side)][src] += 1
        by_domain[str(domain)][src] += 1

    n = sum(total.values())
    ext_share = total["external"] / n if n else 0.0
    record(
        run_id,
        "V7",
        "zstar_external_share",
        ext_share,
        ext_share >= 0.8 if n else None,
        {"n": n, "counts": total, "by_model": dict(by_model), "by_domain": dict(by_domain)},
    )
    print(json.dumps({"n": n, "counts": total, "external_share": ext_share}, indent=2))
    print("DONE V7")


if __name__ == "__main__":
    main()
