"""
Score Edge-CUE / HandoffGain for MAS generations (model keys like pcv:gemma2:2b).

Writes scores rows with sides edge:critic_to_proposer and edge:handoff_gain.
Skips silently when no MAS generations exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import REPO_ROOT, load_config  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from creativegainbench.metrics.edge_cue import handoff_gain_rate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Base model; scores pcv:<model>")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    metric_version = cfg["metric_version"]
    results_dir = cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    bases = [args.model] if args.model else list(cfg["generation"]["models"])
    pcv_models = [f"pcv:{m}" for m in bases]

    with psycopg.connect(cfg["db_url"]) as conn:
        n_pcv = conn.execute(
            "SELECT COUNT(*) FROM generations WHERE model = ANY(%s)",
            (pcv_models,),
        ).fetchone()[0]
    if n_pcv == 0:
        print("No MAS (pcv:*) generations — skip 03b_score_edge_cue")
        return

    # Score stored edge_cue_chain if generations.output is JSON MAS row.
    out_path = results_dir / "edge_cue_mas.jsonl"
    n_done = 0
    with open(out_path, "w") as fout, psycopg.connect(cfg["db_url"]) as conn:
        for model in pcv_models:
            rows = conn.execute(
                """
                SELECT g.poem_id, g.output, p.prompt, p.domain_cluster
                FROM generations g
                JOIN poems p ON p.id = g.poem_id
                WHERE g.model = %s AND p.split = %s
                ORDER BY g.poem_id
                """,
                (model, cfg["split"]),
            ).fetchall()
            if args.limit is not None:
                rows = rows[: args.limit]
            for pid, output, prompt, domain in rows:
                chain = []
                handoff = None
                try:
                    payload = json.loads(output)
                    if isinstance(payload, dict) and "edge_cue_chain" in payload:
                        chain = payload["edge_cue_chain"]
                        handoff = payload.get("handoff_gain_rate")
                        if handoff is None:
                            scored = [e for e in chain if not e.get("diagnostic")]
                            handoff = handoff_gain_rate(scored if scored else chain)
                except (json.JSONDecodeError, TypeError):
                    # Plain text joint output — cannot recover edges without re-run.
                    continue
                if not chain:
                    continue
                edge_payload = {
                    "model": model,
                    "domain_cluster": int(domain) if domain is not None else None,
                    "edge_cue_chain": chain,
                    "handoff_gain_rate": handoff,
                }
                # Side for critic→proposer edges aggregate
                critic_edges = [
                    e
                    for e in chain
                    if e.get("edge_id") == "proposer_draft_to_revision"
                ]
                mean_cue = (
                    sum(float(e["cue"]) for e in critic_edges) / len(critic_edges)
                    if critic_edges
                    else 0.0
                )
                conn.execute(
                    """
                    INSERT INTO scores (poem_id, side, metric_version, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (poem_id, side, metric_version)
                    DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
                    """,
                    (
                        str(pid),
                        "edge:critic_to_proposer",
                        metric_version,
                        json.dumps(
                            {
                                **edge_payload,
                                "mean_edge_cue": mean_cue,
                                "n_edges": len(critic_edges),
                            }
                        ),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO scores (poem_id, side, metric_version, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (poem_id, side, metric_version)
                    DO UPDATE SET payload = EXCLUDED.payload, created_at = now()
                    """,
                    (
                        str(pid),
                        "edge:handoff_gain",
                        metric_version,
                        json.dumps(edge_payload),
                    ),
                )
                fout.write(
                    json.dumps({"id": str(pid), "model": model, **edge_payload}) + "\n"
                )
                n_done += 1
        conn.commit()
    print(f"DONE 03b_score_edge_cue n={n_done} → {out_path}")


if __name__ == "__main__":
    main()
