"""
E5 — Minimal supervised slice (report-only).

Samples pairs across CUE×R_D gate quadrants; scores structural inventiveness
with a local Ollama judge (gemma2:2b). Does NOT refit δ_D.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import spearman  # noqa: E402
from db.connection import connect, run_migrations  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402
from lib import load_config  # noqa: E402

JUDGE_PROMPT = """Rate structural inventiveness of this poem relative to a typical competent poem on the same prompt.
Respond with ONLY JSON: {{"inventiveness": <1|2|3>}} where 1=low/conventional, 2=mid, 3=high/structurally inventive.

Prompt:
{prompt}

Poem:
{text}
"""


def _judge(client: OpenAI, model: str, prompt: str, text: str) -> int:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    prompt=prompt[:1500], text=text[:2000]
                ),
            }
        ],
        temperature=0,
        n=1,
    )
    raw = completion.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return 2
    try:
        v = int(json.loads(m.group(0)).get("inventiveness", 2))
        return max(1, min(3, v))
    except Exception:
        return 2


def main() -> None:
    run_migrations()
    cfg = load_config()
    run_id = new_run_id()
    rng = random.Random(cfg["seed"])

    with connect() as conn:
        thr = {
            int(d): float(t)
            for d, t in conn.execute(
                "SELECT domain_cluster, delta_d_95 FROM delta_d_thresholds"
            ).fetchall()
        }

    cue_dir = Path(__file__).resolve().parents[2] / "experiment1" / "results"
    # Prefer gemma cue file for a single consistent generator side
    path = cue_dir / "cue_gemma2_2b.jsonl"
    if not path.exists():
        record(
            run_id,
            "E5",
            "judge_agreement",
            None,
            None,
            {"error": f"missing {path}"},
        )
        print("DONE E5 (early)")
        return

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    with connect() as conn:
        prompts = {
            str(r[0]): (r[1], r[2], int(r[3]) if r[3] is not None else -1)
            for r in conn.execute(
                "SELECT id, prompt, body, domain_cluster FROM poems WHERE split='eval'"
            ).fetchall()
        }

    # Build quadrants on human side
    buckets: dict[str, list] = {
        "cue1_d1": [],
        "cue1_d0": [],
        "cue0_d1": [],
        "cue0_d0": [],
    }
    for rec in rows:
        meta = prompts.get(rec["id"])
        if not meta:
            continue
        prompt, body, domain = meta
        h = rec["human"]
        rd = h.get("r_d_norm")
        cg = bool(h.get("cue_gate"))
        if rd is None or domain not in thr:
            continue
        dg = float(rd) > thr[domain]
        key = f"cue{1 if cg else 0}_d{1 if dg else 0}"
        buckets[key].append((rec["id"], prompt, body, 1 if (cg and dg) else 0))

    sample = []
    per = 20
    for key, items in buckets.items():
        sample.extend(rng.sample(items, min(per, len(items))))

    client = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )
    labels = []
    gates = []
    for pid, prompt, body, joint in sample:
        try:
            inv = _judge(client, "gemma2:2b", prompt, body)
        except Exception as e:
            print(f"[warn] judge failed {pid}: {e}", flush=True)
            continue
        labels.append(float(inv))
        gates.append(float(joint))

    rho = spearman(gates, labels) if len(labels) >= 5 else float("nan")
    record(
        run_id,
        "E5",
        "judge_agreement",
        None if (isinstance(rho, float) and rho != rho) else float(rho),
        None,  # report-only
        {
            "spearman_joint_vs_inventiveness": rho,
            "n": len(labels),
            "bucket_sizes": {k: len(v) for k, v in buckets.items()},
            "sampled": len(sample),
            "note": "report only — do not refit delta_D",
        },
    )
    print("DONE E5")


if __name__ == "__main__":
    main()
