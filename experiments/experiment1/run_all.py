#!/usr/bin/env python3
"""
Orchestrate experiment1 components in order.

Usage:
  python experiments/experiment1/run_all.py
  python experiments/experiment1/run_all.py --from 03_score_rd
  python experiments/experiment1/run_all.py --only 01_schema
  python experiments/experiment1/run_all.py --skip-cue
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
COMPONENTS = [
    "00_ensure_delta_d",
    "01_schema",
    "02_generate",
    "03_score_rd",
    "03b_score_edge_cue",
    "04_score_cue",
    "05_aggregate",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--skip-cue", action="store_true")
    parser.add_argument("--skip-edge-cue", action="store_true")
    parser.add_argument("--model", default=None, help="Pass through to generate/score")
    parser.add_argument("--limit", type=int, default=None, help="Limit generations (smoke)")
    parser.add_argument(
        "--mas",
        action="store_true",
        help="Pass --mode mas --score-edge-cue to 02_generate",
    )
    args = parser.parse_args()

    steps = list(COMPONENTS)
    if args.skip_cue:
        steps = [s for s in steps if s != "04_score_cue"]
    if args.skip_edge_cue:
        steps = [s for s in steps if s != "03b_score_edge_cue"]
    if args.only:
        steps = [args.only]
    elif args.from_step:
        if args.from_step not in steps:
            raise SystemExit(f"unknown step {args.from_step}")
        steps = steps[steps.index(args.from_step) :]

    for step in steps:
        script = EXP_ROOT / "components" / f"{step}.py"
        cmd = [sys.executable, str(script)]
        if step == "02_generate":
            if args.model:
                cmd += ["--model", args.model]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if args.mas:
                cmd += ["--mode", "mas", "--score-edge-cue"]
        elif step in {"03_score_rd", "03b_score_edge_cue", "04_score_cue"} and args.model:
            cmd += ["--model", args.model]
        if step == "03b_score_edge_cue" and args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        print(f"\n=== {step} ===", flush=True)
        log_path = EXP_ROOT / "logs" / f"{step}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd,
                cwd=str(EXP_ROOT.parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(proc.stdout)
            print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"{step} failed with code {proc.returncode}; see {log_path}")
    print("\n=== experiment1 complete ===")


if __name__ == "__main__":
    main()
