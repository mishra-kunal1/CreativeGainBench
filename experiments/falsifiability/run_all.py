#!/usr/bin/env python3
"""Orchestrate falsifiability E5/E7. Does not touch construct-validity IDs.

Usage:
  python experiments/falsifiability/run_all.py --phase a --limit 8 --synthetic --skip-score
  python experiments/falsifiability/run_all.py --phase a --limit 4 --base-url http://127.0.0.1:11434/v1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
REPO = EXP_ROOT.parents[1]

STEPS = [
    "construct_contributions",
    "construct_probe_pairs",
    "score_cue_panel",
    "score_rd_encoder",
    "analyze_e7",
    "analyze_e5",
]
SCORE_STEPS = {"score_cue_panel", "score_rd_encoder"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), default="a")
    parser.add_argument("--from", dest="from_step", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke n")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Held-out prompts + mixed-quality y + across-domain cross",
    )
    parser.add_argument("--skip-score", action="store_true", help="No Ollama / MiniLM scorers")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--encoder", default=None, help="minilm | hash")
    parser.add_argument("--math-validity", action="store_true")
    args = parser.parse_args()

    steps = list(STEPS)
    if args.skip_score:
        steps = [s for s in steps if s not in SCORE_STEPS]
        # Live scores are required for analysis; constructors-only smoke stops here
        # unless the user resumes at an analyze_* step.
        if not (args.from_step or "").startswith("analyze"):
            steps = [s for s in steps if not s.startswith("analyze_")]
    if args.skip_analyze:
        steps = [s for s in steps if not s.startswith("analyze_")]
    if args.from_step:
        if args.from_step not in STEPS:
            raise SystemExit(f"unknown step {args.from_step}; choose from {STEPS}")
        allowed = set(STEPS[STEPS.index(args.from_step) :])
        steps = [s for s in steps if s in allowed]

    logs = EXP_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    for step in steps:
        script = EXP_ROOT / f"{step}.py"
        cmd = [sys.executable, str(script)]
        if step == "construct_contributions":
            cmd += ["--phase", args.phase]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if args.synthetic:
                cmd += ["--synthetic"]
            if args.heldout:
                cmd += ["--heldout"]
        elif step == "construct_probe_pairs":
            if args.synthetic:
                cmd += ["--synthetic"]
            # --heldout uses poetry_v2_ctx probes (omit --synthetic)
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
        elif step == "score_cue_panel":
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if args.base_url:
                cmd += ["--base-url", args.base_url]
            if args.encoder:
                cmd += ["--encoder", args.encoder]
            if args.math_validity or args.phase == "b":
                cmd += ["--math-validity"]
        elif step == "score_rd_encoder":
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
        elif step in {"analyze_e7", "analyze_e5"} and args.limit is not None:
            # analyzers read whatever scores exist; small n_boot for smoke
            cmd += ["--n-boot", "200", "--n-perm", "200"]

        print(f"\n=== {step} ===", flush=True)
        log_path = logs / f"{step}.log"
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(proc.stdout)
            print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"{step} failed ({proc.returncode}); see {log_path}")
    print("\n=== falsifiability complete ===")


if __name__ == "__main__":
    main()
