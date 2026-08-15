"""
Empirical PROOF-07 check: among rows with large G_k, does joint CUE exceed
max(solo CUEs)? Surfaces receiver-calibration failures; does not assume the
Lean bridge holds empirically.

  Gk_implies_CUE_improvement under ReceiverCalibratedMAS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_rows(
    rows: list[dict],
    *,
    gk_threshold: float = 0.0,
) -> dict:
    """
    Expect each row to carry:
      g_k, joint_cue, solo_cues: list[float]
    """
    eligible = [r for r in rows if float(r.get("g_k", 0.0)) > gk_threshold]
    if not eligible:
        return {
            "n_eligible": 0,
            "n_hold": 0,
            "pass_rate": None,
            "gk_threshold": gk_threshold,
        }
    n_hold = 0
    for r in eligible:
        joint = float(r["joint_cue"])
        solos = [float(x) for x in r.get("solo_cues") or []]
        if solos and joint > max(solos):
            n_hold += 1
    return {
        "n_eligible": len(eligible),
        "n_hold": n_hold,
        "pass_rate": n_hold / len(eligible),
        "gk_threshold": gk_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical MASBridge / PROOF-07 check")
    parser.add_argument("--results", type=Path, required=True, help="JSONL with g_k, joint_cue, solo_cues")
    parser.add_argument("--gk-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    summary = check_rows(rows, gk_threshold=args.gk_threshold)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
