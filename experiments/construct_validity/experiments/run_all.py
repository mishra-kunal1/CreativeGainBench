#!/usr/bin/env python3
"""Orchestrate construct_validity: migrations → negative bank → δ_D → E0–E5 → report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]

STEPS = [
    ("migrations", None),  # handled inline
    ("build_negative_bank", ROOT / "calibration" / "build_negative_bank.py"),
    ("calibrate_delta_d", ROOT / "calibration" / "calibrate_delta_d.py"),
    ("e0", ROOT / "experiments" / "e0_data_audit.py"),
    ("e1", ROOT / "experiments" / "e1_known_groups.py"),
    ("e2", ROOT / "experiments" / "e2_axiom_unit_tests.py"),
    ("e3", ROOT / "experiments" / "e3_calibration_acceptance.py"),
    ("e4", ROOT / "experiments" / "e4_convergent_discriminant.py"),
    ("e5", ROOT / "experiments" / "e5_supervised_slice.py"),
    ("e6", ROOT / "experiments" / "e6_mas_handoff_validity.py"),
    ("v7", ROOT / "experiments" / "v7_zstar_source_audit.py"),
    ("v8", ROOT / "experiments" / "v8_gk_semantics_audit.py"),
    ("report", ROOT / "analysis" / "report_builder.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", default=None)
    parser.add_argument("--skip-e5", action="store_true")
    parser.add_argument("--skip-bank", action="store_true", help="reuse existing negative bank")
    parser.add_argument("--per-type", type=int, default=30)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from db.connection import run_migrations

    names = [n for n, _ in STEPS]
    start = 0
    if args.from_step:
        if args.from_step not in names:
            raise SystemExit(f"unknown step {args.from_step}; choose from {names}")
        start = names.index(args.from_step)

    for name, script in STEPS[start:]:
        if name == "migrations":
            print("=== migrations ===", flush=True)
            run_migrations()
            continue
        if args.skip_bank and name == "build_negative_bank":
            print("=== skip build_negative_bank ===", flush=True)
            continue
        if args.skip_e5 and name == "e5":
            print("=== skip e5 ===", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        cmd = [sys.executable, str(script)]
        if name == "build_negative_bank":
            cmd += ["--per-type", str(args.per_type), "--device", "cpu"]
        log = ROOT / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            proc = subprocess.run(
                cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            f.write(proc.stdout)
            print(proc.stdout, end="", flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"{name} failed ({proc.returncode}); see {log}")

    print("\n=== construct_validity complete ===")


if __name__ == "__main__":
    main()
