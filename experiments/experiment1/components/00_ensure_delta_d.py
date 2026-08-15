"""
Ensure poetry_v2/delta_d_thresholds.json exists (negative-bank contract).

Does not re-run the full construct_validity suite — only checks for the
thresholds artifact (or copies from construct_validity/results if present).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    dest = Path(cfg["artifacts"]) / "delta_d_thresholds.json"
    if dest.exists():
        print(f"OK thresholds present: {dest}")
        return
    cv = (
        Path(__file__).resolve().parents[2]
        / "construct_validity"
        / "results"
        / "delta_d_thresholds.json"
    )
    if cv.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cv, dest)
        print(f"copied {cv} → {dest}")
        return
    raise SystemExit(
        f"Missing {dest} and {cv}. Run "
        "`python experiments/construct_validity/calibration/calibrate_delta_d.py` "
        "after building the negative bank."
    )


if __name__ == "__main__":
    main()
