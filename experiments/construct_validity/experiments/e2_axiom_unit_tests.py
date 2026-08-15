"""E2 — Lean-aligned axiom checks on live poetry_v2 compressor."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.connection import connect, run_migrations  # noqa: E402
from db.queries import fetch_poems_by_split  # noqa: E402
from experiments._common import new_run_id, record  # noqa: E402
from lib import load_config  # noqa: E402
from metrics.pipeline import load_stack  # noqa: E402


def main() -> None:
    run_migrations()
    cfg = load_config()
    run_id = new_run_id()
    stack = load_stack(device="cpu")
    rng = random.Random(cfg["seed"])

    exact_ok = exact_n = 0
    pad_ok = pad_n = 0
    mono_ok = mono_n = 0

    with connect() as conn:
        for d, ctx in sorted(stack.domain_ctx.items()):
            train = fetch_poems_by_split(conn, d, "train")
            if len(train) < 3:
                continue
            sample = rng.sample(train, min(8, len(train)))

            # D4: exact H member → small deformation
            for pid, body, *_ in sample:
                rd = stack.score_r_d(body, d)
                exact_n += 1
                if abs(rd.r_d_norm) < 1e-2:  # WB count reinforcement needs looser tol
                    exact_ok += 1

            # Anti-padding: pad should not substantially increase r_d_norm
            for pid, body, *_ in sample[:5]:
                base = stack.score_r_d(body, d)
                padded = body.rstrip() + "\n\n" + "\n".join(["(padding)"] * 8)
                pad_rd = stack.score_r_d(padded, d)
                pad_n += 1
                if pad_rd.r_d_norm <= base.r_d_norm + 1e-3:
                    pad_ok += 1

            # Monotonicity sanity on symbols: reinforcing a probe vs random OOV-ish text
            if ctx.probe_texts:
                probe_text = ctx.probe_texts[0]
                junk = "xyzzy " * 40
                d_probe = stack.score_r_d(probe_text, d)
                d_junk = stack.score_r_d(junk, d)
                mono_n += 1
                # Reinforcing probe content should not be worse than nonsense for deformation
                # (soft check — junk can be negative)
                if d_probe.r_d_raw >= d_junk.r_d_raw - 1e-6:
                    mono_ok += 1

    exact_rate = exact_ok / max(exact_n, 1)
    pad_rate = pad_ok / max(pad_n, 1)
    mono_rate = mono_ok / max(mono_n, 1)

    record(
        run_id,
        "E2",
        "exact_copy_invariance",
        exact_rate,
        exact_rate >= 0.95,
        {"ok": exact_ok, "n": exact_n, "tol": 1e-2},
    )
    record(
        run_id,
        "E2",
        "anti_padding",
        pad_rate,
        pad_rate >= 0.95,
        {"ok": pad_ok, "n": pad_n},
    )
    record(
        run_id,
        "E2",
        "monotonicity_sanity",
        mono_rate,
        mono_rate >= 0.95,
        {"ok": mono_ok, "n": mono_n},
    )
    print("DONE E2")


if __name__ == "__main__":
    main()
