"""
Calibrate δ_D so paraphrases fail the D-gate and restructures pass.

Writes artifacts/delta_d_v1.json and updates score.delta_d guidance in a
sidecar consumed by benchmark_eval.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from creativegainbench.ideas.artifacts import load_artifacts
from creativegainbench.metrics.structural_novelty import (
    ProbeSet,
    compute_structural_novelty,
)

ARTIFACTS_ROOT = Path(__file__).resolve().parent


def _paraphrases(probe_seed: str) -> list[tuple[str, str]]:
    """Surface rewordings / near-copies — should not clear a sensible δ_D."""
    return [
        ("para_copy", probe_seed),
        (
            "para_minor",
            probe_seed.replace("Propose", "Suggest").replace("Write", "Compose")
            if ("Propose" in probe_seed or "Write" in probe_seed)
            else ("In short: " + probe_seed),
        ),
        (
            "para_wrap",
            f"Here is the same request restated: {probe_seed}",
        ),
    ]


def _restructures() -> list[tuple[str, str]]:
    """Outputs that deliberately change conceptual structure vs probes."""
    return [
        (
            "re_0",
            "We throw away frozen probes. Creativity is redefined as maximizing "
            "clickthrough on social media. The compressor is fine-tuned online "
            "on yesterday's viral posts. Idea units are banned; only emoji "
            "bigrams are kept. Utility equals advertising revenue.",
        ),
        (
            "re_1",
            "Step A: sample random byte strings. Step B: gzip them against Wikipedia. "
            "Step C: call the length reduction 'transformational creativity'. "
            "Step D: never measure receiver decisions. Step E: resample the probe "
            "set every gradient update so the baseline chases the policy.",
        ),
        (
            "re_2",
            "Mathematics plan: replace proofs with mood boards. Science plan: "
            "replace experiments with horoscopes. Writing plan: shuffle syllables "
            "until novelty judges smile. Interaction gain is computed by counting "
            "exclamation marks across agents.",
        ),
    ]


def calibrate_delta_d(
    *,
    version: str = "v1",
    artifacts_root: Path | None = None,
    probe_limit: int = 20,
) -> dict:
    root = artifacts_root or ARTIFACTS_ROOT
    pipeline = load_artifacts(
        version=version, device="cpu", artifacts_root=root, verify_hashes=True
    )
    tiny = ProbeSet(
        strings=pipeline.probe_set.strings[:probe_limit],
        seed=pipeline.seed,
        strata=pipeline.probe_set.strata,
    )

    def _rd(y: str) -> float:
        return compute_structural_novelty(
            y,
            probe_set=tiny,
            compressor=pipeline.compressor,
            codebook=pipeline.codebook,
            span_encoder=pipeline.span_encoder,
            boundary_detector=pipeline.boundary_detector,
            n=pipeline.n,
            boundary_threshold=pipeline.boundary_threshold,
            device="cpu",
        )

    seed_probe = pipeline.probe_set.strings[0]
    para_scores = [(k, _rd(t)) for k, t in _paraphrases(seed_probe)]
    re_scores = [(k, _rd(t)) for k, t in _restructures()]
    para_vals = [s for _, s in para_scores]
    re_vals = [s for _, s in re_scores]

    para_hi = max(para_vals)
    re_lo = min(re_vals)
    # Place threshold between paraphrase max and restructure min when separable;
    # otherwise fall back to paraphrase 90th percentile + epsilon.
    if re_lo > para_hi:
        delta = 0.5 * (para_hi + re_lo)
    else:
        ordered = sorted(para_vals)
        idx = max(0, int(0.9 * (len(ordered) - 1)))
        delta = ordered[idx] + 1e-3

    payload = {
        "version": version,
        "delta_d": float(delta),
        "paraphrase_r_d": {k: float(v) for k, v in para_scores},
        "restructure_r_d": {k: float(v) for k, v in re_scores},
        "paraphrase_mean": float(statistics.fmean(para_vals)),
        "restructure_mean": float(statistics.fmean(re_vals)),
        "probe_limit": probe_limit,
        "separable": bool(re_lo > para_hi),
    }
    out = root / f"delta_d_{version}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Calibrated δ_D={delta:.6f} → {out}")
    print(f"  paraphrase r_d: {para_vals}")
    print(f"  restructure r_d: {re_vals}")
    return payload


def main() -> None:
    calibrate_delta_d()


if __name__ == "__main__":
    main()
