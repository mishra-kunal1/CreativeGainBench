"""R_D across original / plain / technical (and scrambled) probe banks.

Keeps H, codebook, MiniLM, and δ_D frozen. Rebuilds probe encodings only.
Writes ``results/e5_rd_panel.jsonl``. Requires poetry_v2 domain ctx pickles
when run live; analysis tests feed synthetic JSONL instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    PROBE_VARIANTS,
    SCRAMBLED_VARIANTS,
    assert_output_isolated,
    ensure_results_dir,
    load_config,
    read_jsonl,
    write_jsonl,
)


def load_probe_pairs(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    banks = data.get("banks") or data
    return {k: list(v) for k, v in banks.items()}


def per_item_cv(values: list[float]) -> float:
    import numpy as np

    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if abs(mean) < 1e-12:
        return float("nan")
    return sd / abs(mean)


def swap_probe_encodings(ctx: Any, probe_texts: list[str], probe_seqs: list[list[int]]) -> Any:
    """Rebuild probe encodings only; ``base_lm`` (H) stays the same object."""
    from creativegainbench.metrics.deformation import DomainDeformationContext

    kept = [(t, s) for t, s in zip(probe_texts, probe_seqs) if s]
    if not kept:
        raise ValueError("no non-empty probe encodings after variant rebuild")
    return DomainDeformationContext(
        domain=ctx.domain,
        base_lm=ctx.base_lm,
        probe_texts=[t for t, _ in kept],
        probe_symbol_seqs=[s for _, s in kept],
        vocab_size=ctx.vocab_size,
        order=ctx.order,
    )


def _load_frozen_stack(cfg: dict, device: str):
    artifacts = Path(cfg["artifacts"])
    codebook_path = artifacts / "idea_codebook.pt"
    if not codebook_path.exists():
        raise RuntimeError(
            f"Missing frozen codebook at {codebook_path}. "
            "score_rd_encoder needs poetry_v2 artifacts (prepare-artifacts). "
            "It does not rebuild H or the codebook. For tests, pass synthetic "
            "scores to analyze_e5.py instead."
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "construct_validity"))
    from metrics.pipeline import load_stack  # noqa: E402

    return load_stack(device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=None, help="y_panel.jsonl (matched y as eval texts)")
    parser.add_argument("--probes", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--domain", type=int, default=None, help="poetry_v2 domain cluster (default: first available)")
    args = parser.parse_args()

    cfg = load_config()
    results = ensure_results_dir(cfg)
    panel_path = args.panel or (results / "y_panel.jsonl")
    probes_path = args.probes or (results / "probe_pairs.json")
    out_path = assert_output_isolated(args.out or (results / "e5_rd_panel.jsonl"))
    if not panel_path.exists():
        raise SystemExit(f"missing {panel_path}")
    if not probes_path.exists():
        raise SystemExit(f"missing {probes_path}; run construct_probe_pairs.py")

    stack = _load_frozen_stack(cfg, args.device)
    if not stack.domain_ctx:
        raise SystemExit("frozen domain ctx missing; cannot score R_D")
    domain = args.domain if args.domain is not None else sorted(stack.domain_ctx)[0]
    base_ctx = stack.domain_ctx[domain]

    banks = load_probe_pairs(probes_path)
    from creativegainbench.metrics.deformation import compute_deformation, encode_text

    variant_ctx = {}
    for name in list(PROBE_VARIANTS) + list(SCRAMBLED_VARIANTS) + ["scrambled"]:
        texts = banks.get(name) or []
        if not texts:
            continue
        seqs = [
            encode_text(
                t,
                span_encoder=stack.encoder,
                codebook=stack.codebook,
                boundary_detector=stack.boundary,
            )
            for t in texts
        ]
        variant_ctx[name] = swap_probe_encodings(base_ctx, texts, seqs)

    items = read_jsonl(panel_path)
    if args.limit is not None:
        items = items[: args.limit]

    rows = []
    for rec in items:
        y = rec["y"]["matched"]
        scores: dict[str, float] = {}
        for name, ctx in variant_ctx.items():
            result = compute_deformation(
                y,
                ctx,
                span_encoder=stack.encoder,
                codebook=stack.codebook,
                boundary_detector=stack.boundary,
            )
            scores[name] = float(result.r_d_norm)
            scores[f"{name}_raw"] = float(result.r_d_raw)
        variants = [scores[v] for v in PROBE_VARIANTS if v in scores]
        row = {
            "item_id": rec["item_id"],
            "domain": rec.get("domain"),
            "domain_cluster": rec.get("domain_cluster", domain),
            "r_d_norm": scores,
            "cv_original_plain_technical": per_item_cv(variants),
            "cv_scrambled": per_item_cv(
                [scores[v] for v in SCRAMBLED_VARIANTS if v in scores]
            ),
            "note": "H/codebook/MiniLM/delta_D frozen; probe encodings rebuilt only",
        }
        rows.append(row)
    write_jsonl(out_path, rows)
    print(f"DONE score_rd_encoder n={len(rows)} → {out_path}", flush=True)


if __name__ == "__main__":
    main()
