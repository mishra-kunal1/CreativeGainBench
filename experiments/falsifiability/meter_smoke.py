#!/usr/bin/env python3
"""n=2 live CUE meter smoke on the three free Ollama Cloud models.

Fails if Gemma or Nemotron cannot parse JSON. gpt-oss may update or be
flagged inert, but priors/posteriors must be logged (no silent zeros).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import assert_output_isolated, ensure_results_dir, load_config

from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
from construct_contributions import build_panel, load_heldout_records

FREE_MODELS = ("gpt-oss:20b", "gemma4:31b", "nemotron-3-nano:30b")
MUST_PARSE = ("gemma4:31b", "nemotron-3-nano:30b")


def _is_uniform(probs: list[float] | None, *, eps: float = 1e-6) -> bool:
    if not probs:
        return True
    k = len(probs)
    return all(abs(p - 1.0 / k) <= eps for p in probs)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://ollama.com/v1")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    out = assert_output_isolated(args.out or (ensure_results_dir(cfg) / "meter_smoke.json"))
    records = load_heldout_records(n=max(args.n, 6), seed=42)
    panel = build_panel(
        records, seed=42, n=args.n, phase="b", cross_mode="across_domain"
    )
    items = panel[: args.n]

    rows: list[dict] = []
    failures: list[str] = []
    for model in FREE_MODELS:
        rx = CUEBeliefReceiver(
            CUEBeliefConfig(
                provider="ollama",
                model=model,
                base_url=args.base_url,
                temperature=0.0,
                max_tokens=1024,
            )
        )
        for item in items:
            y = item["y"]["matched"]
            cue, _model, diag = rx.compute_cue_for_output(item["prompt"], y)
            rec = {
                "receiver": model,
                "item_id": item["item_id"],
                "domain": item.get("domain"),
                "cue": cue,
                "parse_ok": diag.get("parse_ok"),
                "parse_ok_prior": diag.get("parse_ok_prior"),
                "parse_ok_posterior": diag.get("parse_ok_posterior"),
                "parse_reason_prior": diag.get("parse_reason_prior"),
                "parse_reason_posterior": diag.get("parse_reason_posterior"),
                "prior": diag.get("prior"),
                "posterior": diag.get("posterior"),
                "raw_preview_prior": diag.get("raw_preview_prior"),
                "raw_preview_posterior": diag.get("raw_preview_posterior"),
                "text_source_prior": diag.get("text_source_prior"),
                "text_source_posterior": diag.get("text_source_posterior"),
                "brier_delta": diag.get("brier_delta"),
                "brier_delta_signed": diag.get("brier_delta_signed"),
                "outcome_label": diag.get("outcome_label"),
            }
            rows.append(rec)
            print(
                f"{model} {item['item_id']}: parse_ok={rec['parse_ok']} "
                f"cue={rec['cue']} signed={rec['brier_delta_signed']} "
                f"src_post={rec['text_source_posterior']}",
                flush=True,
            )

    by_rx: dict[str, list[dict]] = {}
    for rec in rows:
        by_rx.setdefault(str(rec["receiver"]), []).append(rec)

    for model in MUST_PARSE:
        recs = by_rx.get(model, [])
        if not recs or not all(r.get("parse_ok") for r in recs):
            failures.append(f"{model} failed JSON parse")
            continue
        matched_posts = [r.get("posterior") for r in recs]
        if all(_is_uniform(p) for p in matched_posts):
            failures.append(f"{model} posterior is uniform on matched y")

    oss = by_rx.get("gpt-oss:20b", [])
    oss_inert = False
    if oss:
        cues = [r.get("cue") for r in oss]
        logged = all(
            r.get("prior") is not None or r.get("raw_preview_prior") is not None
            for r in oss
        )
        if not logged:
            failures.append("gpt-oss:20b silent zeros (no prior/posterior/raw_preview)")
        finite = [float(c) for c in cues if c is not None]
        oss_inert = (not finite) or (len(set(round(c, 12) for c in finite)) <= 1)
        if oss_inert:
            print("gpt-oss:20b flagged cue_inert with logged vectors", flush=True)

    payload = {
        "n_items": len(items),
        "receivers": list(FREE_MODELS),
        "rows": rows,
        "gpt_oss_inert": oss_inert,
        "failures": failures,
        "passed": not failures,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} passed={payload['passed']}", flush=True)
    if failures:
        raise SystemExit("meter smoke failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
