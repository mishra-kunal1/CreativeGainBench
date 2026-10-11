"""Ollama CUE over receivers × y-arms.

Resume-safe JSONL under ``results/e7_cue_panel.jsonl``.
Caches prior per (receiver, prompt). Freezes z* from y_matched and applies
the same index to every arm via ``external_outcome_index``.

Does not write ``experiments/experiment1/results/cue_*.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    ARMS,
    annotator_domain,
    append_jsonl,
    assert_output_isolated,
    ensure_results_dir,
    load_config,
    read_jsonl,
)

from creativegainbench.metrics.cue_receiver import CUEBeliefConfig, CUEBeliefReceiver
from creativegainbench.receivers.ollama_receiver import OllamaReceiverAgent


def resume_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for rec in read_jsonl(path):
        keys.add((str(rec["item_id"]), str(rec["receiver"]), str(rec["arm"])))
    return keys


def math_validity_prompt(q: str, y: str, agent: Any) -> str:
    """Verifier-style prompt from ``OllamaReceiverAgent.condition(q, context=y)``.

    LLM proxy — not a Lean 4 oracle.
    """
    conditioned = agent.condition(q, context=y)
    return (
        conditioned
        + "\n\nVerifier (LLM proxy, NOT a Lean 4 kernel / Lean 4 oracle): "
        'decide whether the contribution is a valid proof of the task. '
        'Return ONLY JSON {"valid": true|false, "reason": "<short>"}.'
    )


def parse_validity_bit(text: str) -> dict[str, Any]:
    import re

    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not m:
        return {"valid": None, "reason": "unparseable", "raw": (text or "")[:500]}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"valid": None, "reason": "unparseable", "raw": (text or "")[:500]}
    v = data.get("valid")
    if isinstance(v, str):
        v = v.strip().lower() in {"true", "yes", "1"}
    return {"valid": bool(v) if v is not None else None, "reason": data.get("reason")}


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()


class PriorCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], list[float]] = {}

    def get(self, receiver: str, prompt: str) -> list[float] | None:
        with self._lock:
            hit = self._data.get((receiver, _prompt_key(prompt)))
            return list(hit) if hit is not None else None

    def put(self, receiver: str, prompt: str, prior: list[float]) -> list[float]:
        with self._lock:
            self._data[(receiver, _prompt_key(prompt))] = list(prior)
            return list(prior)


def freeze_z_stars(panel: list[dict[str, Any]], annotators: dict[str, Any]) -> dict[str, int]:
    """z* from y_matched only. Same index is applied to all four arms."""
    out: dict[str, int] = {}
    for row in panel:
        domain = row.get("annotator_domain") or annotator_domain(row)
        ann = annotators[domain]
        out[str(row["item_id"])] = int(ann.annotate(row["y"]["matched"]))
    return out


def _clip(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:n]


def score_one(
    *,
    receiver: CUEBeliefReceiver,
    receiver_name: str,
    prompt: str,
    y: str,
    arm: str,
    z_star: int,
    prior_cache: PriorCache,
) -> dict[str, Any]:
    cached = prior_cache.get(receiver_name, prompt)
    cue_val, _model, diag = receiver.compute_cue_for_output(
        prompt,
        y,
        external_outcome_index=z_star,
        prior=cached,
    )
    # Cache only a successful prior. Parse-fail must not freeze a missing/uniform.
    if cached is None and diag.get("parse_ok_prior") and diag.get("prior"):
        prior_cache.put(receiver_name, prompt, list(diag["prior"]))
    rec = {
        "receiver": receiver_name,
        "arm": arm,
        "cue": None if cue_val is None else float(cue_val),
        "brier_delta": diag.get("brier_delta"),
        "brier_delta_signed": diag.get("brier_delta_signed"),
        "bit_length": diag.get("bit_length"),
        "z_star": z_star,
        "z_star_source": "matched_frozen",
        "outcome_index": diag.get("outcome_index"),
        "outcome_label": diag.get("outcome_label"),
        "prior_cached": bool(diag.get("prior_cached")),
        "outcome_source": diag.get("outcome_source"),
        "prior": diag.get("prior"),
        "posterior": diag.get("posterior"),
        "parse_ok": diag.get("parse_ok"),
        "parse_ok_prior": diag.get("parse_ok_prior"),
        "parse_ok_posterior": diag.get("parse_ok_posterior"),
        "parse_reason_prior": diag.get("parse_reason_prior"),
        "parse_reason_posterior": diag.get("parse_reason_posterior"),
        "raw_preview_prior": diag.get("raw_preview_prior"),
        "raw_preview_posterior": diag.get("raw_preview_posterior"),
        "text_source_prior": diag.get("text_source_prior"),
        "text_source_posterior": diag.get("text_source_posterior"),
        "cue_missing_reason": diag.get("cue_missing_reason"),
    }
    return rec


def _build_annotators(panel: list[dict[str, Any]], *, encoder_backend: str, device: str) -> dict[str, Any]:
    from creativegainbench.ideas.span_encoder import build_span_encoder
    from creativegainbench.metrics.outcome_annotator import OutcomeAnnotator

    encoder = build_span_encoder(encoder_backend, device=device)
    domains = sorted({row.get("annotator_domain") or annotator_domain(row) for row in panel})
    out: dict[str, Any] = {}
    for d in domains:
        out[d] = OutcomeAnnotator(span_encoder=encoder, domain=d).fit()
    return out


def _maybe_validity(
    *,
    item: dict[str, Any],
    arm: str,
    y: str,
    agent: Any | None,
    client: Any,
    model: str,
    temperature: float,
) -> dict[str, Any] | None:
    if agent is None:
        return None
    if item.get("phase") != "b":
        return None
    if str(item.get("domain")) not in {"mathematical_proof", "math"}:
        return None
    if arm != "matched":
        return None
    prompt = math_validity_prompt(item["prompt"], y, agent)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        n=1,
    )
    raw = completion.choices[0].message.content or ""
    parsed = parse_validity_bit(raw)
    parsed["proxy"] = "llm_verifier"
    parsed["lean4_oracle"] = False
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--encoder", default="minilm", help="minilm (Phase A default) or hash (smoke)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--receivers", default=None, help="comma-separated override")
    parser.add_argument("--math-validity", action="store_true", help="Phase B LLM validity bit")
    args = parser.parse_args()

    cfg = load_config()
    results = ensure_results_dir(cfg)
    panel_path = args.panel or (results / "y_panel.jsonl")
    out_path = assert_output_isolated(args.out or (results / "e7_cue_panel.jsonl"))
    if not panel_path.exists():
        raise SystemExit(f"missing y panel {panel_path}; run construct_contributions.py")

    panel = read_jsonl(panel_path)
    if args.limit is not None:
        panel = panel[: args.limit]
    if not panel:
        raise SystemExit("empty y panel")

    receivers = (
        [s.strip() for s in args.receivers.split(",") if s.strip()]
        if args.receivers
        else list(cfg["receivers"])
    )
    base_url = args.base_url or cfg.get("base_url")
    temperature = float(cfg.get("temperature", 0.0))
    workers = int(args.workers if args.workers is not None else cfg.get("workers", 2))
    clip = int(cfg.get("clip", 4000))

    print(f"freezing z* from y_matched (encoder={args.encoder}) n={len(panel)}", flush=True)
    annotators = _build_annotators(panel, encoder_backend=args.encoder, device=args.device)
    z_stars = freeze_z_stars(panel, annotators)

    done = resume_keys(out_path)
    prior_cache = PriorCache()
    file_lock = threading.Lock()
    rx_by_name = {
        name: CUEBeliefReceiver(
            CUEBeliefConfig(
                provider="ollama",
                model=name,
                base_url=base_url,
                temperature=temperature,
            )
        )
        for name in receivers
    }

    validity_agent = None
    if args.math_validity:
        from creativegainbench.ideas.span_encoder import build_span_encoder

        validity_agent = OllamaReceiverAgent(
            span_encoder=build_span_encoder(args.encoder, device=args.device),
            model=receivers[0],
            base_url=base_url,
            temperature=temperature,
        )

    jobs: list[tuple[str, dict[str, Any], str]] = []
    for row in panel:
        for rec_name in receivers:
            for arm in ARMS:
                key = (str(row["item_id"]), rec_name, arm)
                if key not in done:
                    jobs.append((rec_name, row, arm))

    print(f"scoring {len(jobs)} (receiver, item, arm) tuples; resume skipped {len(done)}", flush=True)

    def _run(job: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
        rec_name, row, arm = job
        y = _clip(row["y"][arm], clip)
        try:
            rec = score_one(
                receiver=rx_by_name[rec_name],
                receiver_name=rec_name,
                prompt=row["prompt"],
                y=y,
                arm=arm,
                z_star=z_stars[str(row["item_id"])],
                prior_cache=prior_cache,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[warn] {rec_name} {row.get('item_id')} {arm}: {err[:240]}", flush=True)
            return {
                "item_id": row["item_id"],
                "phase": row.get("phase"),
                "domain": row.get("domain"),
                "domain_cluster": row.get("domain_cluster"),
                "receiver": rec_name,
                "arm": arm,
                "cue": None,
                "error": err[:500],
            }
        rec.update(
            {
                "item_id": row["item_id"],
                "phase": row.get("phase"),
                "domain": row.get("domain"),
                "domain_cluster": row.get("domain_cluster"),
            }
        )
        if validity_agent is not None:
            vb = _maybe_validity(
                item=row,
                arm=arm,
                y=y,
                agent=validity_agent,
                client=rx_by_name[rec_name].client,
                model=rec_name,
                temperature=temperature,
            )
            if vb is not None:
                rec["math_validity"] = vb
        return rec

    n_new = 0
    n_err = 0
    n_parse_fail = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_run, job) for job in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            with file_lock:
                append_jsonl(out_path, rec)
            n_new += 1
            if rec.get("error"):
                n_err += 1
            elif rec.get("cue") is None:
                n_parse_fail += 1
            if n_new % 10 == 0:
                print(
                    f"  wrote {n_new}/{len(jobs)} (errors {n_err}, parse_fail {n_parse_fail})",
                    flush=True,
                )
    print(
        f"DONE score_cue_panel wrote {n_new} → {out_path} "
        f"(errors {n_err}, parse_fail {n_parse_fail})",
        flush=True,
    )


if __name__ == "__main__":
    main()
