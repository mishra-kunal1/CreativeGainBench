"""E5 analysis: receiver Spearman/CCC + encoder R_D displacement CV.

Reads ``results/e7_cue_panel.jsonl`` (matched arm) and ``results/e5_rd_panel.jsonl``.
Writes JSON/MD under ``experiments/falsifiability/results/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    PROBE_VARIANTS,
    SCRAMBLED_VARIANTS,
    assert_output_isolated,
    ensure_results_dir,
    load_config,
    read_jsonl,
)

from creativegainbench.stats import (
    ConcordanceCC,
    KrippendorffAlpha,
    Resampler,
    Sample,
    SpearmanRho,
    Verdict,
    krippendorff_alpha,
)
from score_rd_encoder import per_item_cv

# Matched-arm CUE variance at or below this is treated as an inert receiver
# (silent zeros / no update), not as a Spearman ρ failure against γ=0.80.
INERT_VARIANCE_MAX = 1e-12


def is_cue_inert(values: np.ndarray | list[float], *, eps: float = INERT_VARIANCE_MAX) -> bool:
    """True when matched-arm CUE is (near) constant, including n<2."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return True
    return float(np.var(v, ddof=0)) <= eps


def _matched_cue_matrix(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], np.ndarray]:
    """receivers, item_ids, matrix (n_items, n_receivers) of matched-arm CUE."""
    by: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if str(r.get("arm", "matched")) != "matched":
            continue
        if r.get("cue") is None:
            continue
        by[str(r["item_id"])][str(r["receiver"])] = float(r["cue"])
    receivers = sorted({rx for arms in by.values() for rx in arms})
    items = sorted(i for i, arms in by.items() if all(rx in arms for rx in receivers))
    mat = np.array([[by[i][rx] for rx in receivers] for i in items], float)
    return receivers, items, mat


def _pair_agreement(
    a: np.ndarray,
    b: np.ndarray,
    ids: list[str],
    rs: Resampler,
    *,
    gamma: float,
    alpha: float,
) -> dict[str, Any]:
    sample = Sample(a, b, item_ids=np.asarray(ids))
    rho = SpearmanRho(alpha=alpha, margin=gamma).evaluate(sample, rs)
    ccc = ConcordanceCC(alpha=alpha, margin=gamma).evaluate(sample, rs)
    passed = rho.verdict is Verdict.EQUIVALENT and ccc.verdict is Verdict.EQUIVALENT
    return {
        "spearman_rho": rho.as_row(),
        "lins_ccc": ccc.as_row(),
        "passed": passed,
        "pass_rule": "both BCa CI lowers > 0.80",
    }


def analyze_e5a(
    rows: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    gamma: float = 0.80,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    receivers_all, items, mat_all = _matched_cue_matrix(rows)
    cue_inert = [
        rx
        for i, rx in enumerate(receivers_all)
        if is_cue_inert(mat_all[:, i] if mat_all.size else np.asarray([]))
    ]
    keep_idx = [i for i, rx in enumerate(receivers_all) if rx not in cue_inert]
    receivers = [receivers_all[i] for i in keep_idx]
    mat = mat_all[:, keep_idx] if keep_idx and mat_all.size else np.zeros((0, 0), float)
    rs = Resampler(n_boot=n_boot, n_perm=n_perm, ci_level=ci_level, seed=seed)
    pairs: dict[str, Any] = {}
    if len(receivers) < 2:
        return {
            "passed": False,
            "n_items": len(items),
            "receivers": receivers,
            "receivers_all": receivers_all,
            "cue_inert": cue_inert,
            "pairs": pairs,
            "krippendorff_alpha": None,
            "reason": (
                "fewer than 2 non-inert receivers; Spearman/CCC not computed "
                "(inert receivers are excluded, not scored against γ=0.80)"
            ),
            "pass_rule": "each non-inert receiver pair: SpearmanRho AND ConcordanceCC CI lower > 0.80",
        }
    for i, j in itertools.combinations(range(len(receivers)), 2):
        key = f"{receivers[i]} vs {receivers[j]}"
        if len(items) < 3:
            pairs[key] = {"passed": False, "reason": "n<3"}
            continue
        pairs[key] = _pair_agreement(
            mat[:, i], mat[:, j], items, rs, gamma=gamma, alpha=alpha
        )
        pairs[key]["receiver_a"] = receivers[i]
        pairs[key]["receiver_b"] = receivers[j]
    alpha_row = None
    if mat.size and mat.shape[1] >= 2 and mat.shape[0] >= 2:
        sample = Sample(mat[:, 0], mat[:, 0], rater_matrix=mat)
        ka = KrippendorffAlpha(alpha=alpha, margin=gamma, level="interval").evaluate(sample, rs)
        alpha_row = ka.as_row()
        alpha_row["optional"] = True
        alpha_row["point_alpha"] = krippendorff_alpha(mat, level="interval")
    passed = bool(pairs) and all(p.get("passed") for p in pairs.values())
    return {
        "passed": passed,
        "n_items": len(items),
        "receivers": receivers,
        "receivers_all": receivers_all,
        "cue_inert": cue_inert,
        "pairs": pairs,
        "krippendorff_alpha": alpha_row,
        "pass_rule": "each non-inert receiver pair: SpearmanRho AND ConcordanceCC CI lower > 0.80",
    }


def _variant_values(rd: dict[str, Any], keys: tuple[str, ...]) -> list[float]:
    out = []
    for k in keys:
        if k in rd and rd[k] is not None:
            out.append(float(rd[k]))
    return out


def analyze_e5b(
    rows: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci_level: float = 0.95,
    cv_threshold: float = 0.15,
) -> dict[str, Any]:
    rs = Resampler(n_boot=n_boot, n_perm=max(10, n_boot), ci_level=ci_level, seed=seed)
    pos_cv = []
    scr_cv = []
    for rec in rows:
        rd = rec.get("r_d_norm") or rec
        pos = _variant_values(rd, PROBE_VARIANTS)
        if len(pos) >= 2:
            pos_cv.append(per_item_cv(pos))
        scr = _variant_values(rd, SCRAMBLED_VARIANTS)
        if len(scr) < 2:
            # allow a packed list
            packed = rd.get("scrambled_variants")
            if isinstance(packed, (list, tuple)):
                scr = [float(x) for x in packed]
        if len(scr) >= 2:
            scr_cv.append(per_item_cv(scr))

    def _mean_cv_ci(cvs: list[float]) -> dict[str, Any]:
        arr = np.asarray(cvs, float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return {
                "n": int(arr.size),
                "mean_cv": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "passed": False,
                "reason": "n<3 finite CVs",
            }

        def stat(block):
            return float(np.mean(block))

        est, lo, hi = rs.bca_ci_rows(stat, arr)
        return {
            "n": int(arr.size),
            "mean_cv": est,
            "ci_low": lo,
            "ci_high": hi,
            "threshold": cv_threshold,
            "passed": bool(np.isfinite(hi) and hi < cv_threshold),
        }

    positive = _mean_cv_ci(pos_cv)
    scrambled = _mean_cv_ci(scr_cv) if scr_cv else {
        "n": 0, "passed": False, "reason": "no scrambled variants",
    }
    # Negative control must NOT pass the CV bar.
    scrambled_ok = (scrambled.get("n", 0) >= 3) and (not scrambled.get("passed"))
    passed = bool(positive.get("passed") and scrambled_ok)
    return {
        "passed": passed,
        "positive": positive,
        "scrambled_negative_control": scrambled,
        "scrambled_must_not_pass": True,
        "scrambled_control_ok": scrambled_ok,
        "pass_rule": (
            f"BCa CI upper of mean CV({'+'.join(PROBE_VARIANTS)}) < {cv_threshold}; "
            "scrambled variants must not pass that bar"
        ),
    }


def analyze_e5(
    cue_rows: list[dict[str, Any]],
    rd_rows: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    gamma: float = 0.80,
    cv_threshold: float = 0.15,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    e5a = analyze_e5a(
        cue_rows, n_boot=n_boot, n_perm=n_perm, seed=seed, alpha=alpha,
        gamma=gamma, ci_level=ci_level,
    )
    e5b = analyze_e5b(
        rd_rows, n_boot=n_boot, seed=seed, ci_level=ci_level, cv_threshold=cv_threshold,
    )
    return {
        "experiment": "E5",
        "note": "Not construct-validity E5. Isolation: did not read/write experiments/experiment1/results/cue_*.jsonl.",
        "e5a_receiver": e5a,
        "e5b_encoder": e5b,
        "passed": bool(e5a["passed"] and e5b["passed"]),
    }


def _to_md(report: dict[str, Any]) -> str:
    a, b = report["e5a_receiver"], report["e5b_encoder"]
    lines = [
        "# E5 receiver + encoder stability",
        "",
        f"**Passed (E5a ∧ E5b):** {report['passed']}",
        "",
        "## E5a receivers",
        f"Passed: {a['passed']}  ·  n={a['n_items']}  ·  non-inert={a['receivers']}",
        f"cue_inert (excluded from Spearman/CCC, not a ρ-bar fail): {a.get('cue_inert') or []}",
        "",
    ]
    for key, pair in (a.get("pairs") or {}).items():
        if "spearman_rho" not in pair:
            lines.append(f"- {key}: {pair}")
            continue
        rho, ccc = pair["spearman_rho"], pair["lins_ccc"]
        lines.append(
            f"- {key}: Spearman ρ={rho['estimate']:.3f} CI=[{rho['ci_low']:.3f}, "
            f"{rho['ci_high']:.3f}] {rho['verdict']}; CCC={ccc['estimate']:.3f} "
            f"CI=[{ccc['ci_low']:.3f}, {ccc['ci_high']:.3f}] {ccc['verdict']}; "
            f"pass={pair['passed']}"
        )
    if a.get("krippendorff_alpha"):
        ka = a["krippendorff_alpha"]
        lines.append(
            f"- Krippendorff α (optional): {ka['estimate']:.3f} "
            f"CI=[{ka['ci_low']:.3f}, {ka['ci_high']:.3f}] {ka['verdict']}"
        )
    pos = b["positive"]
    scr = b["scrambled_negative_control"]
    lines += [
        "",
        "## E5b encoder",
        f"Passed: {b['passed']}",
        f"- mean CV original/plain/technical = {pos.get('mean_cv')} "
        f"CI=[{pos.get('ci_low')}, {pos.get('ci_high')}] pass={pos.get('passed')}",
        f"- scrambled negative control mean CV = {scr.get('mean_cv')} "
        f"CI=[{scr.get('ci_low')}, {scr.get('ci_high')}] "
        f"(must NOT pass; control_ok={b.get('scrambled_control_ok')})",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cue-scores", type=Path, default=None)
    parser.add_argument("--rd-scores", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--n-perm", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    results = ensure_results_dir(cfg)
    cue_path = args.cue_scores or (results / "e7_cue_panel.jsonl")
    rd_path = args.rd_scores or (results / "e5_rd_panel.jsonl")
    cue_rows = read_jsonl(cue_path) if cue_path.exists() else []
    rd_rows = read_jsonl(rd_path) if rd_path.exists() else []
    if not cue_rows and not rd_rows:
        raise SystemExit("need e7_cue_panel.jsonl and/or e5_rd_panel.jsonl")
    e5 = cfg.get("e5") or {}
    report = analyze_e5(
        cue_rows,
        rd_rows,
        n_boot=int(args.n_boot if args.n_boot is not None else cfg.get("n_boot", 2000)),
        n_perm=int(args.n_perm if args.n_perm is not None else cfg.get("n_perm", 2000)),
        seed=int(cfg.get("seed", 42)),
        alpha=float(cfg.get("alpha", 0.05)),
        gamma=float(e5.get("agreement_gamma", 0.80)),
        cv_threshold=float(e5.get("encoder_cv_threshold", 0.15)),
        ci_level=float(cfg.get("ci_level", 0.95)),
    )
    out_json = assert_output_isolated(args.out_json or (results / "e5_report.json"))
    out_md = assert_output_isolated(args.out_md or (results / "e5_report.md"))
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    print(f"E5 passed={report['passed']} → {out_json}", flush=True)


if __name__ == "__main__":
    main()
