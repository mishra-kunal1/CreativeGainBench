"""Falsifiability E5/E7 constructors and analyzers — no Ollama, GPU, or Postgres."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from creativegainbench.metrics.outcome_annotator import (
    DOMAIN_EXEMPLARS,
    EXEMPLARS,
    exemplars_for_domain,
)
from creativegainbench.receivers.ollama_receiver import OllamaReceiverAgent
from creativegainbench.utils.text_length import length_match

FALS = Path(__file__).resolve().parents[1] / "experiments" / "falsifiability"
if str(FALS) not in sys.path:
    sys.path.insert(0, str(FALS))

import analyze_e5  # noqa: E402
import analyze_e7  # noqa: E402
import construct_contributions as cc  # noqa: E402
import construct_probe_pairs as pp  # noqa: E402
import lib as flib  # noqa: E402
from score_cue_panel import (  # noqa: E402
    freeze_z_stars,
    math_validity_prompt,
    parse_validity_bit,
)
from score_rd_encoder import per_item_cv  # noqa: E402


N_BOOT = 60
N_PERM = 60
ARMS = ("matched", "cross", "random", "irrelevant")
RECEIVERS = ("gemma2:2b", "llama3.1:8b", "phi4:14b")


def test_length_match_tolerance():
    short = "hello"
    out = length_match(short, 100, tol=0.2)
    assert 80 <= len(out) <= 120
    long = "x" * 500
    out2 = length_match(long, 100, tol=0.2)
    assert len(out2) <= 120


def test_isolation_refuses_experiment1_cue_glob():
    forbidden = (
        flib.REPO_ROOT
        / "experiments"
        / "experiment1"
        / "results"
        / "cue_gemma2_2b.jsonl"
    )
    with pytest.raises(RuntimeError, match="Isolation"):
        flib.assert_output_isolated(forbidden)
    ok = FALS / "results" / "e7_cue_panel.jsonl"
    assert flib.assert_output_isolated(ok) == ok.resolve()


def test_poetry_exemplars_not_overwritten_by_phase_b_banks():
    poetry = EXEMPLARS["novel_structure"][0]
    math_ex = DOMAIN_EXEMPLARS["mathematical_proof"]["novel_structure"][0]
    assert poetry != math_ex
    assert exemplars_for_domain("poetry") is EXEMPLARS
    assert exemplars_for_domain("mathematical_proof") is not EXEMPLARS
    assert EXEMPLARS["novel_structure"][0] == poetry


def test_construct_contributions_four_arms_and_length(tmp_path):
    records = cc.synthetic_records(8, seed=42, phase="a")
    panel = cc.build_panel(records, seed=42, n=8, phase="a", length_tol=0.2)
    assert len(panel) == 8
    for row in panel:
        assert set(row["y"]) == set(ARMS)
        assert cc.length_match_ok(row, tol=0.2)
        assert row["y"]["matched"]
        assert " " in row["y"]["random"] or "\n" in row["y"]["random"]
        assert row["y_source"]["cross"] != row["item_id"] or len(panel) == 1
    assert any(r["cross_same_domain"] for r in panel)
    donors = [r["y_source"]["cross"] for r in panel]
    assert len(donors) == len(set(donors))  # no replacement
    out = tmp_path / "y_panel.jsonl"
    flib.write_jsonl(out, panel)
    assert out.exists()
    assert "experiment1" not in str(out)


def test_construct_contributions_phase_b_math_irrelevant():
    records = cc.synthetic_records(6, seed=42, phase="b")
    panel = cc.build_panel(records, seed=42, n=6, phase="b", length_tol=0.2)
    math_rows = [r for r in panel if r["domain"] == "mathematical_proof"]
    assert math_rows
    kinds = {json.dumps(r["y_source"]["irrelevant"]) for r in math_rows}
    assert any("triangle" in k or "math_" in k for k in kinds)
    for row in panel:
        assert cc.length_match_ok(row, tol=0.2)


def test_construct_probe_pairs_register_shift_and_scrambled():
    payload = pp.build_probe_pairs(pp.SYNTHETIC_PROBES, seed=42)
    banks = payload["banks"]
    for key in ("original", "plain", "technical", "scrambled", "scrambled_0", "scrambled_1", "scrambled_2"):
        assert key in banks
        assert len(banks[key]) == len(banks["original"])
    assert any(b.startswith("In other words:") for b in banks["plain"])
    assert any(b.startswith("Technical restatement:") for b in banks["technical"])
    assert banks["scrambled_0"] != banks["original"]


def _cue_rows_pass(n: int = 24, receiver: str = "llama3.1:8b") -> list[dict]:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        matched = 0.55 + rng.normal(0, 0.02)
        for arm, mu in (("matched", matched), ("cross", 0.12), ("random", 0.08), ("irrelevant", 0.10)):
            cue = mu + rng.normal(0, 0.01)
            rows.append(
                {
                    "item_id": f"i{i}",
                    "receiver": receiver,
                    "arm": arm,
                    "cue": float(cue),
                    "brier_delta": float(max(cue, 0.0) * 400),
                    "bit_length": 800.0 + (0 if arm == "matched" else float(rng.normal(0, 10))),
                }
            )
    return rows


def _cue_rows_fail(n: int = 24, receiver: str = "llama3.1:8b") -> list[dict]:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        base = 0.3 + rng.normal(0, 0.02)
        for arm in ARMS:
            rows.append(
                {
                    "item_id": f"i{i}",
                    "receiver": receiver,
                    "arm": arm,
                    "cue": float(base + rng.normal(0, 0.01)),
                    "brier_delta": 0.2,
                    "bit_length": 800.0,
                }
            )
    return rows


def test_analyze_e7_pass_wiring():
    report = analyze_e7.analyze_e7(
        _cue_rows_pass(),
        primary_receiver="llama3.1:8b",
        n_boot=N_BOOT,
        n_perm=N_PERM,
        seed=42,
    )
    assert report["passed"] is True
    assert report["length"]["passed"] is True
    assert all(c["passed"] for c in report["cue_contrasts"])
    assert all(c["positive_estimate"] for c in report["cue_contrasts"])
    assert all(c["verdict"] == "different" for c in report["cue_contrasts"])


def test_analyze_e7_fail_wiring():
    report = analyze_e7.analyze_e7(
        _cue_rows_fail(),
        primary_receiver="llama3.1:8b",
        n_boot=N_BOOT,
        n_perm=N_PERM,
        seed=42,
    )
    assert report["passed"] is False
    assert not all(c.get("passed") for c in report["cue_contrasts"])


def _e5a_rows(correlated: bool, n: int = 30) -> list[dict]:
    rng = np.random.default_rng(7)
    latent = rng.normal(0.4, 0.15, n)
    rows = []
    for i in range(n):
        for rx in RECEIVERS:
            if correlated:
                cue = float(latent[i] + rng.normal(0, 0.005))
            else:
                cue = float(rng.normal(0.4, 0.15))
            rows.append(
                {
                    "item_id": f"i{i}",
                    "receiver": rx,
                    "arm": "matched",
                    "cue": cue,
                }
            )
    return rows


def _e5b_rows(stable: bool, n: int = 24) -> list[dict]:
    rng = np.random.default_rng(9)
    rows = []
    for i in range(n):
        if stable:
            base = 0.4 + rng.normal(0, 0.01)
            rd = {
                "original": base,
                "plain": base + 0.005,
                "technical": base - 0.005,
                "scrambled_0": float(rng.uniform(0.05, 1.2)),
                "scrambled_1": float(rng.uniform(0.05, 1.2)),
                "scrambled_2": float(rng.uniform(0.05, 1.2)),
            }
        else:
            rd = {
                "original": float(rng.uniform(0.05, 1.0)),
                "plain": float(rng.uniform(0.05, 1.0)),
                "technical": float(rng.uniform(0.05, 1.0)),
                "scrambled_0": 0.4,
                "scrambled_1": 0.41,
                "scrambled_2": 0.39,
            }
        rows.append({"item_id": f"i{i}", "r_d_norm": rd})
    return rows


def test_analyze_e5_pass_wiring():
    report = analyze_e5.analyze_e5(
        _e5a_rows(True),
        _e5b_rows(True),
        n_boot=N_BOOT,
        n_perm=N_PERM,
        seed=42,
    )
    assert report["e5a_receiver"]["passed"] is True
    assert report["e5b_encoder"]["passed"] is True
    assert report["passed"] is True
    assert report["e5b_encoder"]["scrambled_negative_control"]["passed"] is False
    assert report["e5b_encoder"]["scrambled_control_ok"] is True


def test_analyze_e5_fail_wiring():
    report = analyze_e5.analyze_e5(
        _e5a_rows(False),
        _e5b_rows(False),
        n_boot=N_BOOT,
        n_perm=N_PERM,
        seed=42,
    )
    assert report["e5a_receiver"]["passed"] is False
    assert report["e5b_encoder"]["passed"] is False
    assert report["passed"] is False


def test_freeze_z_star_from_matched_only():
    class Ann:
        def annotate(self, text: str) -> int:
            return 0 if "MATCHED_MARK" in text else 3

    panel = [
        {
            "item_id": "a",
            "annotator_domain": "poetry",
            "y": {
                "matched": "MATCHED_MARK lyric",
                "cross": "other poem",
                "random": "lorem ipsum",
                "irrelevant": "harvest moon",
            },
        }
    ]
    z = freeze_z_stars(panel, {"poetry": Ann()})
    assert z["a"] == 0
    assert Ann().annotate(panel[0]["y"]["random"]) == 3


def test_math_validity_prompt_is_llm_proxy_not_lean():
    class Agent:
        condition = OllamaReceiverAgent.condition

    prompt = math_validity_prompt("Prove n(n+1)/2", "base case...", Agent())
    assert "Context:" in prompt
    assert "Lean 4" in prompt
    assert "oracle" in prompt.lower()
    parsed = parse_validity_bit('{"valid": true, "reason": "steps ok"}')
    assert parsed["valid"] is True
    assert parse_validity_bit("not json")["valid"] is None


def test_per_item_cv():
    assert per_item_cv([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    cv = per_item_cv([1.0, 2.0, 3.0])
    assert cv > 0.15


def test_run_all_synthetic_constructors():
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(FALS / "run_all.py"),
            "--phase",
            "a",
            "--limit",
            "6",
            "--synthetic",
            "--skip-score",
        ],
        cwd=str(flib.REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (FALS / "results" / "y_panel.jsonl").exists()
    assert (FALS / "results" / "probe_pairs.json").exists()
