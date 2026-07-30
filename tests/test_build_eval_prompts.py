"""Eval prompt builder domain tagging."""

from __future__ import annotations

import json
from pathlib import Path

from creativegainbench.utils.build_eval_prompts import _write_records


def test_write_records_preserves_domain(tmp_path: Path):
    records = [
        {"prompt": "Write a story.", "domain": "creative_writing"},
        {"prompt": "Propose a study.", "domain": "scientific_proposal"},
        {"prompt": "Prove n is even.", "domain": "mathematical_proof"},
    ]
    out = tmp_path / "eval_all_domains.jsonl"
    _write_records(out, records)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["domain"] for r in rows] == [
        "creative_writing",
        "scientific_proposal",
        "mathematical_proof",
    ]
    assert "mixed" not in {r["domain"] for r in rows}
