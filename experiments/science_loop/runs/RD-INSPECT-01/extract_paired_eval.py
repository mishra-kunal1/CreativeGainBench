#!/usr/bin/env python3
"""Parse the poems pg_dump into paired eval R_D rows. No Postgres, no LLM."""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DUMP = REPO / "data" / "poems_20260813T171429Z.sql.gz"
OUT = Path(__file__).resolve().parent / "paired_eval.jsonl"

MODELS = ("gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b")
COPY_RE = re.compile(r"^COPY public\.(\w+)")


def _unescape_copy(field: str) -> str | None:
    if field == r"\N":
        return None
    out: list[str] = []
    i = 0
    while i < len(field):
        if field[i] == "\\" and i + 1 < len(field):
            nxt = field[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def _split_copy_row(line: str) -> list[str]:
    return line.rstrip("\n").split("\t")


def iter_copy_tables(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        current = None
        for line in fh:
            m = COPY_RE.match(line)
            if m:
                current = m.group(1)
                continue
            if current and line.startswith("\\."):
                current = None
                continue
            if current:
                yield current, _split_copy_row(line)


def payload_fields(payload: dict) -> dict:
    return {
        "r_d_raw": payload.get("r_d_raw"),
        "r_d_norm": payload.get("r_d_norm"),
        "r_d_gate": payload.get("r_d_gate"),
        "y_n_symbols": payload.get("y_n_symbols"),
        "delta_d_norm": payload.get("delta_d_norm"),
        "feasibility_bit": payload.get("feasibility_bit"),
        "length_clip_chars": payload.get("length_clip_chars"),
    }


def extract(dump: Path = DUMP, out: Path = OUT) -> list[dict]:
    poems: dict[str, dict] = {}
    scores: dict[str, dict[str, dict]] = defaultdict(dict)

    for table, cols in iter_copy_tables(dump):
        if table == "poems":
            pid = _unescape_copy(cols[0])
            poems[pid] = {
                "poem_id": pid,
                "source": _unescape_copy(cols[5]),
                "domain": int(cols[12]) if cols[12] not in (r"\N", "") else None,
                "split": _unescape_copy(cols[13]),
                "body_chars": len(_unescape_copy(cols[3]) or ""),
                "llm_output_chars": len(_unescape_copy(cols[11]) or ""),
            }
        elif table == "scores":
            pid = _unescape_copy(cols[0])
            side = _unescape_copy(cols[1])
            version = _unescape_copy(cols[2])
            if version != "poetry_v2":
                continue
            raw = _unescape_copy(cols[3])
            payload = json.loads(raw) if raw else {}
            scores[pid][side] = payload_fields(payload)

    rows = []
    for pid, meta in poems.items():
        if meta["split"] != "eval":
            continue
        by_side = scores.get(pid, {})
        human = by_side.get("human")
        if human is None or human.get("r_d_norm") is None:
            continue
        rec = {
            **meta,
            "human": human,
            "models": {},
        }
        for model in MODELS:
            if model in by_side and by_side[model].get("r_d_norm") is not None:
                rec["models"][model] = by_side[model]
        if rec["models"]:
            rows.append(rec)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(rows)} eval pairs to {out}")
    return rows


if __name__ == "__main__":
    extract()
