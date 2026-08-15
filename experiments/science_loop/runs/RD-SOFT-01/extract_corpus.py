#!/usr/bin/env python3
"""
Extract poetry texts from the poems pg_dump (no Postgres required).

Writes:
  experiments/science_loop/runs/RD-SOFT-01/corpus_by_domain.jsonl
  one JSON object per poem with body, generations, domain, split, ...
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DUMP = REPO / "data" / "poems_20260813T171429Z.sql.gz"
OUT = Path(__file__).resolve().parent / "corpus_by_domain.jsonl"

COPY_RE = re.compile(r"^COPY public\.(\w+)")
MODELS = ("gemma2:2b", "mistral:latest", "llama3.1:8b", "phi4:14b")


def _unescape(field: str) -> str | None:
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


def iter_copy(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
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
                yield current, line.rstrip("\n").split("\t")


def extract(dump: Path = DUMP, out: Path = OUT) -> int:
    poems: dict[str, dict] = {}
    gens: dict[str, dict[str, str]] = {}

    for table, cols in iter_copy(dump):
        if table == "poems":
            pid = _unescape(cols[0])
            poems[pid] = {
                "poem_id": pid,
                "body": _unescape(cols[3]) or "",
                "source": _unescape(cols[5]),
                "prompt": _unescape(cols[10]),
                "llm_output": _unescape(cols[11]),
                "domain": int(cols[12]) if cols[12] not in (r"\N", "") else None,
                "split": _unescape(cols[13]),
            }
        elif table == "generations":
            pid = _unescape(cols[0])
            model = _unescape(cols[1])
            output = _unescape(cols[2]) or ""
            gens.setdefault(pid, {})[model] = output

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for pid, rec in poems.items():
            if rec["domain"] is None or rec["split"] is None:
                continue
            rec["generations"] = {
                m: gens.get(pid, {}).get(m)
                for m in MODELS
                if gens.get(pid, {}).get(m) is not None
            }
            # Prefer generations table; fall back to poems.llm_output as gemma.
            if "gemma2:2b" not in rec["generations"] and rec.get("llm_output"):
                rec["generations"]["gemma2:2b"] = rec["llm_output"]
            fh.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Wrote {n} poems to {out}")
    return n


if __name__ == "__main__":
    extract()
