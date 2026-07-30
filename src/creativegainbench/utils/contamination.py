"""
Protocol contamination guards.

Frozen probe set P and evaluation prompts must never enter the KenLM / codebook
training corpus. Eval prompts must never equal probe strings.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


def normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def load_probe_hashes(probes_path: Path) -> set[str]:
    with open(probes_path) as f:
        data = json.load(f)
    return {text_hash(s) for s in data.get("strings", [])}


def filter_contaminated(
    texts: Iterable[str],
    banned_hashes: set[str],
) -> list[str]:
    kept: list[str] = []
    for t in texts:
        if text_hash(t) in banned_hashes:
            continue
        kept.append(t)
    return kept


def assert_no_probe_overlap(prompts: Iterable[str], probe_hashes: set[str]) -> None:
    overlaps = [p for p in prompts if text_hash(p) in probe_hashes]
    if overlaps:
        raise RuntimeError(
            f"Protocol violation: {len(overlaps)} eval prompt(s) overlap the "
            f"frozen probe set P (example: {overlaps[0][:80]!r}...)"
        )


def write_exclusion_manifest(
    path: Path,
    *,
    probe_hashes: set[str],
    eval_hashes: set[str],
    train_raw: int,
    train_kept: int,
    train_dropped_contamination: int,
    train_dropped_deduplication: int,
) -> None:
    """
    Write contamination audit metadata for the KenLM / codebook train corpus.

    Invariant:
      train_raw == train_kept + train_dropped_contamination + train_dropped_deduplication
    """
    if (
        train_raw
        != train_kept + train_dropped_contamination + train_dropped_deduplication
    ):
        raise ValueError(
            "Train corpus accounting mismatch: "
            f"raw={train_raw}, kept={train_kept}, "
            f"dropped_contamination={train_dropped_contamination}, "
            f"dropped_deduplication={train_dropped_deduplication}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_probe_hashes": len(probe_hashes),
        "n_eval_hashes": len(eval_hashes),
        "train_texts_raw": train_raw,
        "train_texts_kept": train_kept,
        "train_texts_dropped_contamination": train_dropped_contamination,
        "train_texts_dropped_deduplication": train_dropped_deduplication,
        "probe_hash_sample": sorted(probe_hashes)[:5],
        "eval_hash_sample": sorted(eval_hashes)[:5],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
