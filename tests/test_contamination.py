"""Contamination / protocol guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from creativegainbench.utils.contamination import (
    assert_no_probe_overlap,
    filter_contaminated,
    load_probe_hashes,
    text_hash,
    write_exclusion_manifest,
)

PROBES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "creativegainbench"
    / "artifacts"
    / "probes"
    / "probes_v1_seed42.json"
)


def test_probe_hashes_nonempty():
    hs = load_probe_hashes(PROBES)
    assert len(hs) >= 50


def test_filter_removes_probes():
    import json

    probes = json.loads(PROBES.read_text())["strings"]
    banned = load_probe_hashes(PROBES)
    kept = filter_contaminated(probes + ["totally unique held-out prompt xyz"], banned)
    assert probes[0] not in kept
    assert any("unique held-out" in t for t in kept)


def test_assert_overlap_raises():
    import json

    probes = json.loads(PROBES.read_text())["strings"]
    banned = {text_hash(probes[0])}
    with pytest.raises(RuntimeError):
        assert_no_probe_overlap([probes[0]], banned)


def test_exclusion_manifest_accounting(tmp_path):
    import json

    probes = json.loads(PROBES.read_text())["strings"]
    banned = load_probe_hashes(PROBES)
    raw = [probes[0], "unique text a", "unique text a", "unique text b"]
    kept = filter_contaminated(raw, banned)
    seen: set[str] = set()
    unique: list[str] = []
    for t in kept:
        h = text_hash(t)
        if h in seen:
            continue
        seen.add(h)
        unique.append(t)

    train_raw = len(raw)
    after_filter = len(kept)
    train_kept = len(unique)
    out = tmp_path / "contamination_test.json"
    write_exclusion_manifest(
        out,
        probe_hashes=banned,
        eval_hashes=set(),
        train_raw=train_raw,
        train_kept=train_kept,
        train_dropped_contamination=train_raw - after_filter,
        train_dropped_deduplication=after_filter - train_kept,
    )
    payload = json.loads(out.read_text())
    assert payload["train_texts_raw"] == 4
    assert payload["train_texts_dropped_contamination"] == 1
    assert payload["train_texts_dropped_deduplication"] == 1
    assert payload["train_texts_kept"] == 2
    assert (
        payload["train_texts_raw"]
        == payload["train_texts_kept"]
        + payload["train_texts_dropped_contamination"]
        + payload["train_texts_dropped_deduplication"]
    )


def test_exclusion_manifest_rejects_bad_accounting(tmp_path):
    with pytest.raises(ValueError, match="accounting mismatch"):
        write_exclusion_manifest(
            tmp_path / "bad.json",
            probe_hashes=set(),
            eval_hashes=set(),
            train_raw=10,
            train_kept=5,
            train_dropped_contamination=3,
            train_dropped_deduplication=1,
        )
