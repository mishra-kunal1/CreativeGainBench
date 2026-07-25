"""Contamination / protocol guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from creativegainbench.utils.contamination import (
    assert_no_probe_overlap,
    filter_contaminated,
    load_probe_hashes,
    text_hash,
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
