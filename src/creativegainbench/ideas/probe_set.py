"""Frozen probe set P used by ProbeCompressor R_D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ProbeSet:
    strings: List[str]
    seed: int
    strata: List[str]
