"""Shared paths / config for construct_validity suite."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

EXP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXP_ROOT.parents[1]


def load_config() -> dict:
    cfg = yaml.safe_load((EXP_ROOT / "config" / "db.yaml").read_text())
    cfg["database_url"] = os.environ.get("DATABASE_URL", cfg["database_url"])
    cfg["artifacts"] = REPO_ROOT / cfg["artifacts_rel"]
    cfg["results_dir"] = EXP_ROOT / "results"
    cfg["logs_dir"] = EXP_ROOT / "logs"
    return cfg
