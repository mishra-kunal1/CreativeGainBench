"""Shared helpers for experiment1 components."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

EXP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXP_ROOT.parents[1]


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (EXP_ROOT / "config.toml")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    db = os.environ.get("DATABASE_URL", cfg.get("db_url"))
    cfg["db_url"] = db
    cfg["artifacts"] = REPO_ROOT / cfg["artifacts_rel"]
    cfg["results_dir"] = EXP_ROOT / "results"
    cfg["logs_dir"] = EXP_ROOT / "logs"
    return cfg
