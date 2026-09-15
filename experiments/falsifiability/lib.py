"""Shared paths, config, isolation, and JSONL helpers for the E5/E7 suite."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

EXP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXP_ROOT.parents[1]
RESULTS_DIR = EXP_ROOT / "results"
LOGS_DIR = EXP_ROOT / "logs"

# Construct-validity E4 reads this glob; this suite must never touch it.
EXPERIMENT1_CUE_GLOB = REPO_ROOT / "experiments" / "experiment1" / "results"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ARMS = ("matched", "cross", "random", "irrelevant")
CONTROLS = ("cross", "random", "irrelevant")
PROBE_VARIANTS = ("original", "plain", "technical")
SCRAMBLED_VARIANTS = ("scrambled_0", "scrambled_1", "scrambled_2")
PHASE_B_DOMAINS = (
    "scientific_proposal",
    "creative_writing",
    "mathematical_proof",
)


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (EXP_ROOT / "config.toml")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    db = os.environ.get("DATABASE_URL", cfg.get("db_url"))
    cfg["db_url"] = db
    cfg["artifacts"] = REPO_ROOT / cfg.get("artifacts_rel", "src/creativegainbench/artifacts/poetry_v2")
    cfg["results_dir"] = RESULTS_DIR
    cfg["logs_dir"] = LOGS_DIR
    return cfg


def assert_output_isolated(path: Path | str) -> Path:
    """Refuse any write under experiment1/results (especially cue_*.jsonl)."""
    path = Path(path).resolve()
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        rel = path
    posix = rel.as_posix()
    if "experiments/experiment1/results" in posix or path.is_relative_to(EXPERIMENT1_CUE_GLOB):
        raise RuntimeError(
            f"Isolation: refusing to write {path}. Falsifiability outputs must "
            "live under experiments/falsifiability/ — never "
            "experiments/experiment1/results/cue_*.jsonl."
        )
    return path


def ensure_results_dir(cfg: dict | None = None) -> Path:
    d = Path((cfg or load_config())["results_dir"])
    assert_output_isolated(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = assert_output_isolated(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = assert_output_isolated(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def annotator_domain(item: dict[str, Any]) -> str:
    d = item.get("domain")
    if isinstance(d, str) and d and not d.isdigit():
        return d
    return "poetry"
