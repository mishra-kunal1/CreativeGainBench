"""Deterministic original / plain / technical probe banks + scrambled negative control.

Phase A register-shift templates follow construct-validity ``_paraphrase``
(no LLM). Writes ``results/probe_pairs.json``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    REPO_ROOT,
    assert_output_isolated,
    ensure_results_dir,
    load_config,
)

from creativegainbench.utils.text_length import length_match

# Same substitutions as experiments/construct_validity/calibration/build_negative_bank.py
_PARAPHRASE_AND = (re.compile(r"\bAnd\b"), "Plus")
_PARAPHRASE_THE = re.compile(r"\bthe\b", re.IGNORECASE)

SYNTHETIC_PROBES = [
    "A volta that turns grief into a misdrawn map legend.",
    "Each couplet rewrites the previous metaphor's physics.",
    "Two meters braid until a third rhythm appears.",
    "Sunrise-as-hope restated in conventional nature lyric.",
    "Broken lines with topic drift and unfinished clauses.",
    "Prove the inductive step without naming the hypothesis.",
    "A sensor network that reuses bus routes as sample paths.",
    "A scene where rival chefs invent a dish neither can claim.",
]


def paraphrase_plain(text: str) -> str:
    """Register-shift toward plain vocab (F10-style ``_paraphrase``)."""
    t = (text or "").strip()
    t = _PARAPHRASE_AND[0].sub(_PARAPHRASE_AND[1], t)
    t = _PARAPHRASE_THE.sub("a", t, count=3)
    t = re.sub(r"\butilize\b", "use", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdemonstrate\b", "show", t, flags=re.IGNORECASE)
    t = re.sub(r"\bestablish\b", "prove", t, flags=re.IGNORECASE)
    return f"In other words:\n{t}"


def paraphrase_technical(text: str) -> str:
    """Register-shift toward technical vocab (same concept, different words)."""
    t = (text or "").strip()
    t = re.sub(r"\buse\b", "utilize", t, flags=re.IGNORECASE)
    t = re.sub(r"\bshow\b", "demonstrate", t, flags=re.IGNORECASE)
    t = re.sub(r"\bprove\b", "establish", t, flags=re.IGNORECASE)
    t = re.sub(r"\bwrite\b", "compose", t, flags=re.IGNORECASE)
    return f"Technical restatement:\n{t}"


def scramble_probe(text: str, rng: random.Random) -> str:
    tokens = re.findall(r"\S+", text or "")
    if len(tokens) < 2:
        tokens = list(tokens) + ["scrambled", "tokens", "without", "structure"]
    rng.shuffle(tokens)
    return " ".join(tokens)


def _load_poetry_probes(artifacts: Path) -> list[str] | None:
    texts: list[str] = []
    meta_path = artifacts / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for d_str in meta.get("domains", {}):
        pkl = artifacts / f"domain_{int(d_str)}_ctx.pkl"
        if not pkl.exists():
            continue
        try:
            with open(pkl, "rb") as f:
                ctx = pickle.load(f)
        except Exception:
            continue
        probes = getattr(ctx, "probe_texts", None) or []
        texts.extend(str(t) for t in probes if t)
    return texts or None


def _load_packaged_probes() -> list[str] | None:
    path = REPO_ROOT / "src" / "creativegainbench" / "artifacts" / "probes" / "probes_v1_seed42.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    strings = data.get("strings") or []
    return [s for s in strings if isinstance(s, str) and s.strip()] or None


def load_original_probes(cfg: dict, *, synthetic: bool) -> tuple[list[str], str]:
    if synthetic:
        return list(SYNTHETIC_PROBES), "synthetic"
    poetry = _load_poetry_probes(Path(cfg["artifacts"]))
    if poetry:
        return poetry, "poetry_v2_ctx"
    packaged = _load_packaged_probes()
    if packaged:
        return packaged, "probes_v1_seed42"
    return list(SYNTHETIC_PROBES), "synthetic_fallback"


def build_probe_pairs(
    originals: list[str],
    *,
    seed: int = 42,
    length_tol: float = 0.2,
    source: str = "unknown",
) -> dict[str, Any]:
    rng = random.Random(seed)
    original, plain, technical = [], [], []
    scrambled_variants: dict[str, list[str]] = {"scrambled_0": [], "scrambled_1": [], "scrambled_2": []}
    for text in originals:
        t = (text or "").strip()
        if not t:
            continue
        target = len(t)
        original.append(t)
        plain.append(length_match(paraphrase_plain(t), target, tol=length_tol))
        technical.append(length_match(paraphrase_technical(t), target, tol=length_tol))
        for key in scrambled_variants:
            scrambled_variants[key].append(
                length_match(scramble_probe(t, rng), target, tol=length_tol)
            )
    return {
        "seed": seed,
        "source": source,
        "n": len(original),
        "length_tol": length_tol,
        "note": "H / codebook / MiniLM / delta_D stay frozen; only probe encodings are rebuilt.",
        "banks": {
            "original": original,
            "plain": plain,
            "technical": technical,
            "scrambled": scrambled_variants["scrambled_0"],
            **scrambled_variants,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    originals, source = load_original_probes(cfg, synthetic=args.synthetic)
    if args.limit is not None:
        originals = originals[: args.limit]
    payload = build_probe_pairs(
        originals, seed=seed, length_tol=float(cfg.get("length_tol", 0.2)), source=source
    )
    out = args.out or (ensure_results_dir(cfg) / "probe_pairs.json")
    assert_output_isolated(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote n={payload['n']} source={source} → {out}", flush=True)


if __name__ == "__main__":
    main()
