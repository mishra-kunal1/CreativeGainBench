"""
Build held-out evaluation prompt JSONL files.

Priority:
1) HuggingFace-downloaded subsets in data/ (if present), decontaminated vs probes
2) Packaged held-out prompt bank (artifacts/heldout_prompts_v1.json)

Always writes prompt-normalized JSONL and refuses probe overlap.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from creativegainbench.utils.contamination import (
    assert_no_probe_overlap,
    load_probe_hashes,
    text_hash,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA = PACKAGE_ROOT.parent.parent / "data"
ARTIFACTS = PACKAGE_ROOT / "artifacts"
HELDOUT_BANK = ARTIFACTS / "heldout_prompts_v1.json"
PROBES = ARTIFACTS / "probes" / "probes_v1_seed42.json"

SEED = 42
SUBSET_SIZE = 300


def _write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            row = {"prompt": record["prompt"], "domain": record["domain"]}
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(records)} prompts → {path}")


def _write_prompts(path: Path, prompts: list[str], domain: str) -> None:
    _write_records(
        path,
        [{"prompt": p, "domain": domain} for p in prompts],
    )


def _from_infinity(path: Path, n: int, rng: random.Random) -> list[str]:
    items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng.shuffle(items)
    out: list[str] = []
    for item in items:
        msg = next(
            (m["content"] for m in item.get("messages", []) if m.get("role") == "user"),
            None,
        )
        if msg:
            out.append(msg)
        if len(out) >= n:
            break
    return out


def _rinobench_prompt(item: dict) -> str | None:
    idea = item.get("research_idea")
    if isinstance(idea, dict):
        parts = [
            idea.get("objective", ""),
            idea.get("problem_statement", ""),
            idea.get("solution_approach", ""),
        ]
        text = "\n\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
        return text or None
    if isinstance(idea, str) and idea.strip():
        return idea.strip()
    for key in ("prompt", "input", "text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _from_rinobench(path: Path, n: int, rng: random.Random) -> list[str]:
    items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng.shuffle(items)
    out: list[str] = []
    for item in items:
        text = _rinobench_prompt(item)
        if text:
            out.append(text)
        if len(out) >= n:
            break
    return out


def _formalmath_prompt(item: dict) -> str | None:
    for key in ("refined_statement", "formal_statement", "informal_statement", "problem", "prompt"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return f"Prove or rigorously justify the following.\n\n{val.strip()}"
    return None


def _from_formalmath(path: Path, n: int, rng: random.Random) -> list[str]:
    items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    filtered = [
        it
        for it in items
        if "Algebra" in str(it.get("domain", ""))
        or "Number Theory" in str(it.get("domain", ""))
    ]
    rng.shuffle(filtered)
    out: list[str] = []
    for item in filtered:
        text = _formalmath_prompt(item)
        if text:
            out.append(text)
        if len(out) >= n:
            break
    return out


def _from_heldout_bank() -> dict[str, list[str]]:
    data = json.loads(HELDOUT_BANK.read_text())
    return {k: list(v) for k, v in data["domains"].items()}


def main() -> None:
    rng = random.Random(SEED)
    probe_hashes = load_probe_hashes(PROBES)
    out_dir = REPO_DATA / "subset"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = _from_heldout_bank()
    sources = {
        "creative_writing": REPO_DATA / "infinity_chat_creative.jsonl",
        "scientific_proposal": REPO_DATA / "rinobench_low_novelty.jsonl",
        "mathematical_proof": REPO_DATA / "formalmath.jsonl",
    }
    extractors = {
        "creative_writing": _from_infinity,
        "scientific_proposal": _from_rinobench,
        "mathematical_proof": _from_formalmath,
    }
    filenames = {
        "creative_writing": "infinity_chat_subset.jsonl",
        "scientific_proposal": "rinobench_subset.jsonl",
        "mathematical_proof": "formalmath_subset.jsonl",
    }

    all_eval: list[dict[str, str]] = []
    for domain, src in sources.items():
        prompts: list[str] = []
        if src.exists():
            try:
                prompts = extractors[domain](src, SUBSET_SIZE, rng)
                print(f"{domain}: loaded {len(prompts)} from {src.name}")
            except Exception as e:
                print(f"{domain}: failed HF extract ({e}); using held-out bank")
                prompts = []
        if len(prompts) < 20:
            prompts = list(bank[domain])
            print(f"{domain}: using packaged held-out bank ({len(prompts)})")

        # Decontaminate vs probes.
        prompts = [p for p in prompts if text_hash(p) not in probe_hashes]
        assert_no_probe_overlap(prompts, probe_hashes)
        all_eval.extend({"prompt": p, "domain": domain} for p in prompts)
        _write_prompts(out_dir / filenames[domain], prompts, domain)

    # Combined eval file — preserve per-row domain for downstream grouping.
    rng.shuffle(all_eval)
    _write_records(out_dir / "eval_all_domains.jsonl", all_eval)

    meta = {
        "seed": SEED,
        "n_total": len(all_eval),
        "probe_hashes": len(probe_hashes),
        "files": list(filenames.values()) + ["eval_all_domains.jsonl"],
    }
    (out_dir / "eval_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Done.", meta)


if __name__ == "__main__":
    main()
