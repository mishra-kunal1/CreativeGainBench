"""Build the four item-yoked y arms for E7.

Writes ``experiments/falsifiability/results/y_panel.jsonl`` (never experiment1).

Data sources, in order:
  * ``--from-jsonl`` rows with prompt + body/y + domain
  * Phase A: Postgres ``poems`` eval split (clear error if missing)
  * Phase B: ``data/subset/eval_all_domains.jsonl`` (+ optional ``--y-jsonl``)
  * ``--synthetic`` smoke bank (unit tests; no Postgres)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    ARMS,
    PHASE_B_DOMAINS,
    REPO_ROOT,
    annotator_domain,
    assert_output_isolated,
    ensure_results_dir,
    load_config,
    write_jsonl,
)

from creativegainbench.utils.text_length import length_match

FILLER = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam"
)

# Phase A: fluent verse on a different topic (not the item's prompt).
IRRELEVANT_VERSE = (
    "The harvest moon forgets the names of rivers and counts only the stones "
    "that learned to sleep facing north.\n"
    "A kettle of unclaimed letters boils on the sill; each steam-curl is a "
    "town that never asked to be mapped.\n"
    "I loan my shadow to a fence post and it comes back speaking in rust."
)

# Phase B math: domain-plausible but wrong method (LLM-free template).
IRRELEVANT_MATH_INDUCTION = (
    "Apply the triangle inequality to bound the residual of the inductive "
    "step: ||a_{n+1}|| ≤ ||a_n|| + ||a_{n+1} - a_n||, then invoke a random "
    "Toeplitz spectral-radius estimate that does not use the inductive hypothesis."
)
IRRELEVANT_MATH_DEFAULT = (
    "Diagonalize an unrelated circulant matrix and quote the spectral theorem; "
    "this does not address the claimed statement."
)
IRRELEVANT_SCIENCE = (
    "We will run an underpowered crossover on an unrelated biomarker panel "
    "and interpret any p < 0.05 as mechanistic confirmation."
)
IRRELEVANT_WRITING = (
    "Meanwhile, in a suburban kitchen, someone reheats leftovers and narrates "
    "the microwave's hum as if it were the plot."
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text or "")


def make_random_y(text: str, target: int, rng: random.Random, *, tol: float = 0.2) -> str:
    """Length-matched token shuffle / filler with no task structure."""
    tokens = _tokens(text)
    if len(tokens) < 4:
        tokens = FILLER.split()
    rng.shuffle(tokens)
    return length_match(" ".join(tokens), target, tol=tol)


def make_irrelevant_y(
    item: dict[str, Any],
    target: int,
    rng: random.Random,
    *,
    pool: Sequence[dict[str, Any]] | None = None,
    phase: str = "a",
    tol: float = 0.2,
) -> tuple[str, dict[str, Any]]:
    """Domain-plausible but wrong y, then F10 length-match to |y_matched|."""
    domain = str(item.get("domain") or item.get("domain_cluster") or "")
    prompt = (item.get("prompt") or "").lower()
    meta: dict[str, Any] = {"kind": "template"}

    if phase == "b" and domain in ("mathematical_proof", "math"):
        if "induct" in prompt:
            body = IRRELEVANT_MATH_INDUCTION
            meta["kind"] = "math_triangle_inequality_on_induction"
        else:
            body = IRRELEVANT_MATH_DEFAULT
            meta["kind"] = "math_wrong_method"
    elif phase == "b" and domain in ("scientific_proposal", "science"):
        body = IRRELEVANT_SCIENCE
        meta["kind"] = "science_wrong_design"
    elif phase == "b" and domain in ("creative_writing", "writing"):
        body = IRRELEVANT_WRITING
        meta["kind"] = "writing_wrong_topic"
    else:
        # Phase A: fluent verse from a different topic / other cluster when possible.
        other = None
        if pool:
            cand = [
                r
                for r in pool
                if r.get("item_id") != item.get("item_id")
                and str(r.get("domain_cluster", r.get("domain")))
                != str(item.get("domain_cluster", item.get("domain")))
            ]
            if cand:
                other = rng.choice(cand)
        if other is not None and (other.get("y_matched") or other.get("body")):
            body = other.get("y_matched") or other.get("body")
            meta = {
                "kind": "other_cluster_verse",
                "source_id": other.get("item_id"),
            }
        else:
            body = IRRELEVANT_VERSE
            meta["kind"] = "fluent_verse_other_topic"
    return length_match(body, target, tol=tol), meta


def derange_within_domain(
    items: Sequence[dict[str, Any]],
    *,
    domain_key: str = "domain",
) -> dict[str, str]:
    """Fixed permutation, no replacement, inside each domain (cycle shift).

    Returns map item_id → donor item_id. Domains of size 1 fall back to a
    global derangement (flagged later as cross_same_domain=false).
    """
    by_d: dict[str, list[str]] = defaultdict(list)
    ids: list[str] = []
    for it in items:
        iid = str(it["item_id"])
        ids.append(iid)
        by_d[str(it.get(domain_key, it.get("domain_cluster", "")))].append(iid)
    mapping: dict[str, str] = {}
    leftovers: list[str] = []
    for group in by_d.values():
        if len(group) < 2:
            leftovers.extend(group)
            continue
        for src, dst in zip(group, group[1:] + group[:1]):
            mapping[src] = dst
    if leftovers:
        if len(leftovers) == 1:
            # borrow from the global cycle, excluding self if possible
            others = [i for i in ids if i != leftovers[0]]
            mapping[leftovers[0]] = others[0] if others else leftovers[0]
        else:
            for src, dst in zip(leftovers, leftovers[1:] + leftovers[:1]):
                mapping[src] = dst
    return mapping


def stratified_sample(
    items: Sequence[dict[str, Any]],
    n: int,
    rng: random.Random,
    *,
    domain_key: str = "domain",
) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(items):
        return list(items)
    by_d: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_d[str(it.get(domain_key, it.get("domain_cluster", "")))].append(it)
    per = max(1, n // max(len(by_d), 1))
    out: list[dict[str, Any]] = []
    for d in sorted(by_d):
        pool = list(by_d[d])
        rng.shuffle(pool)
        out.extend(pool[: min(per, len(pool))])
    if len(out) < n:
        rest = [it for it in items if it not in out]
        rng.shuffle(rest)
        out.extend(rest[: n - len(out)])
    rng.shuffle(out)
    return out[:n]


def synthetic_records(n: int, seed: int = 42, phase: str = "a") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if phase == "b":
        domains: list[Any] = list(PHASE_B_DOMAINS)
    else:
        domains = [0, 1, 2]
    # At least 2 items per domain so cross is a true same-domain derangement.
    n = max(n, 2 * len(domains))
    records = []
    for i in range(n):
        d = domains[i % len(domains)]
        if phase == "b" and d == "mathematical_proof" and i % 2 == 0:
            prompt = "Show that the sum of the first n positive integers equals n(n+1)/2 by induction."
            body = (
                "Base case n=1 holds. Assume true for k; then the k+1 sum is "
                "k(k+1)/2 + (k+1) = (k+1)(k+2)/2."
            )
        elif phase == "b" and d == "scientific_proposal":
            prompt = "Propose a low-cost sensor network for urban air-quality monitoring."
            body = (
                "We reuse existing bus routes as mobile sample paths and identify "
                "hotspots via a change-of-support estimator, not a restated RCT."
            )
        elif phase == "b":
            prompt = "Write a scene where two rival chefs invent a dish neither can claim."
            body = (
                "They plate silence first: a consommé of names they refuse to say, "
                "then a crust that shatters into two incompatible recipes."
            )
        else:
            prompt = f"Write a short lyric about domain-cluster {d} object {i}."
            body = (
                f"Cluster {d} poem {i}: the volta turns the river inside out;\n"
                "each couplet rewrites the previous metaphor's physics."
            )
        rec = {
            "item_id": f"syn-{phase}-{i}",
            "prompt": prompt,
            "body": body,
            "domain": d if isinstance(d, str) else str(d),
            "domain_cluster": d if isinstance(d, int) else None,
            "phase": phase,
        }
        records.append(rec)
    rng.shuffle(records)
    return records


def load_jsonl_records(path: Path, *, phase: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        body = rec.get("body") or rec.get("y") or rec.get("output") or rec.get("text")
        prompt = rec.get("prompt") or rec.get("q") or rec.get("input")
        if not prompt:
            continue
        iid = str(rec.get("item_id") or rec.get("id") or rec.get("poem_id") or f"row-{len(rows)}")
        domain = rec.get("domain") or rec.get("domain_cluster") or "unknown"
        rows.append(
            {
                "item_id": iid,
                "prompt": prompt,
                "body": body,
                "domain": domain if isinstance(domain, str) else str(domain),
                "domain_cluster": rec.get("domain_cluster", domain if isinstance(domain, int) else None),
                "phase": phase,
            }
        )
    return rows


def load_phase_a_postgres(cfg: dict) -> list[dict[str, Any]]:
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError(
            "Phase A needs psycopg to read Postgres poems, or pass --synthetic / "
            "--from-jsonl. Install psycopg and set DATABASE_URL."
        ) from e
    db = cfg.get("db_url")
    if not db:
        raise RuntimeError("No DATABASE_URL / db_url. Use --synthetic for smoke.")
    try:
        with psycopg.connect(db) as conn:
            rows = conn.execute(
                """
                SELECT id, prompt, body, domain_cluster
                FROM poems
                WHERE split = %s AND domain_cluster IS NOT NULL
                  AND prompt IS NOT NULL AND body IS NOT NULL
                ORDER BY id
                """,
                (cfg.get("split", "eval"),),
            ).fetchall()
    except Exception as e:
        raise RuntimeError(
            f"Postgres unavailable ({e}). Use --synthetic or --from-jsonl."
        ) from e
    if not rows:
        raise RuntimeError("Postgres poems query returned no eval rows. Use --synthetic.")
    out = []
    for pid, prompt, body, domain in rows:
        out.append(
            {
                "item_id": str(pid),
                "prompt": prompt,
                "body": body,
                "domain": str(domain),
                "domain_cluster": int(domain) if domain is not None else None,
                "phase": "a",
            }
        )
    return out


def load_phase_b_eval(
    cfg: dict,
    *,
    y_jsonl: Path | None = None,
) -> list[dict[str, Any]]:
    eval_path = REPO_ROOT / "data" / "subset" / "eval_all_domains.jsonl"
    if not eval_path.exists():
        raise RuntimeError(
            f"Missing {eval_path}. Run create-subset, pass --from-jsonl, or --synthetic."
        )
    prompts = load_jsonl_records(eval_path, phase="b")
    prompts = [r for r in prompts if r.get("domain") in PHASE_B_DOMAINS]
    y_map: dict[str, str] = {}
    if y_jsonl and y_jsonl.exists():
        for rec in load_jsonl_records(y_jsonl, phase="b"):
            if rec.get("body"):
                y_map[rec["item_id"]] = rec["body"]
                y_map[rec["prompt"]] = rec["body"]
    for rec in prompts:
        body = y_map.get(rec["item_id"]) or y_map.get(rec["prompt"]) or rec.get("body")
        rec["body"] = body
    return prompts


def build_panel(
    records: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    n: int | None = None,
    phase: str = "a",
    length_tol: float = 0.2,
    clip: int = 4000,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items = [dict(r) for r in records if r.get("prompt") and r.get("body")]
    if not items:
        raise RuntimeError(
            "No records with both prompt and body/y. For Phase B pass --y-jsonl "
            "or --synthetic; for Phase A use Postgres or --from-jsonl."
        )
    domain_key = "domain_cluster" if phase == "a" and items[0].get("domain_cluster") is not None else "domain"
    if n is not None:
        items = stratified_sample(items, n, rng, domain_key=domain_key)
    # Stable order for the frozen permutation.
    items.sort(key=lambda r: str(r["item_id"]))
    by_id = {str(r["item_id"]): r for r in items}
    cross_map = derange_within_domain(items, domain_key=domain_key)

    # Attach y_matched for irrelevant donor lookup.
    for r in items:
        r["y_matched"] = (r.get("body") or "")[:clip]

    panel: list[dict[str, Any]] = []
    for rec in items:
        iid = str(rec["item_id"])
        y_m = rec["y_matched"]
        target = len(y_m)
        donor_id = cross_map[iid]
        donor = by_id[donor_id]
        y_cross = length_match((donor.get("body") or "")[:clip], target, tol=length_tol)
        y_rand = make_random_y(y_m, target, rng, tol=length_tol)
        y_irr, irr_meta = make_irrelevant_y(
            rec, target, rng, pool=items, phase=phase, tol=length_tol
        )
        same_domain = str(donor.get(domain_key)) == str(rec.get(domain_key))
        row = {
            "item_id": iid,
            "phase": phase,
            "domain": rec.get("domain"),
            "domain_cluster": rec.get("domain_cluster"),
            "annotator_domain": annotator_domain(rec) if phase == "b" else "poetry",
            "prompt": rec["prompt"],
            "y": {
                "matched": y_m,
                "cross": y_cross,
                "random": y_rand,
                "irrelevant": y_irr,
            },
            "y_source": {
                "matched": "self",
                "cross": donor_id,
                "random": "token_shuffle",
                "irrelevant": irr_meta,
            },
            "cross_same_domain": same_domain,
            "char_len_matched": target,
            "length_tol": length_tol,
            "seed": seed,
        }
        panel.append(row)
    return panel


def length_match_ok(row: dict[str, Any], *, tol: float = 0.2) -> bool:
    target = len(row["y"]["matched"])
    if target <= 0:
        return False
    lo = max(1, int(target * (1.0 - tol)))
    hi = max(lo, int(target * (1.0 + tol)))
    return all(lo <= len(row["y"][arm]) <= hi for arm in ARMS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), default="a")
    parser.add_argument("--limit", type=int, default=None, help="Smoke n (default: config n)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--from-jsonl", type=Path, default=None)
    parser.add_argument("--y-jsonl", type=Path, default=None, help="Phase B matched y's")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    n = args.limit if args.limit is not None else int(cfg["n"])
    tol = float(cfg.get("length_tol", 0.2))
    clip = int(cfg.get("clip", 4000))
    phase = args.phase

    if args.synthetic:
        records = synthetic_records(n, seed=seed, phase=phase)
    elif args.from_jsonl:
        records = load_jsonl_records(args.from_jsonl, phase=phase)
    elif phase == "a":
        records = load_phase_a_postgres(cfg)
    else:
        records = load_phase_b_eval(cfg, y_jsonl=args.y_jsonl)
        if any(not r.get("body") for r in records):
            raise RuntimeError(
                "Phase B eval prompts have no matched y. Pass --y-jsonl or --synthetic."
            )

    panel = build_panel(records, seed=seed, n=n, phase=phase, length_tol=tol, clip=clip)
    out = args.out or (ensure_results_dir(cfg) / "y_panel.jsonl")
    assert_output_isolated(out)
    write_jsonl(out, panel)
    n_len = sum(1 for r in panel if length_match_ok(r, tol=tol))
    print(
        f"wrote {len(panel)} items → {out} (length-match ok {n_len}/{len(panel)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
