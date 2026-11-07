"""
Build per-domain negative constructs for one-sided δ_D calibration.

Construct types: probe_paraphrase, shuffle, pad, exact_h_member, ood.
Does NOT use human-vs-model scores or z* to define negatives.

F10: length-match constructs to the domain's eval median char length so
δ_D is not dominated by long padded strings.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.connection import connect, run_migrations  # noqa: E402
from db.queries import clear_calibration, fetch_poems_by_split  # noqa: E402
from lib import load_config  # noqa: E402
from metrics.pipeline import feasibility_bit, load_stack  # noqa: E402
from creativegainbench.utils.text_length import length_match as _length_match  # noqa: E402


def _median_chars(bodies: list[str]) -> int:
    lens = [len((b or "").strip()) for b in bodies if (b or "").strip()]
    if not lens:
        return 400
    return int(statistics.median(lens))


def _paraphrase(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\bAnd\b", "Plus", t)
    t = re.sub(r"\bthe\b", "a", t, count=3, flags=re.IGNORECASE)
    return f"In other words:\n{t}"


def _shuffle_lines(text: str, rng: random.Random) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return text
    rng.shuffle(lines)
    return "\n".join(lines)


def _pad_length_matched(text: str, target: int) -> str:
    """Pad with filler lines only up to the domain target length (F10)."""
    t = text.rstrip()
    filler = "(padding)"
    while len(t) < target:
        t = f"{t}\n{filler}"
        if t.count(filler) > 200:
            break
    return _length_match(t, target)


def _pick_length_matched(
    items: list[tuple], target: int, k: int, rng: random.Random, *, tol: float = 0.35
) -> list[tuple]:
    """Prefer poems whose body length is near target; fall back to random."""
    if not items:
        return []
    lo, hi = target * (1 - tol), target * (1 + tol)
    near = [it for it in items if lo <= len((it[1] or "").strip()) <= hi]
    pool = near if len(near) >= max(3, k // 2) else items
    return rng.sample(pool, min(k, len(pool)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain", type=int, default=None)
    args = parser.parse_args()

    run_migrations()
    cfg = load_config()
    stack = load_stack(device=args.device)
    rng = random.Random(args.seed)

    domains = sorted(stack.domain_ctx.keys())
    if args.domain is not None:
        domains = [args.domain]

    with connect() as conn:
        clear_calibration(conn, args.domain)
        all_by_domain = {
            d: fetch_poems_by_split(conn, d, "eval")
            + fetch_poems_by_split(conn, d, "train")
            for d in sorted(stack.domain_ctx.keys())
        }

        for d in domains:
            if d not in stack.domain_ctx:
                print(f"skip domain {d}: no ctx", flush=True)
                continue
            train = fetch_poems_by_split(conn, d, "train")
            probe = fetch_poems_by_split(conn, d, "probe")
            eval_poems = fetch_poems_by_split(conn, d, "eval")
            if len(probe) < 3 or len(train) < 5:
                print(f"skip domain {d}: train={len(train)} probe={len(probe)}", flush=True)
                continue

            target = _median_chars([b for _, b, *_ in (eval_poems or train)])
            print(f"domain {d}: length-match target_chars={target}", flush=True)

            constructs: list[tuple[str, str | None, str]] = []

            for pid, body, *_ in _pick_length_matched(
                probe, target, args.per_type, rng
            ):
                constructs.append(
                    (
                        "probe_paraphrase",
                        str(pid),
                        _length_match(_paraphrase(body), target),
                    )
                )

            pool = train + eval_poems
            for pid, body, *_ in _pick_length_matched(pool, target, args.per_type, rng):
                constructs.append(
                    (
                        "shuffle",
                        str(pid),
                        _length_match(_shuffle_lines(body, rng), target),
                    )
                )

            for pid, body, *_ in _pick_length_matched(train, target, args.per_type, rng):
                constructs.append(
                    ("pad", str(pid), _pad_length_matched(body, target))
                )

            for pid, body, *_ in _pick_length_matched(train, target, args.per_type, rng):
                constructs.append(
                    ("exact_h_member", str(pid), _length_match(body, target))
                )

            others = [od for od in all_by_domain if od != d and all_by_domain[od]]
            ood_items: list[tuple] = []
            for od in others:
                ood_items.extend(all_by_domain[od])
            if ood_items:
                for pid, body, *_ in _pick_length_matched(
                    ood_items, target, args.per_type, rng
                ):
                    constructs.append(
                        ("ood", str(pid), _length_match(body, target))
                    )

            n_ok = 0
            for ctype, source_id, text in constructs:
                if not feasibility_bit(text):
                    continue
                result = stack.score_r_d(text, d)
                syms = stack.to_idea_symbols(text)
                conn.execute(
                    """
                    INSERT INTO delta_d_calibration
                      (domain_cluster, construct_type, source_poem_id, text,
                       n_symbols, r_d_raw, r_d_norm)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        d,
                        ctype,
                        source_id,
                        text[:8000],
                        len(syms),
                        result.r_d_raw,
                        result.r_d_norm,
                    ),
                )
                n_ok += 1
            print(f"domain {d}: inserted {n_ok} constructs", flush=True)

    print("DONE build_negative_bank")


if __name__ == "__main__":
    main()
