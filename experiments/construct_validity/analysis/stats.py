from __future__ import annotations

import math
import random
from typing import Sequence


def bootstrap_ci(
    diffs: Sequence[float], n_boot: int = 2000, seed: int = 42
) -> tuple[float, tuple[float, float]]:
    if not diffs:
        return 0.0, (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return sum(diffs) / n, (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3 or n != len(ys):
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def ranks(vals: Sequence[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[idx]
