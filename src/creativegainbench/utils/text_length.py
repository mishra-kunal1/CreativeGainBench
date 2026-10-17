"""F10 length-matching: force text into [target*(1-tol), target*(1+tol)] chars.

Shared by construct-validity negative-bank construction and the falsifiability
y-arm / probe constructors. Shorter → repeat lines; longer → truncate on a
line boundary when that still stays inside the band, else a hard char cut.
"""

from __future__ import annotations


def length_match(text: str, target: int, *, tol: float = 0.2) -> str:
    t = (text or "").strip()
    if target <= 0:
        return t
    lo = max(1, int(target * (1.0 - tol)))
    hi = max(lo, int(target * (1.0 + tol)))
    if len(t) < lo:
        lines = [ln for ln in t.splitlines() if ln.strip()] or [t or "_"]
        acc = t
        i = 0
        while len(acc) < lo:
            acc = (acc + "\n" + lines[i % len(lines)]) if acc else lines[i % len(lines)]
            i += 1
            if i > 500:
                break
        t = acc
    if len(t) > hi:
        cut = t[:hi]
        if "\n" in cut:
            lined = cut.rsplit("\n", 1)[0]
            # Do not undo padding: if the line cut falls below lo, keep the hard cut.
            if len(lined.strip()) >= lo:
                cut = lined
        t = cut.strip()
    if len(t) < lo:
        filler = " lorem"
        while len(t) < lo:
            t = t + filler
            if len(t) > hi + len(filler):
                break
        if len(t) > hi:
            t = t[:hi].strip()
    return t
