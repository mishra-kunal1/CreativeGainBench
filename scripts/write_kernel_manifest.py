#!/usr/bin/env python3
"""Write kernel_manifest.json for frozen poetry_v2 kernel JSON artifacts.

Domain *_kernel_ctx.pt files are regenerable (~100MB) and gitignored; they are
listed under local_contexts but not hashed in-repo. After
`scripts/build_kernel_poetry_contexts.py`, re-run this script or rely on the
build script's full local manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "src/creativegainbench/artifacts/poetry_v2"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((ART / "kernel_meta.json").read_text())
    files = {
        "kernel_meta.json": sha256(ART / "kernel_meta.json"),
        "kernel_delta_d_thresholds.json": sha256(ART / "kernel_delta_d_thresholds.json"),
    }
    local_contexts = []
    for info in meta.get("domains", {}).values():
        name = info["kernel_ctx"]
        local_contexts.append(name)
        p = ART / name
        if p.exists():
            # Optional: hash locally when present (not required in git).
            files[name] = sha256(p)

    # In-repo verification only for JSON shipped with the package.
    committed = {
        k: v
        for k, v in files.items()
        if k.endswith(".json")
    }
    (ART / "kernel_manifest.json").write_text(
        json.dumps(
            {
                "backend": "kernel_parzen",
                "sigma": meta["sigma"],
                "max_bank": meta.get("max_bank"),
                "files": committed,
                "local_contexts": local_contexts,
                "note": (
                    "Rebuild domain_*_kernel_ctx.pt with "
                    "scripts/build_kernel_poetry_contexts.py --device cuda"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"wrote kernel_manifest.json ({len(committed)} committed hashes, "
        f"{len(local_contexts)} local contexts)"
    )


if __name__ == "__main__":
    main()
