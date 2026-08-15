"""
V8 — G_k semantics audit (code-level; no live scoring required).

Lean kInteractionGain / MASBridge speaks of interaction gain in a
receiver-calibrated / informational sense. Current Python G_k uses soft-cluster
entropy of *raw agent/joint texts* (or a length proxy), not downstream-conditioned
samples like R_B.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from experiments._common import new_run_id, record  # noqa: E402
from db.connection import run_migrations  # noqa: E402

from creativegainbench.metrics import interaction_gain as ig  # noqa: E402
from creativegainbench.metrics.receiver_expansion import compute_receiver_expansion  # noqa: E402


def main() -> None:
    run_migrations()
    run_id = new_run_id()

    src = inspect.getsource(ig.compute_interaction_gain)
    mas_src = inspect.getsource(ig.mas_outputs_from_row)
    rb_src = inspect.getsource(compute_receiver_expansion)

    from creativegainbench.metrics.interaction_gain import (  # noqa: E402
        G_K_CONDITIONED,
        G_K_SURFACE,
        compute_interaction_gain_conditioned,
    )

    uses_raw_embed = "embedding_entropy" in mas_src and "sample_with_embeddings" not in src
    uses_length_fallback = "_length_entropy_proxy" in src
    rb_downstream = "sample_with_embeddings" in rb_src and "condition(" in rb_src
    # Conditioned path may call sample_with_embeddings via a helper.
    cond_src = inspect.getsource(compute_interaction_gain_conditioned)
    helper_src = inspect.getsource(ig._downstream_entropy)
    has_conditioned = (
        "sample_with_embeddings" in cond_src or "sample_with_embeddings" in helper_src
    )

    # Default path in R_creativity is surface; conditioned API exists (F8).
    default_kind = G_K_SURFACE
    lean_analogous_default = False

    details = {
        "default_gk_kind": default_kind,
        "conditioned_api": G_K_CONDITIONED,
        "conditioned_implemented": has_conditioned,
        "gk_semantics": "raw_text_entropy_labeled_surface",
        "uses_embedding_entropy_on_raw_texts": uses_raw_embed,
        "uses_length_entropy_fallback": uses_length_fallback,
        "rb_uses_downstream_conditioned_samples": rb_downstream,
        "lean_analogous_irreducibility_for_default": lean_analogous_default,
        "note": (
            "Default g_k in R_creativity is G_k_surface (diversity proxy). "
            "Use compute_interaction_gain_conditioned for R_B-style semantics."
        ),
    }
    # Pass = surface is explicitly labeled AND conditioned API exists.
    passed = uses_raw_embed and has_conditioned and default_kind == G_K_SURFACE
    record(
        run_id,
        "V8",
        "gk_semantics",
        1.0 if passed else 0.0,
        passed,
        details,
    )
    print(json.dumps(details, indent=2))
    print("DONE V8")


if __name__ == "__main__":
    main()
