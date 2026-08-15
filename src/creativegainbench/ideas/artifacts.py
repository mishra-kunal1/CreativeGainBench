"""
Load frozen idea-pipeline artifacts (codebook, deformation context, boundary).

All artifacts are versioned and hash-checked against artifacts/manifest.json
so benchmark runs cannot silently pick up mutated checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

from creativegainbench.ideas.idea_extractor import IdeaBoundaryDetector
from creativegainbench.ideas.idea_ngram import IdeaCodebook
from creativegainbench.ideas.probe_set import ProbeSet
from creativegainbench.ideas.span_encoder import build_span_encoder
from creativegainbench.metrics.deformation import (
    DomainDeformationContext,
    KernelDomainDeformationContext,
    load_kernel_domain_context,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PACKAGE_ROOT / "artifacts"
CONFIG_PATH = PACKAGE_ROOT / "config" / "benchmark_defaults.toml"
POETRY_V2_ROOT = ARTIFACTS_ROOT / "poetry_v2"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


@dataclass
class KernelBackend:
    """Frozen poetry_v2 Parzen R_D backend: per-domain ctx + gate thresholds."""

    sigma: float
    contexts: dict[int, KernelDomainDeformationContext]
    delta_d: dict[int, float]
    meta: dict[str, Any]

    def context_for(self, domain: int) -> KernelDomainDeformationContext:
        if domain not in self.contexts:
            raise KeyError(f"No kernel context for domain {domain}")
        return self.contexts[domain]

    def delta_d_for(self, domain: int, default: float = 0.0) -> float:
        return float(self.delta_d.get(domain, default))


def load_kernel_backend(
    artifacts_dir: Path | None = None,
    *,
    verify_hashes: bool = True,
) -> KernelBackend:
    """
    Load the frozen kernel_parzen backend for poetry_v2.

    Reads kernel_meta.json, kernel_delta_d_thresholds.json, and every
    domain_*_kernel_ctx.pt referenced by the meta. When verify_hashes is set
    and kernel_manifest.json exists, refuses to load mutated frozen assets.
    """
    root = artifacts_dir or POETRY_V2_ROOT
    meta_path = root / "kernel_meta.json"
    thr_path = root / "kernel_delta_d_thresholds.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run scripts/build_kernel_poetry_contexts.py first."
        )
    meta = json.loads(meta_path.read_text())
    thresholds_raw = json.loads(thr_path.read_text()) if thr_path.exists() else {}

    manifest_path = root / "kernel_manifest.json"
    if verify_hashes and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for rel, expected in manifest.get("files", {}).items():
            fpath = root / rel
            if not fpath.exists():
                raise FileNotFoundError(f"Missing kernel artifact: {fpath}")
            actual = _sha256_file(fpath)
            if actual != expected:
                raise RuntimeError(
                    f"Kernel artifact hash mismatch for {rel}: "
                    f"expected {expected}, got {actual}. "
                    "Refusing to load mutated frozen assets."
                )

    contexts: dict[int, KernelDomainDeformationContext] = {}
    missing: list[str] = []
    for d_str, info in meta.get("domains", {}).items():
        d = int(d_str)
        ctx_name = info.get("kernel_ctx", f"domain_{d}_kernel_ctx.pt")
        ctx_path = root / ctx_name
        if ctx_path.exists():
            contexts[d] = load_kernel_domain_context(ctx_path)
        else:
            missing.append(ctx_name)
    if missing:
        raise FileNotFoundError(
            "Missing kernel context files (gitignored; rebuild locally):\n  "
            + "\n  ".join(missing)
            + "\nRun: PYTHONPATH=src python scripts/build_kernel_poetry_contexts.py "
            "--device cuda"
        )

    delta_d = {
        int(k): float(v.get("delta_d_95", 0.0))
        for k, v in thresholds_raw.items()
        if isinstance(v, dict)
    }
    return KernelBackend(
        sigma=float(meta.get("sigma", 1.0)),
        contexts=contexts,
        delta_d=delta_d,
        meta=meta,
    )


def load_frozen_probe_set(path: str | Path, seed: int) -> ProbeSet:
    with open(path) as f:
        data = json.load(f)
    if data["seed"] != seed:
        raise AssertionError(
            f"Probe set seed mismatch — protocol violation "
            f"(expected {seed}, got {data['seed']})"
        )
    return ProbeSet(
        strings=list(data["strings"]),
        seed=int(data["seed"]),
        strata=list(data.get("strata", [])),
    )


def load_task_battery(path: str | Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return list(data["tasks"])


@dataclass
class IdeaPipeline:
    """Frozen idea pipeline used by all metrics."""

    version: str
    seed: int
    span_encoder: nn.Module
    boundary_detector: IdeaBoundaryDetector | None
    codebook: IdeaCodebook
    deformation_ctx: DomainDeformationContext
    probe_set: ProbeSet
    task_battery: list[dict]
    n: int
    boundary_threshold: float
    span_encoder_backend: str = "minilm"
    span_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

    def to(self, device: str) -> "IdeaPipeline":
        self.device = device
        self.span_encoder = self.span_encoder.to(device)
        if self.boundary_detector is not None:
            self.boundary_detector = self.boundary_detector.to(device)
        self.codebook = IdeaCodebook(centroids=self.codebook.centroids.to(device))
        return self


def _verify_manifest(manifest: dict, version: str, artifacts_root: Path) -> None:
    entry = manifest.get("versions", {}).get(version)
    if entry is None:
        raise FileNotFoundError(f"Artifact version {version!r} missing from manifest")
    for rel, expected in entry.get("files", {}).items():
        path = artifacts_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact file: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Artifact hash mismatch for {rel}: expected {expected}, got {actual}. "
                "Refusing to load mutated frozen assets."
            )


def load_artifacts(
    version: str | None = None,
    device: str = "cpu",
    artifacts_root: Path | None = None,
    verify_hashes: bool = True,
) -> IdeaPipeline:
    root = artifacts_root or ARTIFACTS_ROOT
    cfg = load_config()
    version = version or cfg["artifacts"]["version"]
    seed = int(cfg["artifacts"]["seed"])

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `prepare-artifacts` first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if verify_hashes:
        _verify_manifest(manifest, version, root)

    ideas_cfg = cfg["ideas"]
    ngram_cfg = cfg.get("count_ngram", {})
    version_meta = manifest.get("versions", {}).get(version, {})

    codebook_path = root / "codebook" / f"idea_codebook_{version}.pt"
    boundary_path = root / "models" / f"idea_boundary_{version}.pt"
    probes_path = root / "probes" / f"probes_{version}_seed{seed}.json"
    battery_path = root / f"task_battery_{version}.json"

    ctx_rel = version_meta.get(
        "deformation_ctx_path", f"models/deformation_ctx_{version}.pkl"
    )
    ctx_path = root / ctx_rel
    order = int(
        version_meta.get("ngram_order", ngram_cfg.get("order", ideas_cfg.get("n", 3)))
    )

    codebook_state = torch.load(codebook_path, map_location="cpu", weights_only=True)
    centroids = codebook_state["centroids"]
    codebook = IdeaCodebook(centroids=centroids)
    embedding_dim = int(codebook.embedding_dim)

    with open(ctx_path, "rb") as f:
        deformation_ctx = pickle.load(f)
    if not isinstance(deformation_ctx, DomainDeformationContext):
        raise TypeError(f"Expected DomainDeformationContext in {ctx_path}")
    if deformation_ctx.order != order:
        # Prefer on-disk ctx order; warn via attribute consistency only.
        pass

    boundary_detector: IdeaBoundaryDetector | None = None
    if boundary_path.exists():
        boundary_detector = IdeaBoundaryDetector(hidden_dim=embedding_dim)
        boundary_detector.load_state_dict(
            torch.load(boundary_path, map_location="cpu", weights_only=True)
        )
        boundary_detector.eval()

    span_backend = version_meta.get(
        "span_encoder", ideas_cfg.get("span_encoder", "minilm")
    )
    span_model = version_meta.get(
        "span_encoder_model",
        ideas_cfg.get(
            "span_encoder_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    span_encoder = build_span_encoder(
        span_backend,
        model_name=span_model,
        embedding_dim=embedding_dim,
        device=device,
    )
    enc_dim = int(getattr(span_encoder, "embedding_dim", embedding_dim))
    if enc_dim != embedding_dim:
        raise RuntimeError(
            f"Span encoder dim {enc_dim} != codebook dim {embedding_dim}. "
            "Re-run prepare-artifacts with the same encoder."
        )

    probe_set = load_frozen_probe_set(probes_path, seed=seed)
    task_battery = load_task_battery(battery_path)

    pipeline = IdeaPipeline(
        version=version,
        seed=seed,
        span_encoder=span_encoder,
        boundary_detector=boundary_detector,
        codebook=codebook,
        deformation_ctx=deformation_ctx,
        probe_set=probe_set,
        task_battery=task_battery,
        n=int(ideas_cfg["n"]),
        boundary_threshold=float(ideas_cfg["boundary_threshold"]),
        span_encoder_backend=str(span_backend),
        span_encoder_model=str(span_model),
        device=device,
    )
    return pipeline.to(device)
