"""
Load frozen idea-pipeline artifacts (codebook, KenLM, optional boundary detector).

All artifacts are versioned and hash-checked against artifacts/manifest.json
so benchmark runs cannot silently pick up mutated checkpoints.
"""

from __future__ import annotations

import hashlib
import json
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
from creativegainbench.ideas.span_encoder import build_span_encoder
from creativegainbench.metrics.kenlm_compressor import (
    KenLMCompressor,
    load_kenlm_compressor,
)
from creativegainbench.metrics.structural_novelty import ProbeSet


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PACKAGE_ROOT / "artifacts"
CONFIG_PATH = PACKAGE_ROOT / "config" / "benchmark_defaults.toml"


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
    compressor: KenLMCompressor
    probe_set: ProbeSet
    task_battery: list[dict]
    n: int
    boundary_threshold: float
    span_encoder_backend: str = "minilm"
    span_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

    # Back-compat alias used by older call sites.
    @property
    def model(self) -> KenLMCompressor:
        return self.compressor

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
    kenlm_cfg = cfg.get("kenlm", {})
    version_meta = manifest.get("versions", {}).get(version, {})

    codebook_path = root / "codebook" / f"idea_codebook_{version}.pt"
    boundary_path = root / "models" / f"idea_boundary_{version}.pt"
    probes_path = root / "probes" / f"probes_{version}_seed{seed}.json"
    battery_path = root / f"task_battery_{version}.json"

    kenlm_rel = version_meta.get("kenlm_path", f"models/idea_kenlm_{version}.arpa")
    kenlm_path = root / kenlm_rel
    order = int(version_meta.get("kenlm_order", kenlm_cfg.get("order", ideas_cfg.get("n", 3))))

    codebook_state = torch.load(codebook_path, map_location="cpu", weights_only=True)
    centroids = codebook_state["centroids"]
    codebook = IdeaCodebook(centroids=centroids)
    embedding_dim = int(codebook.embedding_dim)

    compressor = load_kenlm_compressor(kenlm_path, order=order)

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
        compressor=compressor,
        probe_set=probe_set,
        task_battery=task_battery,
        n=int(ideas_cfg["n"]),
        boundary_threshold=float(ideas_cfg["boundary_threshold"]),
        span_encoder_backend=str(span_backend),
        span_encoder_model=str(span_model),
        device=device,
    )
    return pipeline.to(device)
