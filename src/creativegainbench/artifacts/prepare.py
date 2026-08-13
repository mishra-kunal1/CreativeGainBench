"""
Prepare frozen v1 artifacts: idea codebook, CountNgram deformation context,
boundary detector, and manifest hashes.

Training corpus excludes frozen probes P and held-out eval prompts.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

from creativegainbench.ideas.idea_extractor import (
    IdeaBoundaryDetector,
    default_sentence_splitter,
    extract_ideas,
)
from creativegainbench.ideas.idea_ngram import IdeaCodebook
from creativegainbench.ideas.span_encoder import build_span_encoder
from creativegainbench.metrics.deformation import build_domain_context
from creativegainbench.utils.build_eval_prompts import (
    _formalmath_prompt,
    _rinobench_prompt,
)
from creativegainbench.utils.contamination import (
    filter_contaminated,
    load_probe_hashes,
    text_hash,
    write_exclusion_manifest,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "benchmark_defaults.toml"
REPO_DATA = PACKAGE_ROOT.parent.parent / "data"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_strings(path: Path, key: str = "strings") -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if key in data:
        return list(data[key])
    if "domains" in data:
        out: list[str] = []
        for vals in data["domains"].values():
            out.extend(vals)
        return out
    return []


def _load_optional_data_texts() -> list[str]:
    texts: list[str] = []
    for name in (
        "infinity_chat_creative.jsonl",
        "rinobench_low_novelty.jsonl",
        "formalmath.jsonl",
    ):
        path = REPO_DATA / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines()[:2000]:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rb = _rinobench_prompt(item)
            if rb:
                texts.append(rb)
                continue
            fm = _formalmath_prompt(item)
            if fm:
                texts.append(fm)
                continue
            for key in ("prompt", "input", "text"):
                if isinstance(item.get(key), str) and item[key].strip():
                    texts.append(item[key].strip())
                    break
            else:
                msgs = item.get("messages")
                if isinstance(msgs, list):
                    for m in msgs:
                        if m.get("role") == "user" and isinstance(m.get("content"), str):
                            texts.append(m["content"].strip())
                            break
    return texts


def _kmeans(
    points: torch.Tensor, k: int, seed: int, iters: int = 25
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = points.size(0)
    if n == 0:
        raise ValueError("No idea embeddings to fit codebook")
    idx = torch.randint(0, n, (k,), generator=g)
    centroids = points[idx].clone()
    for _ in range(iters):
        dists = torch.cdist(points, centroids)
        assign = torch.argmin(dists, dim=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centroids[j] = points[mask].mean(dim=0)
            else:
                centroids[j] = points[torch.randint(0, n, (1,), generator=g).item()]
    norms = torch.linalg.vector_norm(centroids, dim=1, keepdim=True).clamp_min(1e-8)
    return centroids / norms


def _collect_idea_embeddings(
    texts: list[str],
    encoder: nn.Module,
    boundary: IdeaBoundaryDetector | None,
    boundary_threshold: float,
) -> torch.Tensor:
    embeds: list[torch.Tensor] = []
    for text in texts:
        ideas = extract_ideas(
            text,
            span_encoder=encoder,
            boundary_detector=boundary,
            sentence_splitter=default_sentence_splitter,
            boundary_threshold=boundary_threshold,
        )
        for idea in ideas:
            embeds.append(idea.embedding.detach())
    if not embeds:
        return encoder(texts)
    return torch.stack(embeds, dim=0)


def _build_train_corpus(root: Path, version: str, seed: int) -> tuple[list[str], set[str], set[str]]:
    probes_path = root / "probes" / f"probes_{version}_seed{seed}.json"
    heldout_path = root / f"heldout_prompts_{version}.json"
    train_path = root / f"train_corpus_{version}.json"

    probe_hashes = load_probe_hashes(probes_path)
    eval_hashes: set[str] = set()
    if heldout_path.exists():
        for t in _load_json_strings(heldout_path):
            eval_hashes.add(text_hash(t))

    banned = set(probe_hashes) | set(eval_hashes)
    raw: list[str] = []
    if train_path.exists():
        raw.extend(_load_json_strings(train_path))
    raw.extend(_load_optional_data_texts())

    before = len(raw)
    kept = filter_contaminated(raw, banned)
    after_filter = len(kept)
    seen: set[str] = set()
    unique: list[str] = []
    for t in kept:
        h = text_hash(t)
        if h in seen:
            continue
        seen.add(h)
        unique.append(t)

    train_kept = len(unique)
    write_exclusion_manifest(
        root / f"contamination_{version}.json",
        probe_hashes=probe_hashes,
        eval_hashes=eval_hashes,
        train_raw=before,
        train_kept=train_kept,
        train_dropped_contamination=before - after_filter,
        train_dropped_deduplication=after_filter - train_kept,
    )
    if len(unique) < 20:
        raise RuntimeError(
            f"Train corpus too small after decontamination ({len(unique)}). "
            f"Add texts to {train_path}."
        )
    return unique, probe_hashes, eval_hashes


def prepare_artifacts(
    version: str | None = None,
    seed: int | None = None,
    artifacts_root: Path | None = None,
) -> Path:
    root = artifacts_root or ARTIFACTS_ROOT
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    version = version or cfg["artifacts"]["version"]
    seed = int(seed if seed is not None else cfg["artifacts"]["seed"])
    ideas_cfg = cfg["ideas"]
    ngram_cfg = cfg.get("count_ngram", {})

    vocab_size = int(ideas_cfg["vocab_size"])
    boundary_threshold = float(ideas_cfg["boundary_threshold"])
    order = int(ngram_cfg.get("order", ideas_cfg.get("n", 3)))
    span_backend = str(ideas_cfg.get("span_encoder", "minilm"))
    span_model = str(
        ideas_cfg.get(
            "span_encoder_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    configured_dim = int(ideas_cfg["embedding_dim"])

    probes_path = root / "probes" / f"probes_{version}_seed{seed}.json"
    if not probes_path.exists():
        raise FileNotFoundError(f"Missing probe set: {probes_path}")

    train_texts, probe_hashes, eval_hashes = _build_train_corpus(root, version, seed)
    print(
        f"Train corpus: {len(train_texts)} texts "
        f"(excluded {len(probe_hashes)} probes + {len(eval_hashes)} eval hashes)"
    )

    torch.manual_seed(seed)
    encoder = build_span_encoder(
        span_backend,
        model_name=span_model,
        embedding_dim=configured_dim,
        device="cpu",
    )
    embedding_dim = int(getattr(encoder, "embedding_dim", configured_dim))

    boundary = IdeaBoundaryDetector(hidden_dim=embedding_dim)
    with torch.no_grad():
        nn.init.zeros_(boundary.boundary_head.weight)
        nn.init.constant_(boundary.boundary_head.bias, 2.0)

    print(f"Encoding idea spans with {span_backend} ({span_model})...")
    embeds = _collect_idea_embeddings(
        train_texts, encoder, boundary, boundary_threshold
    )
    centroids = _kmeans(embeds, k=vocab_size, seed=seed)
    codebook = IdeaCodebook(centroids=centroids)

    with open(probes_path) as f:
        probe_strings = list(json.load(f)["strings"])

    print(f"Building CountNgram deformation context (order={order})...")
    deformation_ctx = build_domain_context(
        domain=0,
        train_texts=train_texts,
        probe_texts=probe_strings,
        span_encoder=encoder,
        codebook=codebook,
        boundary_detector=boundary,
        sentence_splitter=default_sentence_splitter,
        boundary_threshold=boundary_threshold,
        order=order,
    )

    codebook_path = root / "codebook" / f"idea_codebook_{version}.pt"
    ctx_path = root / "models" / f"deformation_ctx_{version}.pkl"
    boundary_path = root / "models" / f"idea_boundary_{version}.pt"
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    ctx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"centroids": centroids.detach().cpu()}, codebook_path)
    torch.save(boundary.state_dict(), boundary_path)
    with open(ctx_path, "wb") as f:
        pickle.dump(deformation_ctx, f)

    for stale_name in (f"idea_sequence_lm_{version}.pt",):
        stale = root / "models" / stale_name
        if stale.exists():
            stale.unlink()

    corpus_rel = f"train_corpus_{version}.json"
    ctx_rel = f"models/deformation_ctx_{version}.pkl"

    file_map = {
        f"probes/probes_{version}_seed{seed}.json": _sha256_file(probes_path),
        f"task_battery_{version}.json": _sha256_file(root / f"task_battery_{version}.json"),
        corpus_rel: _sha256_file(root / corpus_rel),
        f"codebook/idea_codebook_{version}.pt": _sha256_file(codebook_path),
        ctx_rel: _sha256_file(ctx_path),
        f"models/idea_boundary_{version}.pt": _sha256_file(boundary_path),
    }
    if (root / f"heldout_prompts_{version}.json").exists():
        file_map[f"heldout_prompts_{version}.json"] = _sha256_file(
            root / f"heldout_prompts_{version}.json"
        )
    if (root / f"contamination_{version}.json").exists():
        file_map[f"contamination_{version}.json"] = _sha256_file(
            root / f"contamination_{version}.json"
        )

    manifest = {
        "versions": {
            version: {
                "seed": seed,
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
                "span_encoder": span_backend,
                "span_encoder_model": span_model,
                "compressor": "count_ngram",
                "ngram_order": order,
                "deformation_ctx_path": ctx_rel,
                "train_corpus_size": len(train_texts),
                "files": file_map,
            }
        }
    }
    manifest_path = root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Prepared artifacts version={version} seed={seed}")
    print(f"  span encoder: {span_backend} ({span_model}), dim={embedding_dim}")
    print(f"  codebook: {codebook_path}")
    print(f"  deformation ctx: {ctx_path}")
    print(f"  boundary: {boundary_path}")
    print(f"  manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    prepare_artifacts()


if __name__ == "__main__":
    main()
