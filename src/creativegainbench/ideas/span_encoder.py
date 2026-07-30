"""
Span encoders for idea extraction.

Default: frozen pretrained MiniLM via sentence-transformers.
Fallback: HashSpanEncoder for offline / CI without model downloads.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from creativegainbench.ideas.idea_extractor import HashSpanEncoder

DEFAULT_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMSpanEncoder(nn.Module):
    """
    Frozen SentenceTransformer wrapper with the same call signature as
    HashSpanEncoder: forward(list[str]) -> (n, d) float tensor.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MINILM,
        device: str = "cpu",
        normalize: bool = True,
    ):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self._st = SentenceTransformer(model_name, device=device)
        self._st.eval()
        for p in self._st.parameters():
            p.requires_grad_(False)
        # Probe dim once.
        if hasattr(self._st, "get_embedding_dimension"):
            dim = self._st.get_embedding_dimension()
        else:
            dim = self._st.get_sentence_embedding_dimension()
        if dim is None:
            probe = self._st.encode(
                ["_"],
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            dim = int(probe.shape[-1])
        self.embedding_dim = int(dim)
        self._device = device

    def to(self, device):  # type: ignore[override]
        device_str = str(device)
        if device_str.startswith("cuda") or device_str == "cpu" or device_str.startswith("mps"):
            self._st.to(device)
            self._device = device_str
        return self

    def forward(self, spans: Sequence[str]) -> torch.Tensor:
        if not spans:
            return torch.zeros(0, self.embedding_dim, dtype=torch.float32)
        with torch.no_grad():
            emb = self._st.encode(
                list(spans),
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        return emb.to(dtype=torch.float32)


def build_span_encoder(
    backend: str = "minilm",
    *,
    model_name: str = DEFAULT_MINILM,
    embedding_dim: int = 384,
    device: str = "cpu",
) -> nn.Module:
    """Factory: backend in {minilm, hash}."""
    backend = (backend or "minilm").lower()
    if backend == "hash":
        return HashSpanEncoder(embedding_dim=embedding_dim)
    if backend in {"minilm", "sentence-transformers", "st"}:
        return MiniLMSpanEncoder(model_name=model_name, device=device)
    raise ValueError(f"Unknown span_encoder backend: {backend!r}")
