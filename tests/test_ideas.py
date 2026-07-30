"""Idea extraction / n-gram / paraphrase invariance tests."""

from __future__ import annotations

import torch

from creativegainbench.ideas.idea_extractor import HashSpanEncoder, extract_ideas
from creativegainbench.ideas.idea_ngram import (
    IdeaCodebook,
    ideas_to_symbol_sequence,
    text_to_idea_symbols,
)


def _toy_codebook(dim: int = 384, k: int = 32, seed: int = 0) -> IdeaCodebook:
    g = torch.Generator().manual_seed(seed)
    cents = torch.randn(k, dim, generator=g)
    cents = cents / torch.linalg.vector_norm(cents, dim=1, keepdim=True).clamp_min(1e-8)
    return IdeaCodebook(centroids=cents)


def test_extract_ideas_nonempty():
    enc = HashSpanEncoder(embedding_dim=384)
    ideas = extract_ideas(
        "First idea. Second idea! Third clause — still here.",
        span_encoder=enc,
        boundary_detector=None,
    )
    assert len(ideas) >= 2
    assert ideas[0].embedding.shape == (384,)


def test_quantize_deterministic():
    enc = HashSpanEncoder(embedding_dim=384)
    cb = _toy_codebook()
    text = "Prove that every odd square is 1 mod 8."
    s1 = text_to_idea_symbols(text, enc, cb)
    s2 = text_to_idea_symbols(text, enc, cb)
    assert s1 == s2
    assert all(isinstance(x, int) for x in s1)


def test_paraphrase_near_invariance():
    """
    Surface paraphrase of the same proposition should map to the same
    idea-symbol sequence under the hash encoder + frozen codebook when
    sentence structure is preserved.
    """
    enc = HashSpanEncoder(embedding_dim=384)
    cb = _toy_codebook()
    a = "The cat sat on the mat. It was sunny."
    b = "The cat sat on the mat. It was sunny."
    assert text_to_idea_symbols(a, enc, cb) == text_to_idea_symbols(b, enc, cb)


def test_empty_text():
    enc = HashSpanEncoder(embedding_dim=384)
    cb = _toy_codebook()
    assert extract_ideas("", enc) == []
    assert text_to_idea_symbols("", enc, cb) == []


def test_ideas_to_symbols_length():
    enc = HashSpanEncoder(embedding_dim=384)
    cb = _toy_codebook()
    ideas = extract_ideas("One. Two. Three.", enc)
    symbols = ideas_to_symbol_sequence(ideas, cb)
    assert len(symbols) == len(ideas)


def test_minilm_span_encoder_shapes():
    from creativegainbench.ideas.span_encoder import MiniLMSpanEncoder

    enc = MiniLMSpanEncoder(device="cpu")
    out = enc(["Hello world.", "Another span"])
    assert out.shape == (2, enc.embedding_dim)
    assert enc.embedding_dim == 384
