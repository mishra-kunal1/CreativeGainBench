"""
KenLM-backed compression-cost estimator over discrete idea-cluster symbols.

Training writes a standard ARPA language model (via `lmplz` when available,
otherwise a Witten–Bell ARPA writer). Evaluation scores sequences with the
open-source KenLM runtime (`kenlm.Model`).
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


LOG10_2 = math.log10(2.0)


def symbol_token(sym: int) -> str:
    """Map idea-cluster ID → KenLM vocabulary token."""
    return f"i{int(sym)}"


def tokens_from_symbols(symbols: Sequence[int]) -> List[str]:
    return [symbol_token(s) for s in symbols]


@dataclass
class KenLMCompressor:
    """Frozen KenLM model used as ProbeCompressor code-length estimator."""

    model_path: Path
    order: int

    def __post_init__(self) -> None:
        import kenlm

        self._kenlm = kenlm
        self._model = kenlm.Model(str(self.model_path))

    @property
    def path(self) -> Path:
        return self.model_path


def log10_to_bits(log10_prob: float) -> float:
    """Convert KenLM log10 probability mass to bits (−log2 p)."""
    return float(-log10_prob / LOG10_2)


def sequence_bits(
    compressor: KenLMCompressor,
    symbol_seq: Sequence[int],
    *,
    bos: bool = True,
    eos: bool = True,
) -> float:
    """Cross-entropy bits of an idea-symbol sequence under KenLM."""
    if len(symbol_seq) == 0:
        return 0.0
    tokens = tokens_from_symbols(symbol_seq)
    state = compressor._kenlm.State()
    if bos:
        compressor._model.BeginSentenceWrite(state)
    else:
        compressor._model.NullContextWrite(state)
    total_log10 = 0.0
    for tok in tokens:
        out = compressor._kenlm.State()
        total_log10 += compressor._model.BaseScore(state, tok, out)
        state = out
    if eos:
        out = compressor._kenlm.State()
        total_log10 += compressor._model.BaseScore(state, "</s>", out)
    return log10_to_bits(total_log10)


def context_augmented_bits(
    compressor: KenLMCompressor,
    y_symbols: Sequence[int],
    probe_symbols: Sequence[int],
) -> float:
    """
    Marginal bits of the probe idea-symbol sequence when KenLM state is first
    advanced through y (teacher-forced prefix conditioning).
    """
    if len(probe_symbols) == 0:
        return 0.0

    state = compressor._kenlm.State()
    compressor._model.BeginSentenceWrite(state)
    for tok in tokens_from_symbols(y_symbols):
        out = compressor._kenlm.State()
        compressor._model.BaseScore(state, tok, out)
        state = out

    total_log10 = 0.0
    for tok in tokens_from_symbols(probe_symbols):
        out = compressor._kenlm.State()
        total_log10 += compressor._model.BaseScore(state, tok, out)
        state = out
    out = compressor._kenlm.State()
    total_log10 += compressor._model.BaseScore(state, "</s>", out)
    return log10_to_bits(total_log10)


def _collect_ngrams(sequences: Iterable[Sequence[str]], order: int):
    counts: list[Counter] = [Counter() for _ in range(order)]
    context_totals: list[Counter] = [Counter() for _ in range(order)]
    continuations: list[dict[tuple[str, ...], set[str]]] = [
        defaultdict(set) for _ in range(order)
    ]
    n_tokens = 0

    def _add(gram: tuple[str, ...]) -> None:
        n = len(gram)
        if n < 1 or n > order:
            return
        ctx = gram[:-1]
        word = gram[-1]
        counts[n - 1][gram] += 1
        context_totals[n - 1][ctx] += 1
        continuations[n - 1][ctx].add(word)

    for seq in sequences:
        # Pad with BOS for higher-order contexts; EOS as predicted token.
        padded = ["<s>"] * (order - 1) + list(seq) + ["</s>"]
        # Emit BOS-prefix n-grams so KenLM context-closure holds
        # (every 3-gram context must appear as a 2-gram, including <s> <s>).
        for i in range(order - 1):
            for n in range(1, i + 2):
                _add(tuple(padded[i - n + 1 : i + 1]))
        for i in range(order - 1, len(padded)):
            for n in range(1, order + 1):
                if i - n + 1 < 0:
                    continue
                gram = tuple(padded[i - n + 1 : i + 1])
                _add(gram)
                if n == 1 and gram[0] not in {"<s>"}:
                    n_tokens += 1
    return counts, context_totals, continuations, n_tokens


def write_witten_bell_arpa(
    sequences: List[List[str]],
    order: int,
    out_path: Path,
) -> Path:
    """
    Write a Witten–Bell discounted ARPA file KenLM can load.

    Used when the KenLM `lmplz` binary is not available on PATH.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    counts, context_totals, continuations, n_tokens = _collect_ngrams(sequences, order)
    if n_tokens == 0:
        # Degenerate empty corpus: uniform-ish unigram over unk.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\\data\\\nngram 1=1\n\n\\1-grams:\n-99\t<unk>\n\n\\end\\\n",
            encoding="utf-8",
        )
        return out_path

    # Unigram vocabulary. KenLM requires both <s> and </s> as 1-grams.
    vocab_set = {w for (w,) in counts[0].keys()}
    vocab_set.update({"<s>", "</s>", "<unk>"})
    # Emit specials first, then the rest sorted for stability.
    specials = ["<s>", "</s>", "<unk>"]
    rest = sorted(w for w in vocab_set if w not in specials)
    vocab = specials + rest

    # Precompute probabilities and backoff weights.
    probs: list[dict[tuple[str, ...], float]] = [dict() for _ in range(order)]
    bows: list[dict[tuple[str, ...], float]] = [dict() for _ in range(order - 1)]

    # Unigrams: MLE over observed tokens (include </s>; give <s> a tiny mass).
    uni_total = sum(c for (w,), c in counts[0].items() if w != "<s>")
    uni_total = max(uni_total, 1)
    for w in vocab:
        if w == "<s>":
            # BOS is never "generated"; tiny placeholder prob for KenLM.
            probs[0][(w,)] = 1e-99
            continue
        c = counts[0].get((w,), 0)
        probs[0][(w,)] = (c / uni_total) if c > 0 else 1e-99

    for n in range(2, order + 1):
        for gram, c in counts[n - 1].items():
            ctx = gram[:-1]
            n_ctx = context_totals[n - 1][ctx]
            t_ctx = len(continuations[n - 1][ctx])
            # Witten–Bell: p = c / (N + T)
            probs[n - 1][gram] = c / (n_ctx + t_ctx)

        # Backoff weights for each context at order n-1.
        for ctx, n_ctx in context_totals[n - 1].items():
            t_ctx = len(continuations[n - 1][ctx])
            remaining = t_ctx / (n_ctx + t_ctx) if (n_ctx + t_ctx) > 0 else 1.0
            lower_mass = 0.0
            for w in continuations[n - 1][ctx]:
                if n == 2:
                    lower_gram = (w,)
                else:
                    lower_gram = ctx[1:] + (w,)
                lower_mass += probs[n - 2].get(lower_gram, probs[0].get((w,), 0.0))
            if lower_mass <= 0.0 or remaining <= 0.0:
                bow = 1.0
            else:
                bow = remaining / lower_mass
            bows[n - 2][ctx] = bow

    # KenLM closure: every n-gram's context must exist as an (n-1)-gram.
    for n in range(order, 1, -1):
        for gram in list(probs[n - 1].keys()):
            ctx = gram[:-1]
            if ctx and ctx not in probs[n - 2]:
                # Synthesize from lower-order / tiny mass.
                if n == 2:
                    probs[n - 2][ctx] = probs[0].get(ctx, 1e-99)
                else:
                    # ctx is length n-1; assign WB-ish mass from counts if any.
                    c = counts[n - 2].get(ctx, 0)
                    parent = ctx[:-1]
                    n_parent = context_totals[n - 2].get(parent, 0)
                    t_parent = len(continuations[n - 2].get(parent, set()))
                    denom = max(n_parent + t_parent, 1)
                    probs[n - 2][ctx] = (c / denom) if c > 0 else 1e-99
                if n - 2 < len(bows) and ctx not in bows[n - 2]:
                    bows[n - 2][ctx] = 1.0

    def log10p(p: float) -> float:
        return math.log10(max(p, 1e-99))

    lines: list[str] = ["\\data\\"]
    for n in range(1, order + 1):
        # Count entries we will emit.
        if n == 1:
            n_entries = len(vocab)
        else:
            n_entries = len(probs[n - 1])
        lines.append(f"ngram {n}={n_entries}")
    lines.append("")

    # 1-grams with backoff for empty context / unigram contexts used as bigram ctx.
    lines.append("\\1-grams:")
    for w in vocab:
        p = probs[0][(w,)]
        bow = bows[0].get((w,), 1.0) if order >= 2 else None
        if bow is None or order < 2:
            lines.append(f"{log10p(p):.6f}\t{w}")
        else:
            lines.append(f"{log10p(p):.6f}\t{w}\t{log10p(bow):.6f}")
    lines.append("")

    for n in range(2, order + 1):
        lines.append(f"\\{n}-grams:")
        for gram, p in sorted(probs[n - 1].items()):
            gram_str = " ".join(gram)
            if n < order:
                bow = bows[n - 1].get(gram, 1.0)
                lines.append(f"{log10p(p):.6f}\t{gram_str}\t{log10p(bow):.6f}")
            else:
                lines.append(f"{log10p(p):.6f}\t{gram_str}")
        lines.append("")

    lines.append("\\end\\")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def train_kenlm(
    symbol_sequences: List[List[int]],
    out_arpa: Path,
    order: int = 3,
    *,
    prefer_lmplz: bool = True,
) -> Path:
    """
    Train a KenLM-loadable ARPA model on idea-symbol sequences.

    Prefers the official `lmplz` binary when present; otherwise writes a
    Witten–Bell ARPA file that `kenlm.Model` can load.
    """
    token_seqs = [tokens_from_symbols(seq) for seq in symbol_sequences if seq]
    out_arpa = Path(out_arpa)
    out_arpa.parent.mkdir(parents=True, exist_ok=True)

    lmplz = shutil.which("lmplz") if prefer_lmplz else None
    if lmplz and token_seqs:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            with open(corpus, "w", encoding="utf-8") as f:
                for seq in token_seqs:
                    f.write(" ".join(seq) + "\n")
            # --discount_fallback helps tiny corpora / sparse idea vocabs.
            cmd = [
                lmplz,
                "-o",
                str(order),
                "--discount_fallback",
                "--skip_symbols",
            ]
            with open(corpus, "rb") as stdin, open(out_arpa, "wb") as stdout:
                proc = subprocess.run(
                    cmd, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE
                )
            if proc.returncode != 0:
                # Fall back to pure-Python ARPA.
                write_witten_bell_arpa(token_seqs, order, out_arpa)
            else:
                # Optional binary compaction.
                build_binary = shutil.which("build_binary")
                if build_binary:
                    klm = out_arpa.with_suffix(".klm")
                    bproc = subprocess.run(
                        [build_binary, str(out_arpa), str(klm)],
                        stderr=subprocess.PIPE,
                    )
                    if bproc.returncode == 0 and klm.exists():
                        return klm
        return out_arpa

    write_witten_bell_arpa(token_seqs, order, out_arpa)
    return out_arpa


def load_kenlm_compressor(path: str | Path, order: int) -> KenLMCompressor:
    return KenLMCompressor(model_path=Path(path), order=order)
