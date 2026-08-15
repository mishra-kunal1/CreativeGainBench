# Soft ProbeCompressor (instance of AbstractCompressor)

Lean `ProbeCompressor.corpusDeformationGain` is alphabet-agnostic: it only
needs a code-length functional \(N(\cdot\mid\mathcal{H})\). Hard poetry_v2
CountNgram used nearest-centroid VQ symbols and order-3 n-grams. That alphabet
discards continuous MiniLM directions that separate human vs LLM in idea
space (RD-INSPECT-01: EMB-PCA CV ≈ 0.94, hard \(R_D\) \(R^2\) on PC1–3 ≈ 0.017).

## Soft alphabet

Given frozen codebook centroids \(\{c_k\}_{k=1}^{K}\) and temperature \(\tau>0\),

\[
\pi_k(e)=\mathrm{softmax}_k\bigl(-\|e-c_k\|_2^2/\tau\bigr).
\]

As \(\tau\to 0\), \(\pi\) recovers hard VQ (one-hot nearest centroid).

## Soft unigram compressor (order-1)

Train fractional counts by summing soft rows over a domain corpus \(\mathcal{H}\).
With add-\(\varepsilon\) smoothing,

\[
p_{\mathcal{H}}(k)=\frac{n_k+\varepsilon}{\sum_j n_j+\varepsilon K}.
\]

Probe code length:

\[
N(s'\mid\mathcal{H})=-\sum_t\log_2\sum_k\pi_k(e'_t)\,p_{\mathcal{H}}(k).
\]

Deformation is unchanged:

\[
D(y,\mathcal{H},\mathcal{P})=\sum_{s'\in\mathcal{P}}\bigl(N(s'\mid\mathcal{H})-N(s'\mid\mathcal{H}\cup\{y\})\bigr),
\]

then \(\lambda_D\)-normalized and gated by \(\delta_D\) calibrated on **negatives
only** (exact H member / shuffle / pad). Temperature \(\tau\) is chosen the same
way. Never fit \(\tau\) or \(\delta_D\) to human/LLM labels.

## Status

Implemented in `creativegainbench.metrics.soft_count_ngram` /
`compute_soft_deformation`. Order-2 soft bigrams are optional follow-on; this
pass uses order-1 intentionally so geometry is not sparsified away. No new Lean
proofs this pass — SoftProbeCompressor is documented as an
`AbstractCompressor` instance over soft alphabets.
