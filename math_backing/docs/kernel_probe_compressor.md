# Kernel / Parzen ProbeCompressor

Lean `ProbeCompressor.corpusDeformationGain` only needs a code-length
\(N(\cdot\mid\mathcal{H})\). Soft order-1 SoftCountLM still failed the geometry
link (RD-SOFT-01: \(R^2\approx 0.017\)) because soft assignments route through a
frozen \(K\)-way alphabet and barely move on large \(\mathcal{H}\).

## Continuous alphabet

Idea units remain MiniLM embeddings \(e\in\mathbb{R}^d\). No codebook for \(R_D\).

Gaussian Parzen density on the corpus bank \(\mathcal{H}\):

\[
\hat p_{\mathcal{H}}(e)=\frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}}
\exp\Bigl(-\frac{\|e-h\|_2^2}{2\sigma^2}\Bigr).
\]

Probe code length:

\[
N(s'\mid\mathcal{H})=-\sum_t\log_2\max\bigl(\hat p_{\mathcal{H}}(e'_t),\varepsilon\bigr).
\]

Deformation is unchanged:

\[
D(y,\mathcal{H},\mathcal{P})=\sum_{s'\in\mathcal{P}}\bigl(N(s'\mid\mathcal{H})-N(s'\mid\mathcal{H}\cup\{y\})\bigr).
\]

Efficient update:

\[
\hat p_{\mathcal{H}\cup\{y\}}(e)=
\frac{|\mathcal{H}|\,\hat p_{\mathcal{H}}(e)+\sum_j K_\sigma(e,y_j)}{|\mathcal{H}|+|y|}.
\]

Bandwidth \(\sigma\) and gate \(\delta_D\) are calibrated on **negatives only**
(exact H member / shuffle / pad). Never fit to human/LLM labels. Banks may be
subsampled (`max_bank`) with a fixed seed for tractability.

## Status

Implemented in `creativegainbench.metrics.kernel_probe_ce` /
`compute_kernel_deformation`. Inspected under RD-KERNEL-01.
