# Experiment 1 — Human poems vs LLM generations (poetry_v2)

Paired discrimination experiment: for each reverse-engineered prompt, does CreativeGainBench’s measurement stack rank the human poem above a gemma2:2b completion?

Ground-truth assumption: published human poetry is more creative than a 2B model’s completion of the same prompt. Metrics that cannot clear this bar are not capturing the intended phenomenon.

**Metric version:** `poetry_v2`  
**Date:** 2026-08-03  
**Formal reference:** `math_backing/docs/bridged_validation.tex` (CUE, \(R_D\) / ProbeCompressor, gated \(R_{\mathrm{creativity}}\); \(R_C\) ignored)

---

## Data

| Field | Value |
|--------|--------|
| Source DB | Postgres `poems` (`postgres://poems:poems@localhost:5432/poems`) |
| Corpus size | 4,673 poems, all with `prompt` + `llm_output` |
| LLM generator | gemma2:2b via local Ollama (`max_tokens=1024`, temp=1.0) |
| Sources | poetrydb, gutenberg_poetry_corpus, gutenberg (+ 3 poetry_foundation_kaggle) |
| Pairing | Same prompt → human `body` vs `llm_output`; texts clipped to 4,000 chars |

### Splits (author-aware, per domain)

1. Embed each human poem body (MiniLM).
2. k-means into **k = 12** domain clusters (`seed=42`).
3. Per cluster, assign authors greedily into **probe / eval / train**, preferring small authors for probes (cap 3 poems/author in probe) so one prolific author cannot consume the probe budget.
4. Persist `domain_cluster` and `split` on the `poems` table.

| Split | n | Role |
|--------|---|------|
| train | 2,979 | Background corpus \(\mathcal{H}\) + idea codebook training |
| probe | 205 | Frozen per-domain probe sets \(\mathcal{P}\) (~18/domain when possible) |
| **eval** | **1,489** | Scored pairs (primary analysis set) |

Domains 4 and 11 had no eval poems after splitting; scoring used the 10 domains with eval rows.

Artifacts live under `src/creativegainbench/artifacts/poetry_v2/`.  
Scores also written to Postgres table `scores` with `metric_version='poetry_v2'`.

**Scripts:**
- `scripts/prepare_poetry_experiment.py` — cluster, split, fit codebook, build per-domain LMs, calibrate \(\delta_D\)
- `scripts/score_poems_v2.py` — score eval pairs and write JSONL / summary

**Outputs:**
- `data/evaluation/poems_v2_human_vs_llm.jsonl`
- `data/evaluation/poems_v2_human_vs_llm.summary.json`

---

## Idea pipeline (shared by \(R_D\))

| Step | Implementation |
|------|----------------|
| Segmentation | `poetry_line_splitter` — line / stanza breaks (not prose sentence-only splits) |
| Encoding | Frozen MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) |
| Quantization | k-means idea codebook, \(K=512\), fit on **train** idea spans only |
| Symbols | Sequence of cluster IDs per poem (eval means: **37.6** human / **27.4** LLM symbols) |

---

## Metrics — how each was calculated

Canonical score from the paper (not the training objective):

\[
R_{\mathrm{creativity}}
=
\mathbf{1}[\mathrm{CUE}>0]
\cdot
\mathbf{1}[R_D>\delta_D]
\cdot
\bigl(\mathrm{CUE}\cdot(1+\alpha\,R_B^{\to A})+\lambda_G G_k\bigr)
\]

In this experiment: \(\lambda_G=0\) (no MAS), \(R_B\) dropped from the product (saturated in v1), and continuous components are reported alongside gates.

### \(R_D\) — probe-relative structural deformation (primary)

**Definition (Lean `ProbeCompressor` / bridged_validation):**

\[
D(y,\mathcal{H},\mathcal{P})
=
\sum_{s'\in\mathcal{P}}
\bigl(
N(s'\mid\mathcal{H})
-
N(s'\mid\mathcal{H}\cup\{y\})
\bigr)
\]

where \(N(s'\mid\mathcal{H})\) is the cross-entropy bit cost of probe \(s'\) under an n-gram LM trained on background corpus \(\mathcal{H}\).

**Implementation (`count_ngram.py` / `deformation.py`):**
- Per domain: \(\mathcal{H}\) = train poems in that cluster; \(\mathcal{P}\) = probe poems in that cluster.
- Witten–Bell count n-gram LM (order 3) over idea-cluster symbols.
- Score probes under base LM; clone LM, add \(y\)’s symbol sequence, rescore probes; difference = raw \(R_D\) (bits).
- **Normalize** by the paper’s \(\lambda_D\) denominator:

\[
R_D^{\mathrm{norm}}
=
\frac{R_D^{\mathrm{raw}}}{|\mathcal{P}|\cdot\log|\Sigma|\cdot\mathbb{E}[|s'|]}
\]

so domains with different probe counts / lengths are comparable.

**Not used:** the old KenLM “prefix-condition on last \(n-1\) tokens” proxy (v1), which only depended on the last two idea symbols.

**Gate:** \(1[R_D^{\mathrm{norm}} > \delta_D]\) with \(\delta_D\) calibrated per domain (held-out eval vs train copies / paraphrases). Calibration was **non-separable** (negatives scored above positives), so reported gate rates at that threshold are **0%** on both sides. Interpret continuous \(R_D^{\mathrm{norm}}\) (and optional “positive deformation” rate) instead.

**Coverage:** all **1,489** eval pairs (CPU-only).

### CUE — creative update efficiency (secondary)

\[
\mathrm{CUE}(y,A,T) = \frac{\Delta_{\mathrm{Brier}}}{|y|_{\mathrm{bits}}}
\]

**Implementation (`cue_receiver.py` + `outcome_annotator.py`):**
1. Receiver (gemma2:2b) elicits a **prior** over \(Z = \{\)`novel_structure`, `fluent_paraphrase`, `clear_utility`, `low_quality`\(\}\) from the prompt alone.
2. Receiver elicits a **posterior** after seeing \(y\) (clipped to 2,500 chars).
3. Realized outcome \(z^*\) from an **external** MiniLM nearest-exemplar annotator (hand-crafted exemplars per label) — **not** the receiver’s own classification (avoids the v1 circularity).
4. \(\Delta_{\mathrm{Brier}} = \max(0,\; \mathrm{Brier}(\mathrm{prior},z^*) - \mathrm{Brier}(\mathrm{posterior},z^*))\); divide by UTF-8 bit length of \(y\).

**Gate:** \(1[\mathrm{CUE}>0]\).

**Coverage:** stratified subsample of **200** eval pairs.

### \(R_B^{\to A}\) — receiver expansion (diagnostic only)

Soft-cluster entropy of a receiver conditioned on \(y\), averaged over a task battery, normalized by \(\log K\). **Not scored** in poetry_v2 (v1 was saturated near 1.0 for both sides).

### \(G_k\) — interaction gain

MAS-only; no multi-agent human baseline. \(\lambda_G = 0\).

### \(R_C\)

Ignored (not part of canonical \(R_{\mathrm{creativity}}\)).

### Composite \(R_{\mathrm{creativity}}\)

On the CUE subsample:

\[
R_{\mathrm{creativity}}
=
\mathbf{1}[\mathrm{CUE}>0]
\cdot
\mathbf{1}[R_D^{\mathrm{norm}}>\delta_D]
\cdot \mathrm{CUE}
\]

With \(\delta_D\) non-separable, the D-gate never opens → composite **0** on both sides.

---

## Pre-registered endpoints (plan)

1. **Primary:** paired human−LLM divergence in \(\lambda_D\)-normalized \(R_D\) (mean + upper quantiles / tail share), full eval \(n\).
2. **Secondary:** CUE and gate-opening rates with external \(z^*\).
3. **Composite:** both-gates rate / \(R_{\mathrm{creativity}}\) (blocked here by \(\delta_D\)).

Significance: paired bootstrap 95% CI on mean(human − LLM); “significant” if CI excludes 0.

---

## Results

### \(R_D\) (n = 1,489)

| Statistic | Human | LLM |
|-----------|------:|----:|
| Mean \(R_D^{\mathrm{norm}}\) | 3.95×10⁻⁴ | 3.58×10⁻⁴ |
| Median | 5.18×10⁻⁵ | 9.28×10⁻⁵ |
| q90 | 1.04×10⁻³ | 8.67×10⁻⁴ |
| q95 | 1.83×10⁻³ | 1.52×10⁻³ |
| Mean raw bits | 1.99 | 1.64 |
| Share above pooled q90 (tail) | **11.2%** | 8.8% |
| Share with \(R_D^{\mathrm{norm}}>0\) | 55.9% | **64.7%** |
| Gate pass (calibrated \(\delta_D\)) | 0% | 0% |

| Paired comparison | Value |
|-------------------|------:|
| Mean Δ (human − LLM) | +3.75×10⁻⁵ |
| Bootstrap 95% CI | [−4.02×10⁻⁵, +1.18×10⁻⁴] |
| Significant? | **No** |
| Human wins / LLM wins / ties | 720 / 769 / 0 |

**Per-domain** (significant at 95% CI):

| Higher side | Domains (n) |
|-------------|-------------|
| Human | 0 (141), 2 (292), 9 (114), 10 (144) |
| LLM | 6 (124), 7 (81), 8 (158) |
| Tie | 1, 3, 5 |

Overall discrimination is a coin flip; the human edge appears as a **heavier right tail** and in **some domains**, not as a pooled mean shift.

### CUE (n = 200, external \(z^*\))

| Statistic | Human | LLM |
|-----------|------:|----:|
| Mean CUE | 2.02×10⁻⁵ | 1.60×10⁻⁵ |
| Gate pass (CUE > 0) | 42% | 46.5% |
| Both gates open | 0% | 0% |

| Paired comparison | Value |
|-------------------|------:|
| Mean Δ (human − LLM) | +4.22×10⁻⁶ |
| Bootstrap 95% CI | [−1.99×10⁻⁶, +1.09×10⁻⁵] |
| Significant? | **No** |

**External outcome labels:**

| Label | Human | LLM |
|-------|------:|----:|
| novel_structure | 70 | 57 |
| fluent_paraphrase | 90 | 103 |
| clear_utility | 39 | 40 |
| low_quality | 1 | 0 |

Relative to the earlier v1 sanity run (self-classified outcomes, LLM gate 21% vs human 13% significant), external \(z^*\) removes the LLM’s significant CUE-gate advantage. Humans are labeled `novel_structure` more often; the LLM `fluent_paraphrase` more often.

### Composite \(R_{\mathrm{creativity}}\)

Mean 0.0 on both sides (D-gate closed for every text under the prepare-time \(\delta_D\)).

---

## Interpretation

1. **True deformation \(R_D\) is live and domain-sensitive**, but does **not** yet clear a pooled human > LLM test at n=1,489. Tail share and 4/10 domains favor humans; 3/10 favor the LLM — heterogeneity matters more than a single mean.
2. **\(\delta_D\) calibration failed** (train copies / paraphrases scored above held-out humans under Witten–Bell updates), so gated \(R_{\mathrm{creativity}}\) is uninformative in this run. Prefer continuous \(R_D^{\mathrm{norm}}\).
3. **CUE with external outcomes** is better aligned with the VOI story than v1’s circular self-labeling; direction weakly favors humans but is not significant. Magnitudes remain tiny (\(\sim10^{-5}\)) because of bit-length normalization on long poems.
4. Decision criteria from the plan: **not confirmed** as a global human > LLM creativity detector yet; closest reading is **partial / domain-conditional** support for \(R_D\), with calibration and compressor behavior still open issues.

---

## Reproduce

```bash
# Prepare splits + poetry_v2 artifacts (once)
python scripts/prepare_poetry_experiment.py --k 12 --seed 42

# Score (R_D on all eval; CUE on subsample)
python scripts/score_poems_v2.py --cue-sample 200 --seed 42
```

Requires: Postgres poems DB populated, Ollama with `gemma2:2b` for CUE, Python env with package deps (CountNgram ProbeCompressor \(R_D\) path is pure Python).
