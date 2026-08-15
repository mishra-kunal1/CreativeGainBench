# RD-KERNEL-01 — Human vs LLM discrimination (kernel R_D)

Goal: **discriminate** human vs LLM (either direction), and verify the separation is a genuine construct, not a length / idea-count artifact. We do not force human>LLM.

## Per-model separation

| Model | n | AUC(H>M) | Direction | Separation | CV acc r_d | CV acc len | r_d gain | AUC len-resid |
|-------|---|----------|-----------|-----------|-----------|-----------|----------|---------------|
| gemma2:2b | 1489 | 0.268 | model_higher | 0.465 | 0.675 | 0.559 | +0.116 | 0.273 |
| mistral:latest | 1489 | 0.218 | model_higher | 0.563 | 0.709 | 0.606 | +0.103 | 0.229 |
| llama3.1:8b | 1489 | 0.299 | model_higher | 0.402 | 0.637 | 0.535 | +0.102 | 0.292 |
| phi4:14b | 1489 | 0.214 | model_higher | 0.572 | 0.701 | 0.522 | +0.179 | 0.205 |

## Length-confound controls (gemma2:2b)

- Human idea-count mean **28.9** vs model **21.5** — humans are *longer*, yet score lower r_d, so length does not explain the direction.
- Pooled corr(r_d, length) = -0.037.
- Class CV accuracy: r_d **0.675** vs length-only **0.559** (r_d adds **+0.116**).
- AUC after residualizing r_d on length: **0.273** (separation survives length control).

### Matched idea-count quintiles (gemma2:2b)

| Band | sym range | n | AUC(H>M) | Δ(H−M) |
|------|-----------|---|----------|--------|
| 0 | 2–13 | 284 | 0.214 | -0.000103 |
| 1 | 13–16 | 283 | 0.278 | -9.68e-05 |
| 2 | 16–32 | 324 | 0.299 | -0.000113 |
| 3 | 32–39 | 251 | 0.293 | -0.000137 |
| 4 | 39–60 | 347 | 0.236 | -0.000196 |

## Per-domain (gemma2:2b)

| Domain | n | AUC(H>M) | Separation | Direction |
|--------|---|----------|-----------|-----------|
| 0 | 141 | 0.417 | 0.166 | model_higher |
| 1 | 94 | 0.223 | 0.554 | model_higher |
| 2 | 292 | 0.288 | 0.423 | model_higher |
| 3 | 120 | 0.190 | 0.621 | model_higher |
| 5 | 221 | 0.241 | 0.518 | model_higher |
| 6 | 124 | 0.128 | 0.743 | model_higher |
| 7 | 81 | 0.328 | 0.344 | model_higher |
| 8 | 158 | 0.212 | 0.575 | model_higher |
| 9 | 114 | 0.187 | 0.626 | model_higher |
| 10 | 144 | 0.288 | 0.424 | model_higher |

## Read

- Easiest LLM to detect: **phi4:14b**; hardest (closest to human manifold): **llama3.1:8b**.
- Direction is consistently **model_higher**: LLM poems sit *off* the human training manifold, so adding them deforms the Parzen density more. This is the discrimination signal — not a bug, and not length-driven.
- "Too good to be true" watch: a model whose separation → 0 would be indistinguishable from human on this construct; track `hardest_to_detect_model` over time.
