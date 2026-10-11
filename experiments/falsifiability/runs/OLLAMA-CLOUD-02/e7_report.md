# E7 causal contribution controls

**Passed:** False
**Receiver:** `gemma4:31b`
**Items:** 24

PairedMeanDiff DIFFERENT + positive estimate + BY p_adj < alpha for {cross, random, irrelevant}; mean |bits_arm/bits_matched-1| < 0.20

| Control | n | estimate | CI | p | p_adj BY | verdict | pass |
|---------|---|----------|----|---|----------|---------|------|
| cross | 24 | +0.0002 | [+5.261e-05, +0.0004379] | 0.03398 | 0.09345 | different | False |
| random | 24 | +0.0001784 | [+5.084e-05, +0.0004359] | 0.06497 | 0.1191 | indeterminate | False |
| irrelevant | 24 | +0.0002777 | [+0.0001467, +0.0005141] | 0.004498 | 0.02474 | different | True |

Length mean |bits_arm/bits_matched − 1| = 0.1091 (pass True, tol 0.20)

_Co-primary brier_delta contrasts are in the JSON report._

