# E7 causal contribution controls

**Passed:** False
**Receiver:** `gemma4:31b`
**Items:** 24

PairedMeanDiff DIFFERENT + positive estimate + BY p_adj < alpha for {cross, random, irrelevant}; mean |bits_arm/bits_matched-1| < 0.20

| Control | n | estimate | CI | p | p_adj BY | verdict | pass |
|---------|---|----------|----|---|----------|---------|------|
| cross | 24 | +2.059e-05 | [-7.166e-06, +6.958e-05] | 0.4318 | 0.7916 | indeterminate | False |
| random | 24 | +0.0005834 | [+0.000495, +0.0006439] | 0.0004998 | 0.001374 | different | True |
| irrelevant | 24 | +0.0006839 | [+0.0006775, +0.000696] | 0.0004998 | 0.001374 | different | True |

Length mean |bits_arm/bits_matched − 1| = 0.0023 (pass True, tol 0.20)

_Co-primary brier_delta contrasts are in the JSON report._

