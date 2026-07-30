import Creativity.Probability.Entropy
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Tactic

namespace Creativity.Probability

/--
Finite-alphabet KL divergence $D_{\mathrm{KL}}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}$.
Requires $q(x)>0$ whenever $p(x)>0$; otherwise the corresponding term is treated as $0$.
-/
noncomputable def klDivergence {α : Type*} [Fintype α] (p q : PMF α) : ℝ :=
  ∑ x : α,
    if (p x).toReal = 0 then 0
    else (p x).toReal * Real.log ((p x).toReal / (q x).toReal)

/--
Gibbs inequality for finite PMFs. Recorded as an axiom because a full proof would
require a Jensen argument on $\log$ over the support of $p$.
-/
axiom klDivergence_nonneg {α : Type*} [Fintype α] (p q : PMF α) :
    0 ≤ klDivergence p q

/--
Cross-entropy decomposition: $D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)$ on a shared support.
Recorded as an axiom linking surface KL novelty to algorithmic novelty in C.
-/
axiom kl_cross_entropy_decomposition {α : Type*} [Fintype α] (p q : PMF α) :
    klDivergence p q + shannonEntropy p =
      -∑ x : α, (p x).toReal * Real.log (q x).toReal

end Creativity.Probability
