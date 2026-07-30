import Mathlib.Tactic

namespace Creativity.MAS.OInfo

noncomputable def oInformation3
    (H12 H13 H23 H123 : ℝ) : ℝ :=
  H12 + H13 + H23 - H123

theorem oInformation3_negative_of_pair_sum_lt
    (H12 H13 H23 H123 : ℝ)
    (h : H12 + H13 + H23 < H123) :
    oInformation3 H12 H13 H23 H123 < 0 := by
  unfold oInformation3
  linarith

theorem oInformation3_positive_of_triple_lt_pair_sum
    (H12 H13 H23 H123 : ℝ)
    (h : H123 < H12 + H13 + H23) :
    0 < oInformation3 H12 H13 H23 H123 := by
  unfold oInformation3
  linarith

theorem equalMarginals_gain_implies_synergy
    (h H_J : ℝ)
    (_hGain : h < H_J)
    (hSign : 3 * (H_J - h) < H_J) :
    oInformation3 (H_J - h) (H_J - h) (H_J - h) H_J < 0 := by
  unfold oInformation3
  linarith

theorem negativeGain_implies_redundancy
    (h H_J : ℝ)
    (_hCollapse : H_J < h)
    (hSign : H_J < 3 * (h - H_J)) :
    0 < oInformation3 (h - H_J) (h - H_J) (h - H_J) H_J := by
  unfold oInformation3
  linarith

end Creativity.MAS.OInfo
