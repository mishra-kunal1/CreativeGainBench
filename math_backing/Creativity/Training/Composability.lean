import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Tactic

namespace Creativity.Training.Composability

/-!
B–D Pareto frontier for the optional additive diagnostic (no \(R_C\)).
The reported score `Rcreativity` is multiplicative and does not use this
object; the lemma certifies that B and D remain multi-objective.
-/

def jointlyDominatedBD
    (rB1 rB2 rD1 rD2 : ℝ) : Prop :=
  rB2 < rB1 ∧ rD2 ≤ rD1

theorem Bmax_not_jointly_dominated_BD
    {Y : Type*} (rewardB rewardD : Y → ℝ)
    (y_max : Y)
    (hmax : ∀ y : Y, rewardB y ≤ rewardB y_max) :
    ¬ ∃ y' : Y, rewardB y_max < rewardB y' ∧ rewardD y_max ≤ rewardD y' := by
  rintro ⟨y', hB, _hD⟩
  exact (not_lt.mpr (hmax y')) hB

theorem paretoFrontier_nonempty_BD
    {Y : Type*} [Fintype Y] [Nonempty Y]
    (rewardB rewardD : Y → ℝ) :
    ∃ y : Y, ¬ ∃ y' : Y, jointlyDominatedBD
      (rewardB y') (rewardB y) (rewardD y') (rewardD y) := by
  classical
  let s : Finset Y := Finset.univ
  have hs : s.Nonempty := Finset.univ_nonempty
  obtain ⟨ymax, _hy_mem, hsup⟩ := Finset.exists_mem_eq_sup' (s := s) hs rewardB
  use ymax
  rintro ⟨y', hdom⟩
  rcases hdom with ⟨hB, _hD⟩
  have hy'_le : rewardB y' ≤ rewardB ymax := by
    have hle : rewardB y' ≤ s.sup' hs rewardB := by
      exact Finset.le_sup' rewardB (Finset.mem_univ y')
    simpa [hsup] using hle
  exact (not_lt.mpr hy'_le) hB

theorem BD_tension_example :
    ∃ (rewardB rewardD : Fin 2 → ℝ),
      (∀ y, rewardB y ≤ rewardB 0) ∧
      (∀ y, rewardD y ≤ rewardD 1) ∧
      rewardB 1 < rewardB 0 ∧
      rewardD 0 < rewardD 1 := by
  let rewardB : Fin 2 → ℝ := fun y => if y = 0 then 1 else 0
  let rewardD : Fin 2 → ℝ := fun y => if y = 1 then 1 else 0
  refine ⟨rewardB, rewardD, ?_, ?_, ?_, ?_⟩
  · intro y
    fin_cases y <;> norm_num [rewardB]
  · intro y
    fin_cases y <;> norm_num [rewardD]
  · norm_num [rewardB]
  · norm_num [rewardD]

end Creativity.Training.Composability
