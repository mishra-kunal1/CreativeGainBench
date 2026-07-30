import Creativity.C.Compression
import Mathlib.Data.List.Basic
import Mathlib.Tactic

namespace Creativity.D

open Creativity.C

/--
A compressor extended with probe-set monotonicity: incorporating an output into
the corpus weakly reduces the compression cost of probe strings from the same
domain. This formalizes probe-relative structural influence (corpus deformation
gain on a fixed finite probe set), not transformational creativity by definition.
When the probe set is domain-representative (benchmark hold-out protocol), high
deformation gain correlates with transformational creativity in Boden's sense.
-/
class ProbeCompressor (α : Type*) extends AbstractCompressor α where
  incorporation_helps : ∀ (s corpus probe : List α),
    novelty probe (corpus ++ s) ≤ novelty probe corpus
  append_self_invariant : ∀ (corpus probe : List α),
    novelty probe (corpus ++ corpus) = novelty probe corpus

namespace ProbeCompressor

noncomputable def probeNoveltyDelta {α : Type*} [ProbeCompressor α]
    (probe corpus s : List α) : ℝ :=
  novelty probe corpus - novelty probe (corpus ++ s)

lemma probeNoveltyDelta_nonneg {α : Type*} [ProbeCompressor α]
    (probe corpus s : List α) :
    0 ≤ probeNoveltyDelta probe corpus s := by
  unfold probeNoveltyDelta
  linarith [incorporation_helps s corpus probe]

/--
Corpus deformation gain: total compression improvement on a finite probe set
from incorporating output $s$ into corpus $\mathcal{H}$.

In the benchmark, `probes` is a fixed hold-out sample (10--20\% evaluation
withhold, stratified by task category, $|P| \in [100,200]$, frozen with
recorded seed); the D theorems quantify over any such fixed finite `probes`.
-/
noncomputable def corpusDeformationGain {α : Type*} [ProbeCompressor α]
    (s corpus : List α) (probes : List (List α)) : ℝ :=
  probes.foldl (fun acc p => acc + probeNoveltyDelta p corpus s) 0

lemma corpusDeformationGain_nonneg {α : Type*} [ProbeCompressor α]
    (s corpus : List α) (probes : List (List α)) :
    0 ≤ corpusDeformationGain s corpus probes := by
  suffices h : ∀ (ps : List (List α)) (acc : ℝ), 0 ≤ acc →
      0 ≤ ps.foldl (fun a p => a + probeNoveltyDelta p corpus s) acc by
    simpa [corpusDeformationGain] using h probes 0 (by norm_num)
  intro ps acc hacc
  induction ps generalizing acc with
  | nil => simpa using hacc
  | cons p ps ih =>
    simp only [List.foldl]
    exact ih (acc + probeNoveltyDelta p corpus s)
      (add_nonneg hacc (probeNoveltyDelta_nonneg p corpus s))

noncomputable def rewardD {α : Type*} [ProbeCompressor α]
    (s corpus : List α) (probes : List (List α))
    (utility_val threshold : ℝ) : ℝ := by
  classical
  exact if threshold ≤ utility_val then corpusDeformationGain s corpus probes else 0

end ProbeCompressor

end Creativity.D
